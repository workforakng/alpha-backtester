import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
from indicators import compute_indicators, get_latest_signals
from engine import TradeRecord, get_option_premium, calculate_position_size, candle_stream
from config import MAX_CONCURRENT_TRADES, TICKS_PER_CANDLE


@dataclass
class SignalResult:
    direction: Optional[str]
    score: float
    max_score: float
    strength: str
    reasons: List[str] = field(default_factory=list)


def evaluate_confluence(s: dict) -> SignalResult:
    """
    19-signal confluence engine. Min 60% score required to trade.
    Max possible score = 23 points.
    """
    bull = 0.0
    bear = 0.0
    rb = []
    rs = []

    # 1. MACD crossover (2) + histogram (1)
    if s.get("macd_cross_up"):
        bull += 2; rb.append("MACD\u2705cross")
    elif s.get("macd_cross_dn"):
        bear += 2; rs.append("MACD\u2705cross")
    if s.get("macd_bull"):
        bull += 1; rb.append("MACD+hist")
    else:
        bear += 1; rs.append("MACD-hist")

    # 2. Supertrend (2)
    if s.get("st_bullish"):
        bull += 2; rb.append("ST\u2705bull")
    else:
        bear += 2; rs.append("ST\u2705bear")

    # 3. SMA 50(1) 100(1) 200(2) + golden cross state(1) + trigger(1)
    if s.get("above_sma50"):
        bull += 1; rb.append("\u25b2SMA50")
    else:
        bear += 1; rs.append("\u25bcSMA50")
    if s.get("above_sma100"):
        bull += 1; rb.append("\u25b2SMA100")
    else:
        bear += 1; rs.append("\u25bcSMA100")
    if s.get("above_sma200"):
        bull += 2; rb.append("\u25b2SMA200")
    else:
        bear += 2; rs.append("\u25bcSMA200")
    if s.get("sma50_above_sma200"):
        bull += 1; rb.append("GoldenX-state")
    else:
        bear += 1; rs.append("DeathX-state")
    if s.get("sma50_cross_sma200"):
        bull += 1; rb.append("GoldenX\u2705trigger")

    # 4. EMA 15/19 cross (2)
    if s.get("ema15_cross_up"):
        bull += 2; rb.append("EMA15/19\u2705up")
    elif s.get("ema15_cross_dn"):
        bear += 2; rs.append("EMA15/19\u2705dn")

    # 5. EMA 20/50 cross (2)
    if s.get("ema20_cross_up"):
        bull += 2; rb.append("EMA20/50\u2705up")
    elif s.get("ema20_cross_dn"):
        bear += 2; rs.append("EMA20/50\u2705dn")

    # 6. EMA19 vs EMA50 state (1)
    if s.get("ema19_above_ema50"):
        bull += 1; rb.append("EMA19\u25b2EMA50")
    else:
        bear += 1; rs.append("EMA19\u25bcEMA50")

    # 7. RSI extreme (2) + zone (1)
    if s.get("rsi_oversold"):
        bull += 2; rb.append(f"RSI\u2705oversold({s.get('rsi',0):.0f})")
    elif s.get("rsi_overbought"):
        bear += 2; rs.append(f"RSI\u2705overbought({s.get('rsi',0):.0f})")
    if s.get("rsi_bull"):
        bull += 1; rb.append("RSI-zone-bull")
    elif s.get("rsi_bear"):
        bear += 1; rs.append("RSI-zone-bear")

    # 8. Bollinger touch (2) + squeeze (1)
    if s.get("at_bb_lower"):
        bull += 2; rb.append("BB\u2705lower")
    elif s.get("at_bb_upper"):
        bear += 2; rs.append("BB\u2705upper")
    if s.get("bb_squeeze"):
        if s.get("macd_bull"):
            bull += 1; rb.append("BB-squeeze-bull")
        else:
            bear += 1; rs.append("BB-squeeze-bear")

    # 9. VWAP (2)
    if s.get("above_vwap"):
        bull += 2; rb.append("\u25b2VWAP")
    else:
        bear += 2; rs.append("\u25bcVWAP")

    max_score = 23.0
    threshold = max_score * 0.60

    def strength(score):
        pct = score / max_score
        if pct >= 0.80: return "STRONG"
        if pct >= 0.65: return "MODERATE"
        return "WEAK"

    if bull >= threshold and bull > bear:
        return SignalResult("CALL", bull, max_score, strength(bull), rb)
    elif bear >= threshold and bear > bull:
        return SignalResult("PUT", bear, max_score, strength(bear), rs)
    return SignalResult(None, max(bull, bear), max_score, "WEAK", [])


class TickerStrategy:
    def __init__(self, ticker, df):
        self.ticker        = ticker
        self.raw_df        = df
        self.indicator_df  = compute_indicators(df)
        self.open_trades: List[TradeRecord]   = []
        self.closed_trades: List[TradeRecord] = []

    def _rolling_signals(self, up_to):
        sub = self.indicator_df.iloc[:up_to + 1]
        return get_latest_signals(sub) if len(sub) >= 3 else {}

    def _can_open(self):
        return len(self.open_trades) < MAX_CONCURRENT_TRADES

    def _open_trade(self, direction, spot, ts, wallet):
        premium = get_option_premium(spot, direction.lower())
        qty     = calculate_position_size(wallet, spot, premium)
        if qty <= 0 or premium * qty > wallet * 0.25:
            return None
        trade = TradeRecord(ticker=self.ticker, direction=direction,
                            entry_price=spot, premium=premium,
                            qty=qty, entry_time=ts)
        self.open_trades.append(trade)
        return trade

    def _check_exits(self, spot, ts, signal):
        exited = []
        for t in self.open_trades:
            pnl = t.compute_pnl(spot)
            t.update_trailing_stop(pnl)
            reason = None
            if t.is_stopped(pnl):
                reason = "trailing_stop"
            elif t.direction == "CALL" and signal.direction == "PUT":
                reason = "signal_flip"
            elif t.direction == "PUT" and signal.direction == "CALL":
                reason = "signal_flip"
            if reason:
                t.close(spot, ts, reason)
                self.closed_trades.append(t)
                exited.append(t)
        for t in exited:
            self.open_trades.remove(t)
        return exited

    def process_tick(self, ts, price, candle_idx, wallet):
        signals = self._rolling_signals(candle_idx)
        if not signals:
            return [], [], None
        signal  = evaluate_confluence(signals)
        closed  = self._check_exits(price, ts, signal)
        new_trades = []
        if signal.direction and signal.strength in ("STRONG", "MODERATE") and self._can_open():
            existing = {t.direction for t in self.open_trades}
            if signal.direction not in existing:
                t = self._open_trade(signal.direction, price, ts, wallet)
                if t:
                    new_trades.append(t)
        return new_trades, closed, signal

    def get_open_pnl(self, price):
        return sum(t.compute_pnl(price) for t in self.open_trades)

    def get_total_realised_pnl(self):
        return sum(t.pnl for t in self.closed_trades)

    def force_close_all(self, price, ts):
        for t in list(self.open_trades):
            t.close(price, ts, "eod_close")
            self.closed_trades.append(t)
        self.open_trades.clear()
