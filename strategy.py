# strategy.py — Confluence signal engine for CALL/PUT trade generation

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
from indicators import compute_indicators, get_latest_signals
from engine import (
    TradeRecord, get_option_premium,
    calculate_position_size, candle_stream
)
from config import (
    MAX_CONCURRENT_TRADES, TICKS_PER_CANDLE,
    TRAILING_STOP_PCT
)


@dataclass
class SignalResult:
    direction: Optional[str]   # "CALL", "PUT", or None
    score: int                 # Confluence score (0-4)
    reasons: List[str] = field(default_factory=list)


def evaluate_confluence(signals: dict) -> SignalResult:
    """
    Score system — 4 indicators, each contributes 1 point:
      1. MACD crossover (bullish/bearish)
      2. Supertrend direction
      3. Price vs 200 SMA
      4. EMA 20/50 crossover

    CALL: score >= 3 all bullish
    PUT:  score >= 3 all bearish
    """
    bullish_score = 0
    bearish_score = 0
    reasons_bull  = []
    reasons_bear  = []

    # 1. MACD
    if signals.get("macd_cross_up"):
        bullish_score += 1
        reasons_bull.append("MACD✅")
    elif signals.get("macd_cross_dn"):
        bearish_score += 1
        reasons_bear.append("MACD✅")
    else:
        # Non-crossover: use histogram direction
        hist = signals.get("macd_hist", 0) or 0
        if hist > 0:
            bullish_score += 0.5
            reasons_bull.append("MACD~")
        elif hist < 0:
            bearish_score += 0.5
            reasons_bear.append("MACD~")

    # 2. Supertrend
    if signals.get("st_bullish"):
        bullish_score += 1
        reasons_bull.append("ST✅")
    else:
        bearish_score += 1
        reasons_bear.append("ST✅")

    # 3. Price vs 200 SMA
    if signals.get("above_sma200"):
        bullish_score += 1
        reasons_bull.append("SMA200✅")
    else:
        bearish_score += 1
        reasons_bear.append("SMA200✅")

    # 4. EMA 20/50 crossover or relative position
    if signals.get("ema_cross_up"):
        bullish_score += 1
        reasons_bull.append("EMA✅")
    elif signals.get("ema_cross_dn"):
        bearish_score += 1
        reasons_bear.append("EMA✅")
    else:
        ema20 = signals.get("ema20", 0) or 0
        ema50 = signals.get("ema50", 0) or 0
        if ema20 > ema50:
            bullish_score += 0.5
            reasons_bull.append("EMA~")
        elif ema20 < ema50:
            bearish_score += 0.5
            reasons_bear.append("EMA~")

    if bullish_score >= 3 and bullish_score > bearish_score:
        return SignalResult("CALL", int(bullish_score), reasons_bull)
    elif bearish_score >= 3 and bearish_score > bullish_score:
        return SignalResult("PUT", int(bearish_score), reasons_bear)
    return SignalResult(None, max(int(bullish_score), int(bearish_score)), [])


class TickerStrategy:
    """Stateful per-ticker strategy runner."""

    def __init__(self, ticker: str, df: pd.DataFrame):
        self.ticker      = ticker
        self.raw_df      = df
        self.indicator_df = compute_indicators(df)
        self.open_trades: List[TradeRecord] = []
        self.closed_trades: List[TradeRecord] = []
        self.candle_idx  = 0          # tracks current candle for rolling indicators

    def _rolling_signals(self, up_to_candle: int) -> dict:
        """Compute signals using data up to a rolling candle index."""
        sub = self.indicator_df.iloc[: up_to_candle + 1]
        if len(sub) < 2:
            return {}
        return get_latest_signals(sub)

    def _can_open_trade(self) -> bool:
        return len(self.open_trades) < MAX_CONCURRENT_TRADES

    def _open_trade(self, direction: str, spot: float, timestamp, wallet: float) -> Optional[TradeRecord]:
        premium = get_option_premium(spot, direction.lower())
        qty     = calculate_position_size(wallet, spot, premium)
        if qty <= 0 or premium * qty > wallet * 0.25:
            return None
        trade = TradeRecord(
            ticker=self.ticker,
            direction=direction,
            entry_price=spot,
            premium=premium,
            qty=qty,
            entry_time=timestamp,
        )
        self.open_trades.append(trade)
        return trade

    def _check_exits(self, spot: float, timestamp, signal: SignalResult):
        """Check all open trades for trailing stop or opposite signal exit."""
        exited = []
        for trade in self.open_trades:
            current_pnl = trade.compute_pnl(spot)
            trade.update_trailing_stop(current_pnl)

            should_exit = False
            reason      = "signal"

            if trade.is_stopped(current_pnl):
                should_exit = True
                reason = "trailing_stop"
            elif trade.direction == "CALL" and signal.direction == "PUT":
                should_exit = True
            elif trade.direction == "PUT" and signal.direction == "CALL":
                should_exit = True

            if should_exit:
                trade.close(spot, timestamp, reason)
                self.closed_trades.append(trade)
                exited.append(trade)

        for t in exited:
            self.open_trades.remove(t)
        return exited

    def process_tick(
        self,
        timestamp,
        price: float,
        candle_idx: int,
        wallet: float,
    ):
        """
        Main tick processor. Returns (new_trades, closed_trades, signal).
        """
        self.candle_idx = candle_idx
        signals = self._rolling_signals(candle_idx)
        if not signals:
            return [], [], None

        signal = evaluate_confluence(signals)
        closed = self._check_exits(price, timestamp, signal)

        new_trades = []
        if signal.direction and self._can_open_trade():
            # Avoid duplicate direction in open trades
            existing_dirs = {t.direction for t in self.open_trades}
            if signal.direction not in existing_dirs:
                trade = self._open_trade(signal.direction, price, timestamp, wallet)
                if trade:
                    new_trades.append(trade)

        return new_trades, closed, signal

    def get_open_pnl(self, current_price: float) -> float:
        return sum(t.compute_pnl(current_price) for t in self.open_trades)

    def get_total_realised_pnl(self) -> float:
        return sum(t.pnl for t in self.closed_trades)

    def force_close_all(self, price: float, timestamp):
        """Force-close all open positions at end of simulation."""
        for trade in list(self.open_trades):
            trade.close(price, timestamp, "eod_close")
            self.closed_trades.append(trade)
        self.open_trades.clear()
