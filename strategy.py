# strategy.py — Confluence engine + institutional risk management

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List

from indicators import compute_indicators, get_latest_signals
from engine import (
    TradeRecord, get_option_premium,
    calculate_position_size, candle_stream,
    compute_metrics,
)
from config import (
    MAX_CONCURRENT_TRADES, TICKS_PER_CANDLE,
    MAX_DRAWDOWN_PCT, COOLDOWN_CANDLES,
    MAX_TRADES_PER_TICKER, REGIME_FILTER_ENABLED, ADX_THRESHOLD,
)


# ──────────────────────── SIGNAL RESULT ─────────────────────────────

@dataclass
class SignalResult:
    direction:  Optional[str]     # "CALL" | "PUT" | None
    score:      float
    max_score:  float
    strength:   str               # "STRONG" | "MODERATE" | "WEAK"
    reasons:    List[str] = field(default_factory=list)
    regime_ok:  bool = True       # False = filtered by regime check


# ───────────────────── CONFLUENCE ENGINE ───────────────────────────

def evaluate_confluence(s: dict) -> SignalResult:
    """
    26-signal confluence engine with regime gating.
    Max possible score = 30 points.
    Threshold = 60% (18 pts) to open a trade.
    """
    bull = 0.0
    bear = 0.0
    rb   = []   # bull reasons
    rs   = []   # bear reasons

    # ── REGIME GATE: ADX ──────────────────────────────
    # Only score direction if market is trending
    trending     = s.get("trending", True)
    strong_trend = s.get("strong_trend", False)
    if REGIME_FILTER_ENABLED and not trending:
        return SignalResult(None, 0, 30, "WEAK", ["ADX<20 — ranging market"], regime_ok=False)

    # Bonus for strong trend (+1)
    if strong_trend:
        if s.get("di_bull"): bull += 1; rb.append("ADX-strong▲")
        else:                 bear += 1; rs.append("ADX-strong▼")

    # ── 1. MACD crossover (2) + histogram (1) + acceleration (1) ──
    if s.get("macd_cross_up"):
        bull += 2; rb.append("MACD✅cross")
    elif s.get("macd_cross_dn"):
        bear += 2; rs.append("MACD✅cross")
    if s.get("macd_bull"):   bull += 1; rb.append("MACD+hist")
    else:                     bear += 1; rs.append("MACD-hist")
    if s.get("macd_accel"):  bull += 1; rb.append("MACD-accel")

    # ── 2. Supertrend (2) + flip bonus (1) ──
    if s.get("st_bullish"):    bull += 2; rb.append("ST✅bull")
    else:                       bear += 2; rs.append("ST✅bear")
    if s.get("st_flip_bull"): bull += 1; rb.append("ST✅flip▲")
    if s.get("st_flip_bear"): bear += 1; rs.append("ST✅flip▼")

    # ── 3. SMA stack (50=1, 100=1, 200=2, golden=1, trigger=1) ──
    if s.get("above_sma50"):        bull += 1; rb.append("▲SMA50")
    else:                            bear += 1; rs.append("▼SMA50")
    if s.get("above_sma100"):       bull += 1; rb.append("▲SMA100")
    else:                            bear += 1; rs.append("▼SMA100")
    if s.get("above_sma200"):       bull += 2; rb.append("▲SMA200")
    else:                            bear += 2; rs.append("▼SMA200")
    if s.get("sma50_above_sma200"): bull += 1; rb.append("GoldenX")
    else:                            bear += 1; rs.append("DeathX")
    if s.get("sma50_cross_sma200"): bull += 1; rb.append("GoldenX✅")

    # ── 4. EMA 15/19 cross (2) ──
    if s.get("ema15_cross_up"):  bull += 2; rb.append("EMA15/19✅")
    elif s.get("ema15_cross_dn"): bear += 2; rs.append("EMA15/19✅")

    # ── 5. EMA 20/50 cross (2) ──
    if s.get("ema20_cross_up"):  bull += 2; rb.append("EMA20/50✅")
    elif s.get("ema20_cross_dn"): bear += 2; rs.append("EMA20/50✅")

    # ── 6. EMA19 vs EMA50 state (1) ──
    if s.get("ema19_above_ema50"): bull += 1; rb.append("EMA19▲EMA50")
    else:                            bear += 1; rs.append("EMA19▼EMA50")

    # ── 7. RSI extreme (2) + zone (1) + mid-cross (1) ──
    if s.get("rsi_oversold"):      bull += 2; rb.append(f"RSI-OS({s.get('rsi',0):.0f})")
    elif s.get("rsi_overbought"): bear += 2; rs.append(f"RSI-OB({s.get('rsi',0):.0f})")
    if s.get("rsi_bull"):          bull += 1; rb.append("RSI-zone▲")
    elif s.get("rsi_bear"):        bear += 1; rs.append("RSI-zone▼")
    if s.get("rsi_cross_50_up"): bull += 1; rb.append("RSI>50")
    if s.get("rsi_cross_50_dn"): bear += 1; rs.append("RSI<50")

    # ── 8. Stochastic RSI cross (2) + zone (1) ──
    if s.get("stoch_cross_up"):   bull += 2; rb.append("StochRSI✅▲")
    elif s.get("stoch_cross_dn"): bear += 2; rs.append("StochRSI✅▼")
    if s.get("stoch_oversold"):   bull += 1; rb.append("StochOS")
    elif s.get("stoch_overbought"): bear += 1; rs.append("StochOB")

    # ── 9. Bollinger touch (2) + expansion (1) ──
    if s.get("at_bb_lower"):     bull += 2; rb.append("BB▼-touch")
    elif s.get("at_bb_upper"): bear += 2; rs.append("BB▲-touch")
    if s.get("bb_expansion"):
        if s.get("macd_bull"): bull += 1; rb.append("BB-exp▲")
        else:                   bear += 1; rs.append("BB-exp▼")

    # ── 10. VWAP (1 state + 1 cross) ──
    if s.get("above_vwap"):       bull += 1; rb.append("▲VWAP")
    else:                          bear += 1; rs.append("▼VWAP")
    if s.get("vwap_cross_up"):   bull += 1; rb.append("VWAP✅cross")
    if s.get("vwap_cross_dn"):   bear += 1; rs.append("VWAP✅cross")

    # ── 11. OBV (volume confirmation) (1) ──
    if s.get("obv_bull"):  bull += 1; rb.append("OBV-acc")
    elif s.get("obv_bear"): bear += 1; rs.append("OBV-dist")

    # ── 12. CCI (1) ──
    if s.get("cci_bull"):      bull += 1; rb.append("CCI+")
    else:                       bear += 1; rs.append("CCI-")
    if s.get("cci_oversold"): bull += 1; rb.append("CCI-OS")
    if s.get("cci_overbought"): bear += 1; rs.append("CCI-OB")

    # ── 13. Williams %R (1) ──
    if s.get("willr_oversold"):    bull += 1; rb.append("WillR-OS")
    elif s.get("willr_overbought"): bear += 1; rs.append("WillR-OB")

    # ── 14. Candle body strength (1) ──
    if s.get("strong_body"):
        if s.get("macd_bull"): bull += 1; rb.append("BodyMom▲")
        else:                   bear += 1; rs.append("BodyMom▼")

    max_score = 30.0
    threshold = max_score * 0.60   # 18 points = 60%

    def strength(score):
        p = score / max_score
        if p >= 0.80: return "STRONG"
        if p >= 0.65: return "MODERATE"
        return "WEAK"

    if bull >= threshold and bull > bear:
        return SignalResult("CALL", bull, max_score, strength(bull), rb)
    elif bear >= threshold and bear > bull:
        return SignalResult("PUT", bear, max_score, strength(bear), rs)
    return SignalResult(None, max(bull, bear), max_score, "WEAK", [])


# ───────────────────── TICKER STRATEGY ──────────────────────────────

class TickerStrategy:
    """
    Per-ticker stateful strategy runner with:
    - Kelly-adjusted sizing
    - ATR dynamic stops & profit targets
    - Regime filter (ADX)
    - Cooldown after stops
    - Max trades per session (anti-overtrading)
    - Portfolio drawdown guard
    - Rolling performance tracking for live Kelly updates
    """

    def __init__(self, ticker: str, df: pd.DataFrame):
        self.ticker        = ticker
        self.raw_df        = df
        self.indicator_df  = compute_indicators(df)
        self.open_trades:   List[TradeRecord] = []
        self.closed_trades: List[TradeRecord] = []
        self._cooldown      = 0        # candles remaining in cooldown
        self._last_candle   = -1       # last processed candle index
        self._recent_pnls:  deque = deque(maxlen=20)   # rolling last-20 trades

    # ── helpers

    def _rolling_signals(self, up_to: int) -> dict:
        sub = self.indicator_df.iloc[:up_to + 1]
        return get_latest_signals(sub) if len(sub) >= 3 else {}

    def _can_open(self) -> bool:
        if len(self.open_trades) >= MAX_CONCURRENT_TRADES:
            return False
        if len(self.closed_trades) >= MAX_TRADES_PER_TICKER:
            return False
        if self._cooldown > 0:
            return False
        return True

    def _live_kelly_params(self):
        """Rolling win-rate and avg win/loss from last 20 trades."""
        if len(self._recent_pnls) < 5:
            return 0.5, 0.0, 0.0
        wins   = [p for p in self._recent_pnls if p > 0]
        losses = [p for p in self._recent_pnls if p <= 0]
        wr     = len(wins) / len(self._recent_pnls)
        aw     = float(np.mean(wins))   if wins   else 0.0
        al     = float(np.mean(np.abs(losses))) if losses else 0.0
        return wr, aw, al

    def _open_trade(self, direction: str, spot: float, ts,
                    wallet: float, atr: float) -> Optional[TradeRecord]:
        wr, aw, al = self._live_kelly_params()
        premium    = get_option_premium(spot, direction.lower())
        qty        = calculate_position_size(wallet, spot, premium, wr, aw, al)
        cost       = (premium + spot * SLIPPAGE_PCT + premium * BROKERAGE_PCT) * qty
        if qty <= 0 or cost > wallet * 0.20:
            return None
        trade = TradeRecord(
            ticker=self.ticker, direction=direction,
            entry_price=spot, premium=premium,
            qty=qty, entry_time=ts, atr=atr,
        )
        self.open_trades.append(trade)
        return trade

    def _check_exits(self, spot: float, ts, signal: SignalResult) -> List[TradeRecord]:
        exited = []
        for t in self.open_trades:
            pnl    = t.compute_pnl(spot)
            t.update_trailing_stop(pnl)
            reason = None

            if t.is_target_hit(pnl):                           reason = "profit_target"
            elif t.is_stopped(pnl):                            reason = "trailing_stop"
            elif t.direction == "CALL" and signal and signal.direction == "PUT":  reason = "signal_flip"
            elif t.direction == "PUT"  and signal and signal.direction == "CALL": reason = "signal_flip"

            if reason:
                t.close(spot, ts, reason)
                self.closed_trades.append(t)
                self._recent_pnls.append(t.pnl)
                if reason in ("trailing_stop", "atr_stop", "hard_stop"):
                    self._cooldown = COOLDOWN_CANDLES
                exited.append(t)

        for t in exited:
            self.open_trades.remove(t)
        return exited

    # ── main entry point

    def process_tick(
        self, ts, price: float, candle_idx: int, wallet: float,
        portfolio_peak: float = 0.0,
    ):
        """
        Returns (new_trades, closed_trades, signal).
        portfolio_peak: overall portfolio peak wallet for drawdown guard.
        """
        # Tick cooldown per candle boundary
        if candle_idx != self._last_candle:
            if self._cooldown > 0:
                self._cooldown -= 1
            self._last_candle = candle_idx

        signals = self._rolling_signals(candle_idx)
        if not signals:
            return [], [], None

        signal = evaluate_confluence(signals)
        closed = self._check_exits(price, ts, signal)

        new_trades = []
        # Portfolio drawdown guard
        if portfolio_peak > 0:
            dd = (portfolio_peak - wallet) / portfolio_peak
            if dd > MAX_DRAWDOWN_PCT:
                return new_trades, closed, signal   # halt: drawdown breached

        if (signal.direction
                and signal.strength in ("STRONG", "MODERATE")
                and signal.regime_ok
                and self._can_open()):
            existing = {t.direction for t in self.open_trades}
            if signal.direction not in existing:
                atr = float(signals.get("atr", 0) or 0)
                t   = self._open_trade(signal.direction, price, ts, wallet, atr)
                if t:
                    new_trades.append(t)

        return new_trades, closed, signal

    # ── reporting

    def get_open_pnl(self, price: float) -> float:
        return sum(t.compute_pnl(price) for t in self.open_trades)

    def get_total_realised_pnl(self) -> float:
        return sum(t.pnl for t in self.closed_trades)

    def get_metrics(self, initial_wallet: float) -> dict:
        return compute_metrics(self.closed_trades, initial_wallet)

    def force_close_all(self, price: float, ts):
        for t in list(self.open_trades):
            t.close(price, ts, "eod_close")
            self.closed_trades.append(t)
            self._recent_pnls.append(t.pnl)
        self.open_trades.clear()
