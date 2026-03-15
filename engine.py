# engine.py — Simulation core: Black-Scholes, GBM ticks, Kelly sizing, ATR stops

import numpy as np
import pandas as pd
from typing import Generator, Tuple
from scipy.stats import norm
from config import (
    TICKS_PER_CANDLE, GBM_SIGMA_SCALE,
    RISK_FREE_RATE, DAYS_TO_EXPIRY,
    OPTION_PREMIUM_PCT, IMPLIED_VOLATILITY,
    SLIPPAGE_PCT, BROKERAGE_PCT,
    TRAILING_STOP_PCT, HARD_STOP_PCT, PROFIT_TARGET_PCT,
    TRADE_ALLOCATION,
    KELLY_FRACTION, KELLY_MIN_ALLOC, KELLY_MAX_ALLOC,
    ATR_STOP_MULT, ATR_TARGET_MULT, ATR_STOPS_ENABLED,
)


# ──────────────────────── BLACK-SCHOLES ─────────────────────────

def black_scholes_premium(
    spot: float, strike: float, days: int,
    r: float, sigma: float, option_type: str = "call"
) -> float:
    T = max(days / 365.0, 1e-6)
    if spot <= 0 or strike <= 0 or sigma <= 0:
        return spot * OPTION_PREMIUM_PCT
    try:
        d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type.lower() == "call":
            price = spot * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = strike * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        return max(price, spot * 0.001)
    except Exception:
        return spot * OPTION_PREMIUM_PCT


def get_option_premium(spot: float, option_type: str = "call") -> float:
    """ATM option: strike = spot."""
    return black_scholes_premium(
        spot=spot, strike=spot, days=DAYS_TO_EXPIRY,
        r=RISK_FREE_RATE, sigma=IMPLIED_VOLATILITY,
        option_type=option_type,
    )


# ─────────────────────── KELLY CRITERION SIZING ────────────────────────

def kelly_allocation(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Fractional Kelly Criterion: f* = (p*b - q) / b
    where p=win_rate, q=1-p, b=avg_win/avg_loss (reward:risk)
    Clamped to [KELLY_MIN_ALLOC, KELLY_MAX_ALLOC].
    Returns fraction of wallet to allocate.
    """
    if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0:
        return TRADE_ALLOCATION
    q = 1.0 - win_rate
    b = avg_win / avg_loss
    f_star = (win_rate * b - q) / b
    f_kelly = max(f_star, 0.0) * KELLY_FRACTION
    return float(np.clip(f_kelly, KELLY_MIN_ALLOC, KELLY_MAX_ALLOC))


def calculate_position_size(
    wallet: float,
    spot: float,
    premium: float,
    win_rate: float = 0.5,
    avg_win: float = 0.0,
    avg_loss: float = 0.0,
) -> int:
    """Kelly-adjusted position sizing."""
    alloc   = kelly_allocation(win_rate, avg_win, avg_loss)
    capital = wallet * alloc
    if premium <= 0:
        return 0
    lots = int(capital / (premium + spot * SLIPPAGE_PCT))
    return max(lots, 1)


# ─────────────────────── GBM TICK GENERATOR ────────────────────────

def interpolate_candle_ticks(
    open_: float, high: float, low: float, close: float,
    n_ticks: int = TICKS_PER_CANDLE,
    sigma_scale: float = GBM_SIGMA_SCALE,
) -> np.ndarray:
    if n_ticks <= 1:
        return np.array([close])
    candle_range = max(high - low, 1e-6)
    sigma_tick   = (candle_range / open_) * sigma_scale / np.sqrt(n_ticks)
    dt           = 1.0 / n_ticks
    mu           = np.log(close / open_) / n_ticks
    Z            = np.random.standard_normal(n_ticks)
    log_returns  = (mu - 0.5 * sigma_tick**2) * dt + sigma_tick * np.sqrt(dt) * Z
    path         = open_ * np.exp(np.cumsum(log_returns))
    path         = np.clip(path, low * 0.998, high * 1.002)
    path[-1]     = close
    return path


def candle_stream(
    df: pd.DataFrame, n_ticks: int = TICKS_PER_CANDLE
) -> Generator[Tuple[pd.Timestamp, float, int], None, None]:
    if len(df.index) >= 2:
        inferred_step_ns = max(int((pd.Timestamp(df.index[1]) - pd.Timestamp(df.index[0])).value), int(1e9))
    else:
        inferred_step_ns = int(60e9)

    for i, (ts, row) in enumerate(df.iterrows()):
        ticks = interpolate_candle_ticks(
            open_=float(row["Open"]), high=float(row["High"]),
            low=float(row["Low"]),  close=float(row["Close"]),
            n_ticks=n_ticks,
        )
        candle_start_ns = pd.Timestamp(ts).value
        candle_end_ns   = candle_start_ns + inferred_step_ns
        tick_times      = np.linspace(candle_start_ns, candle_end_ns, n_ticks, endpoint=False)
        for j, (tick_ns, price) in enumerate(zip(tick_times, ticks)):
            yield pd.Timestamp(tick_ns, unit="ns", tz="UTC"), float(price), i


# ─────────────────────── TRADE RECORD ─────────────────────────────

class TradeRecord:
    __slots__ = [
        "ticker", "direction", "entry_price", "premium",
        "qty", "entry_time", "exit_price", "exit_time",
        "pnl", "status", "exit_reason",
        "peak_pnl", "atr_stop", "atr_target", "cost",
    ]

    def __init__(self, ticker, direction, entry_price, premium,
                 qty, entry_time, atr=None):
        self.ticker       = ticker
        self.direction    = direction
        self.entry_price  = entry_price
        self.premium      = premium
        self.qty          = qty
        self.entry_time   = entry_time
        self.exit_price   = None
        self.exit_time    = None
        self.pnl          = 0.0
        self.status       = "OPEN"
        self.exit_reason  = "—"
        self.peak_pnl     = 0.0
        self.cost         = (premium + entry_price * SLIPPAGE_PCT + premium * BROKERAGE_PCT) * qty

        # ATR-based dynamic stop & target (in P&L space)
        if atr and atr > 0 and ATR_STOPS_ENABLED:
            # Convert ATR price move to approx P&L via delta ~0.5 ATM
            delta_approx  = 0.5
            atr_pnl       = atr * delta_approx * qty
            self.atr_stop   = -atr_pnl * ATR_STOP_MULT
            self.atr_target = atr_pnl * ATR_TARGET_MULT
        else:
            self.atr_stop   = -self.cost * HARD_STOP_PCT / max(self.cost, 1) * self.cost
            self.atr_target = self.cost * PROFIT_TARGET_PCT

    def cost_basis(self) -> float:
        return self.cost

    def update_trailing_stop(self, current_pnl: float):
        if current_pnl > self.peak_pnl:
            self.peak_pnl = current_pnl

    def is_stopped(self, current_pnl: float) -> bool:
        # 1. ATR hard stop
        if current_pnl <= self.atr_stop:
            return True
        # 2. Hard stop-loss from cost
        if current_pnl <= -self.cost * HARD_STOP_PCT * 10:
            return True
        # 3. Trailing stop from peak
        if self.peak_pnl > 0:
            drawdown = (self.peak_pnl - current_pnl) / (abs(self.peak_pnl) + 1e-9)
            if drawdown > TRAILING_STOP_PCT:
                return True
        return False

    def is_target_hit(self, current_pnl: float) -> bool:
        return current_pnl >= self.atr_target

    def compute_pnl(self, exit_spot: float) -> float:
        exit_premium    = get_option_premium(exit_spot, self.direction.lower())
        raw_pnl         = (exit_premium - self.premium) * self.qty
        slippage_cost   = exit_spot * SLIPPAGE_PCT * self.qty
        brokerage_cost  = exit_premium * BROKERAGE_PCT * self.qty
        return raw_pnl - slippage_cost - brokerage_cost

    def close(self, exit_price: float, exit_time, reason: str = "signal"):
        self.exit_price  = exit_price
        self.exit_time   = exit_time
        self.pnl         = self.compute_pnl(exit_price)
        self.exit_reason = reason
        if reason == "trailing_stop":    self.status = "STOPPED"
        elif reason == "atr_stop":       self.status = "STOPPED"
        elif reason == "hard_stop":      self.status = "STOPPED"
        elif reason == "profit_target":  self.status = "WIN"
        else:                            self.status = "WIN" if self.pnl >= 0 else "LOSS"
        return self.pnl


# ─────────────────────── PERFORMANCE METRICS ──────────────────────

def compute_metrics(trades: list, initial_wallet: float) -> dict:
    """
    Institutional-grade metrics used by quant funds:
    Sharpe, Sortino, Profit Factor, Max Drawdown, Calmar, Expectancy.
    """
    if not trades:
        return {}

    pnls = [t.pnl for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl    = sum(pnls)
    win_rate     = len(wins) / len(pnls)
    avg_win      = float(np.mean(wins))   if wins   else 0.0
    avg_loss     = float(np.mean(losses)) if losses else 0.0
    profit_factor = sum(wins) / (abs(sum(losses)) + 1e-9)
    expectancy   = win_rate * avg_win + (1 - win_rate) * avg_loss

    # Sharpe (annualised, assume each trade ~5min)
    if len(pnls) > 1:
        returns   = np.array(pnls) / initial_wallet
        sharpe    = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 78))
        downside  = np.std([r for r in returns if r < 0] or [0])
        sortino   = float(np.mean(returns) / (downside + 1e-9) * np.sqrt(252 * 78))
    else:
        sharpe = sortino = 0.0

    # Max Drawdown
    cumulative  = initial_wallet + np.cumsum(pnls)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns   = (rolling_max - cumulative) / (rolling_max + 1e-9)
    max_dd      = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    calmar = (total_pnl / initial_wallet) / (max_dd + 1e-9)

    return {
        "total_pnl":      round(total_pnl, 2),
        "win_rate":       round(win_rate * 100, 1),
        "avg_win":        round(avg_win, 2),
        "avg_loss":       round(avg_loss, 2),
        "profit_factor":  round(profit_factor, 3),
        "expectancy":     round(expectancy, 2),
        "sharpe":         round(sharpe, 3),
        "sortino":        round(sortino, 3),
        "max_drawdown":   round(max_dd * 100, 2),
        "calmar":         round(calmar, 3),
        "total_trades":   len(trades),
        "wins":           len(wins),
        "losses":         len(losses),
    }
