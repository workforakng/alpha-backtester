# engine.py — Simulation orchestrator with Brownian motion tick interpolation

import numpy as np
import pandas as pd
from typing import Generator, Tuple
from config import (
    TICKS_PER_CANDLE, GBM_SIGMA_SCALE,
    RISK_FREE_RATE, DAYS_TO_EXPIRY,
    OPTION_PREMIUM_PCT, IMPLIED_VOLATILITY,
    SLIPPAGE_PCT, BROKERAGE_PCT,
    TRAILING_STOP_PCT
)
from scipy.stats import norm


# ── Black-Scholes Premium Calculator ─────────────────────────────────────────

def black_scholes_premium(
    spot: float,
    strike: float,
    days: int,
    r: float,
    sigma: float,
    option_type: str = "call"
) -> float:
    """
    Compute Black-Scholes European option premium.
    Falls back to flat OPTION_PREMIUM_PCT if inputs are degenerate.
    """
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
        # Floor at intrinsic value or minimum tick
        return max(price, spot * 0.001)
    except Exception:
        return spot * OPTION_PREMIUM_PCT


def get_option_premium(spot: float, option_type: str = "call") -> float:
    """ATM option premium: strike = spot."""
    return black_scholes_premium(
        spot=spot,
        strike=spot,
        days=DAYS_TO_EXPIRY,
        r=RISK_FREE_RATE,
        sigma=IMPLIED_VOLATILITY,
        option_type=option_type,
    )


# ── Brownian Motion Tick Generator ───────────────────────────────────────────

def interpolate_candle_ticks(
    open_: float,
    high: float,
    low: float,
    close: float,
    n_ticks: int = TICKS_PER_CANDLE,
    sigma_scale: float = GBM_SIGMA_SCALE,
) -> np.ndarray:
    """
    Interpolate a single OHLC candle into n_ticks micro-prices
    using Geometric Brownian Motion with boundary enforcement.
    """
    if n_ticks <= 1:
        return np.array([close])

    # Estimate per-tick volatility from candle range
    candle_range = max(high - low, 1e-6)
    sigma_tick   = (candle_range / open_) * sigma_scale / np.sqrt(n_ticks)

    # GBM path: S(t+dt) = S(t) * exp((mu - σ²/2)*dt + σ*√dt*Z)
    dt   = 1.0 / n_ticks
    mu   = np.log(close / open_) / n_ticks   # drift to land at close
    Z    = np.random.standard_normal(n_ticks)
    log_returns = (mu - 0.5 * sigma_tick**2) * dt + sigma_tick * np.sqrt(dt) * Z
    path = open_ * np.exp(np.cumsum(log_returns))

    # Soft-clamp to [low, high] range
    path = np.clip(path, low * 0.998, high * 1.002)
    # Force last tick to exactly match close
    path[-1] = close
    return path


def candle_stream(
    df: pd.DataFrame,
    n_ticks: int = TICKS_PER_CANDLE
) -> Generator[Tuple[pd.Timestamp, float, int], None, None]:
    """
    Yields (timestamp, tick_price, candle_index) for all candles in df.
    Timestamps are linearly interpolated within each 1-minute candle.
    """
    for i, (ts, row) in enumerate(df.iterrows()):
        ticks = interpolate_candle_ticks(
            open_=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            n_ticks=n_ticks,
        )
        candle_start_ns = pd.Timestamp(ts).value
        candle_end_ns   = candle_start_ns + int(60e9)  # 60 seconds in ns
        tick_times = np.linspace(candle_start_ns, candle_end_ns, n_ticks, endpoint=False)

        for j, (tick_ns, price) in enumerate(zip(tick_times, ticks)):
            yield pd.Timestamp(tick_ns, unit="ns", tz="UTC"), float(price), i


# ── Trade Execution & P&L ─────────────────────────────────────────────────────

class TradeRecord:
    __slots__ = [
        "ticker", "direction", "entry_price", "premium",
        "qty", "entry_time", "exit_price", "exit_time",
        "pnl", "status", "peak_pnl", "stop_price"
    ]

    def __init__(self, ticker, direction, entry_price, premium, qty, entry_time):
        self.ticker      = ticker
        self.direction   = direction      # "CALL" or "PUT"
        self.entry_price = entry_price    # spot at entry
        self.premium     = premium        # option premium paid
        self.qty         = qty            # number of lots
        self.entry_time  = entry_time
        self.exit_price  = None
        self.exit_time   = None
        self.pnl         = 0.0
        self.status      = "OPEN"         # OPEN | WIN | LOSS | STOPPED
        self.peak_pnl    = 0.0
        self.stop_price  = None           # trailing stop trigger price

    def cost_basis(self) -> float:
        slippage = self.entry_price * SLIPPAGE_PCT
        brokerage = self.premium * BROKERAGE_PCT
        return (self.premium + slippage + brokerage) * self.qty

    def update_trailing_stop(self, current_pnl: float):
        if current_pnl > self.peak_pnl:
            self.peak_pnl = current_pnl

    def is_stopped(self, current_pnl: float) -> bool:
        if self.peak_pnl > 0:
            drawdown = (self.peak_pnl - current_pnl) / abs(self.peak_pnl + 1e-9)
            return drawdown > TRAILING_STOP_PCT
        return False

    def compute_pnl(self, exit_spot: float) -> float:
        """Compute unrealised/realised P&L for an option position."""
        exit_premium = get_option_premium(exit_spot, self.direction.lower())
        raw_pnl = (exit_premium - self.premium) * self.qty
        slippage_cost = exit_spot * SLIPPAGE_PCT * self.qty
        brokerage_cost = exit_premium * BROKERAGE_PCT * self.qty
        return raw_pnl - slippage_cost - brokerage_cost

    def close(self, exit_price: float, exit_time, reason: str = "signal"):
        self.exit_price = exit_price
        self.exit_time  = exit_time
        self.pnl        = self.compute_pnl(exit_price)
        self.status     = "WIN" if self.pnl >= 0 else "LOSS"
        if reason == "trailing_stop":
            self.status = "STOPPED"
        return self.pnl


def calculate_position_size(wallet: float, spot: float, premium: float) -> int:
    """Calculate number of option lots to buy given wallet and trade allocation."""
    from config import TRADE_ALLOCATION
    capital_for_trade = wallet * TRADE_ALLOCATION
    if premium <= 0:
        return 0
    lots = int(capital_for_trade / (premium + spot * SLIPPAGE_PCT))
    return max(lots, 1)
