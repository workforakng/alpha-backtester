# indicators.py — Technical indicator computation module

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False

from config import (
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    SUPERTREND_PERIOD, SUPERTREND_MULT,
    SMA_PERIOD, EMA_FAST, EMA_SLOW
)


# ── Fallback pure-numpy implementations ──────────────────────────────────────

def _ema_numpy(series: pd.Series, period: int) -> pd.Series:
    result = np.full(len(series), np.nan)
    if len(series) < period:
        return pd.Series(result, index=series.index)
    sma_init = series.iloc[:period].mean()
    k = 2.0 / (period + 1)
    result[period - 1] = sma_init
    for i in range(period, len(series)):
        result[i] = series.iloc[i] * k + result[i - 1] * (1 - k)
    return pd.Series(result, index=series.index)


def _macd_fallback(close: pd.Series):
    ema_fast   = _ema_numpy(close, MACD_FAST)
    ema_slow   = _ema_numpy(close, MACD_SLOW)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema_numpy(macd_line.dropna(), MACD_SIGNAL).reindex(close.index)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def _supertrend_fallback(df: pd.DataFrame, period: int, mult: float):
    hl2 = (df["High"] + df["Low"]) / 2
    tr  = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    supertrend = pd.Series(np.nan, index=df.index)
    direction  = pd.Series(1, index=df.index)   # 1=bullish, -1=bearish

    for i in range(period, len(df)):
        prev_upper = upper.iloc[i-1] if not np.isnan(upper.iloc[i-1]) else upper.iloc[i]
        prev_lower = lower.iloc[i-1] if not np.isnan(lower.iloc[i-1]) else lower.iloc[i]

        upper.iloc[i] = min(upper.iloc[i], prev_upper) if df["Close"].iloc[i-1] <= prev_upper else upper.iloc[i]
        lower.iloc[i] = max(lower.iloc[i], prev_lower) if df["Close"].iloc[i-1] >= prev_lower else lower.iloc[i]

        if df["Close"].iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif df["Close"].iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

        supertrend.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]

    return supertrend, direction


# ── Public API ────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicators on OHLCV DataFrame.
    Returns df with added columns:
      macd, macd_signal, macd_hist,
      supertrend, supertrend_dir,
      sma200, ema20, ema50
    """
    out = df.copy()
    close = out["Close"]

    # ── MACD ──
    if _TA_AVAILABLE:
        try:
            macd_df = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
            if macd_df is not None and not macd_df.empty:
                cols = macd_df.columns.tolist()
                out["macd"]        = macd_df[cols[0]].values
                out["macd_signal"] = macd_df[cols[2]].values
                out["macd_hist"]   = macd_df[cols[1]].values
            else:
                raise ValueError("Empty MACD result")
        except Exception:
            out["macd"], out["macd_signal"], out["macd_hist"] = _macd_fallback(close)
    else:
        out["macd"], out["macd_signal"], out["macd_hist"] = _macd_fallback(close)

    # ── Supertrend ──
    if _TA_AVAILABLE:
        try:
            st_df = ta.supertrend(out["High"], out["Low"], close,
                                   length=SUPERTREND_PERIOD, multiplier=SUPERTREND_MULT)
            if st_df is not None and not st_df.empty:
                st_cols = st_df.columns.tolist()
                out["supertrend"]     = st_df[st_cols[0]].values
                out["supertrend_dir"] = st_df[st_cols[2]].values  # direction column
            else:
                raise ValueError("Empty Supertrend result")
        except Exception:
            out["supertrend"], out["supertrend_dir"] = _supertrend_fallback(
                out, SUPERTREND_PERIOD, SUPERTREND_MULT)
    else:
        out["supertrend"], out["supertrend_dir"] = _supertrend_fallback(
            out, SUPERTREND_PERIOD, SUPERTREND_MULT)

    # ── Moving Averages ──
    if _TA_AVAILABLE:
        try:
            out["sma200"] = ta.sma(close, length=SMA_PERIOD).values
            out["ema20"]  = ta.ema(close, length=EMA_FAST).values
            out["ema50"]  = ta.ema(close, length=EMA_SLOW).values
        except Exception:
            out["sma200"] = close.rolling(SMA_PERIOD).mean().values
            out["ema20"]  = _ema_numpy(close, EMA_FAST).values
            out["ema50"]  = _ema_numpy(close, EMA_SLOW).values
    else:
        out["sma200"] = close.rolling(SMA_PERIOD).mean().values
        out["ema20"]  = _ema_numpy(close, EMA_FAST).values
        out["ema50"]  = _ema_numpy(close, EMA_SLOW).values

    return out


def get_latest_signals(df: pd.DataFrame) -> dict:
    """Extract the most recent indicator values from a computed DataFrame."""
    if len(df) < 2:
        return {}
    row  = df.iloc[-1]
    prev = df.iloc[-2]
    return {
        "close":         float(row["Close"]),
        "macd":          float(row.get("macd", np.nan)),
        "macd_signal":   float(row.get("macd_signal", np.nan)),
        "macd_hist":     float(row.get("macd_hist", np.nan)),
        "macd_cross_up": (
            float(prev.get("macd", 0)) < float(prev.get("macd_signal", 0)) and
            float(row.get("macd", 0))  > float(row.get("macd_signal", 0))
        ),
        "macd_cross_dn": (
            float(prev.get("macd", 0)) > float(prev.get("macd_signal", 0)) and
            float(row.get("macd", 0))  < float(row.get("macd_signal", 0))
        ),
        "supertrend":    float(row.get("supertrend", np.nan)),
        "st_bullish":    float(row.get("supertrend_dir", 0)) > 0,
        "sma200":        float(row.get("sma200", np.nan)),
        "ema20":         float(row.get("ema20", np.nan)),
        "ema50":         float(row.get("ema50", np.nan)),
        "above_sma200":  float(row["Close"]) > float(row.get("sma200", 0) or 0),
        "ema_cross_up":  (
            float(prev.get("ema20", 0)) < float(prev.get("ema50", 0)) and
            float(row.get("ema20", 0))  > float(row.get("ema50", 0))
        ),
        "ema_cross_dn":  (
            float(prev.get("ema20", 0)) > float(prev.get("ema50", 0)) and
            float(row.get("ema20", 0))  < float(row.get("ema50", 0))
        ),
    }
