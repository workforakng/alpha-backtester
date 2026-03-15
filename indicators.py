import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    _TA = True
except ImportError:
    _TA = False

from config import (
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    SUPERTREND_PERIOD, SUPERTREND_MULT,
    SMA_50, SMA_100, SMA_200,
    EMA_15, EMA_19, EMA_20, EMA_50,
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    BB_PERIOD, BB_STD, VWAP_ENABLED
)


def _ema(series, period):
    result = np.full(len(series), np.nan)
    if len(series) < period:
        return pd.Series(result, index=series.index)
    k = 2.0 / (period + 1)
    result[period - 1] = series.iloc[:period].mean()
    for i in range(period, len(series)):
        result[i] = series.iloc[i] * k + result[i-1] * (1 - k)
    return pd.Series(result, index=series.index)


def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd_fallback(close):
    fast = _ema(close, MACD_FAST)
    slow = _ema(close, MACD_SLOW)
    macd = fast - slow
    signal = _ema(macd.dropna(), MACD_SIGNAL).reindex(close.index)
    return macd, signal, macd - signal


def _supertrend(df, period, mult):
    hl2 = (df["High"] + df["Low"]) / 2
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    direction = pd.Series(1, index=df.index)
    supertrend = pd.Series(np.nan, index=df.index)
    for i in range(period, len(df)):
        pu = upper.iloc[i-1] if not np.isnan(upper.iloc[i-1]) else upper.iloc[i]
        pl = lower.iloc[i-1] if not np.isnan(lower.iloc[i-1]) else lower.iloc[i]
        upper.iloc[i] = min(upper.iloc[i], pu) if df["Close"].iloc[i-1] <= pu else upper.iloc[i]
        lower.iloc[i] = max(lower.iloc[i], pl) if df["Close"].iloc[i-1] >= pl else lower.iloc[i]
        if df["Close"].iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif df["Close"].iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
        supertrend.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]
    return supertrend, direction


def _vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()


def _bollinger(close, period=20, std=2.0):
    mid = close.rolling(period).mean()
    sd  = close.rolling(period).std()
    return mid - std*sd, mid, mid + std*sd


def compute_indicators(df):
    out = df.copy()
    c = out["Close"]

    # MACD
    if _TA:
        try:
            m = ta.macd(c, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
            cols = m.columns.tolist()
            out["macd"] = m[cols[0]].values
            out["macd_signal"] = m[cols[2]].values
            out["macd_hist"] = m[cols[1]].values
        except:
            out["macd"], out["macd_signal"], out["macd_hist"] = _macd_fallback(c)
    else:
        out["macd"], out["macd_signal"], out["macd_hist"] = _macd_fallback(c)

    # Supertrend
    if _TA:
        try:
            st = ta.supertrend(out["High"], out["Low"], c,
                               length=SUPERTREND_PERIOD, multiplier=SUPERTREND_MULT)
            cols = st.columns.tolist()
            out["supertrend"] = st[cols[0]].values
            out["supertrend_dir"] = st[cols[2]].values
        except:
            out["supertrend"], out["supertrend_dir"] = _supertrend(out, SUPERTREND_PERIOD, SUPERTREND_MULT)
    else:
        out["supertrend"], out["supertrend_dir"] = _supertrend(out, SUPERTREND_PERIOD, SUPERTREND_MULT)

    # SMAs
    out["sma50"]  = c.rolling(SMA_50).mean().values
    out["sma100"] = c.rolling(SMA_100).mean().values
    out["sma200"] = c.rolling(SMA_200).mean().values

    # EMAs
    for p, name in [(EMA_15,"ema15"),(EMA_19,"ema19"),(EMA_20,"ema20"),(EMA_50,"ema50")]:
        if _TA:
            try:
                out[name] = ta.ema(c, length=p).values
            except:
                out[name] = _ema(c, p).values
        else:
            out[name] = _ema(c, p).values

    # RSI
    if _TA:
        try:
            out["rsi"] = ta.rsi(c, length=RSI_PERIOD).values
        except:
            out["rsi"] = _rsi(c, RSI_PERIOD).values
    else:
        out["rsi"] = _rsi(c, RSI_PERIOD).values

    # Bollinger Bands
    out["bb_lower"], out["bb_mid"], out["bb_upper"] = _bollinger(c, BB_PERIOD, BB_STD)

    # VWAP
    if VWAP_ENABLED and "Volume" in out.columns:
        out["vwap"] = _vwap(out).values

    return out


def get_latest_signals(df):
    if len(df) < 3:
        return {}
    r    = df.iloc[-1]
    prev = df.iloc[-2]
    p2   = df.iloc[-3]
    c    = float(r["Close"])

    def f(col):  return float(r.get(col,   np.nan) or 0)
    def fp(col): return float(prev.get(col, np.nan) or 0)

    rsi_val = f("rsi")

    return {
        "close":               c,
        "macd":                f("macd"),
        "macd_signal":         f("macd_signal"),
        "macd_hist":           f("macd_hist"),
        "macd_cross_up":       fp("macd") < fp("macd_signal") and f("macd") > f("macd_signal"),
        "macd_cross_dn":       fp("macd") > fp("macd_signal") and f("macd") < f("macd_signal"),
        "macd_bull":           f("macd_hist") > 0,
        "st_bullish":          f("supertrend_dir") > 0,
        "sma50":               f("sma50"),
        "sma100":              f("sma100"),
        "sma200":              f("sma200"),
        "above_sma50":         c > f("sma50"),
        "above_sma100":        c > f("sma100"),
        "above_sma200":        c > f("sma200"),
        "sma50_above_sma200":  f("sma50") > f("sma200"),
        "sma50_cross_sma200":  fp("sma50") < fp("sma200") and f("sma50") > f("sma200"),
        "ema15":               f("ema15"),
        "ema19":               f("ema19"),
        "ema20":               f("ema20"),
        "ema50":               f("ema50"),
        "ema15_cross_up":      fp("ema15") < fp("ema19") and f("ema15") > f("ema19"),
        "ema15_cross_dn":      fp("ema15") > fp("ema19") and f("ema15") < f("ema19"),
        "ema20_cross_up":      fp("ema20") < fp("ema50") and f("ema20") > f("ema50"),
        "ema20_cross_dn":      fp("ema20") > fp("ema50") and f("ema20") < f("ema50"),
        "ema19_above_ema50":   f("ema19") > f("ema50"),
        "rsi":                 rsi_val,
        "rsi_oversold":        rsi_val < RSI_OVERSOLD,
        "rsi_overbought":      rsi_val > RSI_OVERBOUGHT,
        "rsi_bull":            30 < rsi_val < 60,
        "rsi_bear":            60 < rsi_val < 75,
        "bb_lower":            f("bb_lower"),
        "bb_mid":              f("bb_mid"),
        "bb_upper":            f("bb_upper"),
        "at_bb_lower":         c <= f("bb_lower") * 1.005,
        "at_bb_upper":         c >= f("bb_upper") * 0.995,
        "bb_squeeze":          (f("bb_upper") - f("bb_lower")) / (f("bb_mid") + 1e-9) < 0.02,
        "above_vwap":          c > f("vwap") if f("vwap") > 0 else False,
        "vwap":                f("vwap"),
    }
