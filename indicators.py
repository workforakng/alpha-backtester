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
    BB_PERIOD, BB_STD, VWAP_ENABLED,
    ADX_PERIOD, ATR_PERIOD,
    STOCH_K, STOCH_D, STOCH_SMOOTH,
)


# ────────────────────────── PURE-PYTHON FALLBACKS ───────────────────────────

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
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd_fallback(close):
    fast   = _ema(close, MACD_FAST)
    slow   = _ema(close, MACD_SLOW)
    macd   = fast - slow
    signal = _ema(macd.dropna(), MACD_SIGNAL).reindex(close.index)
    return macd, signal, macd - signal


def _supertrend(df, period, mult):
    hl2 = (df["High"] + df["Low"]) / 2
    tr  = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr   = tr.rolling(period).mean()
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    direction  = pd.Series(1, index=df.index)
    supertrend = pd.Series(np.nan, index=df.index)
    for i in range(period, len(df)):
        pu = upper.iloc[i-1] if not np.isnan(upper.iloc[i-1]) else upper.iloc[i]
        pl = lower.iloc[i-1] if not np.isnan(lower.iloc[i-1]) else lower.iloc[i]
        upper.iloc[i] = min(upper.iloc[i], pu) if df["Close"].iloc[i-1] <= pu else upper.iloc[i]
        lower.iloc[i] = max(lower.iloc[i], pl) if df["Close"].iloc[i-1] >= pl else lower.iloc[i]
        if df["Close"].iloc[i] > upper.iloc[i-1]:   direction.iloc[i] = 1
        elif df["Close"].iloc[i] < lower.iloc[i-1]: direction.iloc[i] = -1
        else:                                         direction.iloc[i] = direction.iloc[i-1]
        supertrend.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]
    return supertrend, direction


def _atr(df, period=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(df, period=14):
    """Average Directional Index (pure Python)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr   = _atr(df, 1)                   # single-bar TR
    up   = high.diff()
    down = -low.diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr14  = pd.Series(plus_dm, index=df.index).rolling(period).sum()
    pdm14  = pd.Series(plus_dm,  index=df.index).rolling(period).sum()
    mdm14  = pd.Series(minus_dm, index=df.index).rolling(period).sum()
    pdi    = 100 * pdm14  / (atr14 + 1e-9)
    mdi    = 100 * mdm14  / (atr14 + 1e-9)
    dx     = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    adx    = dx.rolling(period).mean()
    return adx, pdi, mdi


def _stoch_rsi(series, k=14, d=3, smooth=3):
    rsi_s  = _rsi(series, k)
    rsi_min = rsi_s.rolling(k).min()
    rsi_max = rsi_s.rolling(k).max()
    stoch_k = 100 * (rsi_s - rsi_min) / (rsi_max - rsi_min + 1e-9)
    stoch_k = stoch_k.rolling(smooth).mean()
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def _vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()


def _bollinger(close, period=20, std=2.0):
    mid = close.rolling(period).mean()
    sd  = close.rolling(period).std()
    return mid - std*sd, mid, mid + std*sd


def _obv(df):
    """On-Balance Volume."""
    direction = np.sign(df["Close"].diff().fillna(0))
    return (direction * df["Volume"]).cumsum()


def _cci(df, period=20):
    """Commodity Channel Index."""
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - tp.rolling(period).mean()) / (0.015 * mad + 1e-9)


def _williams_r(df, period=14):
    """Williams %R."""
    highest = df["High"].rolling(period).max()
    lowest  = df["Low"].rolling(period).min()
    return -100 * (highest - df["Close"]) / (highest - lowest + 1e-9)


# ──────────────────────────── PUBLIC API ─────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c   = out["Close"]

    # ─ MACD
    out["macd"], out["macd_signal"], out["macd_hist"] = _macd_fallback(c)

    # ─ Supertrend
    out["supertrend"], out["supertrend_dir"] = _supertrend(out, SUPERTREND_PERIOD, SUPERTREND_MULT)

    # ─ ATR
    out["atr"] = _atr(out, ATR_PERIOD)

    # ─ ADX
    out["adx"], out["plus_di"], out["minus_di"] = _adx(out, ADX_PERIOD)

    # ─ SMAs
    out["sma50"]  = c.rolling(SMA_50).mean()
    out["sma100"] = c.rolling(SMA_100).mean()
    out["sma200"] = c.rolling(SMA_200).mean()

    # ─ EMAs
    for p, name in [(EMA_15,"ema15"),(EMA_19,"ema19"),(EMA_20,"ema20"),(EMA_50,"ema50")]:
        out[name] = _ema(c, p)

    # ─ RSI
    out["rsi"] = _rsi(c, RSI_PERIOD)

    # ─ Stochastic RSI
    out["stoch_k"], out["stoch_d"] = _stoch_rsi(c, STOCH_K, STOCH_D, STOCH_SMOOTH)

    # ─ Bollinger Bands
    out["bb_lower"], out["bb_mid"], out["bb_upper"] = _bollinger(c, BB_PERIOD, BB_STD)
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / (out["bb_mid"] + 1e-9)

    # ─ VWAP
    if VWAP_ENABLED and "Volume" in out.columns:
        out["vwap"] = _vwap(out)

    # ─ OBV
    out["obv"] = _obv(out)
    out["obv_ema"] = _ema(out["obv"], 20)

    # ─ CCI
    out["cci"] = _cci(out, 20)

    # ─ Williams %R
    out["willr"] = _williams_r(out, 14)

    # ─ Candle body ratio (engulfing / momentum detection)
    out["body_ratio"] = (c - out["Open"]).abs() / ((out["High"] - out["Low"]) + 1e-9)

    return out


def get_latest_signals(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return {}
    r    = df.iloc[-1]
    prev = df.iloc[-2]
    c    = float(r["Close"])

    def f(col):  return float(r.get(col,   np.nan) or 0)
    def fp(col): return float(prev.get(col, np.nan) or 0)

    rsi_val   = f("rsi")
    cci_val   = f("cci")
    willr_val = f("willr")
    stoch_k   = f("stoch_k")
    stoch_d   = f("stoch_d")
    adx_val   = f("adx")
    atr_val   = f("atr")

    return {
        # ─ price
        "close":               c,
        "atr":                 atr_val,
        "adx":                 adx_val,
        "trending":            adx_val > 20,
        "strong_trend":        adx_val > 30,
        "plus_di":             f("plus_di"),
        "minus_di":            f("minus_di"),
        "di_bull":             f("plus_di") > f("minus_di"),

        # ─ MACD
        "macd":                f("macd"),
        "macd_signal":         f("macd_signal"),
        "macd_hist":           f("macd_hist"),
        "macd_cross_up":       fp("macd") < fp("macd_signal") and f("macd") > f("macd_signal"),
        "macd_cross_dn":       fp("macd") > fp("macd_signal") and f("macd") < f("macd_signal"),
        "macd_bull":           f("macd_hist") > 0,
        "macd_accel":          f("macd_hist") > fp("macd_hist"),  # histogram growing

        # ─ Supertrend
        "st_bullish":          f("supertrend_dir") > 0,
        "st_flip_bull":        fp("supertrend_dir") < 0 and f("supertrend_dir") > 0,
        "st_flip_bear":        fp("supertrend_dir") > 0 and f("supertrend_dir") < 0,

        # ─ SMAs
        "sma50":               f("sma50"),
        "sma100":              f("sma100"),
        "sma200":              f("sma200"),
        "above_sma50":         c > f("sma50"),
        "above_sma100":        c > f("sma100"),
        "above_sma200":        c > f("sma200"),
        "sma50_above_sma200":  f("sma50") > f("sma200"),
        "sma50_cross_sma200":  fp("sma50") < fp("sma200") and f("sma50") > f("sma200"),

        # ─ EMAs
        "ema15":               f("ema15"),
        "ema19":               f("ema19"),
        "ema20":               f("ema20"),
        "ema50":               f("ema50"),
        "ema15_cross_up":      fp("ema15") < fp("ema19") and f("ema15") > f("ema19"),
        "ema15_cross_dn":      fp("ema15") > fp("ema19") and f("ema15") < f("ema19"),
        "ema20_cross_up":      fp("ema20") < fp("ema50") and f("ema20") > f("ema50"),
        "ema20_cross_dn":      fp("ema20") > fp("ema50") and f("ema20") < f("ema50"),
        "ema19_above_ema50":   f("ema19") > f("ema50"),

        # ─ RSI
        "rsi":                 rsi_val,
        "rsi_oversold":        rsi_val < RSI_OVERSOLD,
        "rsi_overbought":      rsi_val > RSI_OVERBOUGHT,
        "rsi_bull":            30 < rsi_val < 60,
        "rsi_bear":            60 < rsi_val < 75,
        "rsi_cross_50_up":     fp("rsi") < 50 and rsi_val >= 50,
        "rsi_cross_50_dn":     fp("rsi") > 50 and rsi_val <= 50,

        # ─ Stochastic RSI
        "stoch_k":             stoch_k,
        "stoch_d":             stoch_d,
        "stoch_oversold":      stoch_k < 20,
        "stoch_overbought":    stoch_k > 80,
        "stoch_cross_up":      fp("stoch_k") < fp("stoch_d") and stoch_k > stoch_d,
        "stoch_cross_dn":      fp("stoch_k") > fp("stoch_d") and stoch_k < stoch_d,

        # ─ Bollinger
        "bb_lower":            f("bb_lower"),
        "bb_mid":              f("bb_mid"),
        "bb_upper":            f("bb_upper"),
        "bb_width":            f("bb_width"),
        "at_bb_lower":         c <= f("bb_lower") * 1.005,
        "at_bb_upper":         c >= f("bb_upper") * 0.995,
        "bb_squeeze":          f("bb_width") < 0.02,
        "bb_expansion":        f("bb_width") > fp("bb_width") * 1.1,

        # ─ VWAP
        "above_vwap":          c > f("vwap") if f("vwap") > 0 else False,
        "vwap":                f("vwap"),
        "vwap_cross_up":       fp("close") < fp("vwap") and c > f("vwap"),
        "vwap_cross_dn":       fp("close") > fp("vwap") and c < f("vwap"),

        # ─ OBV
        "obv_bull":            f("obv") > f("obv_ema"),   # OBV above its EMA = accumulation
        "obv_bear":            f("obv") < f("obv_ema"),

        # ─ CCI
        "cci":                 cci_val,
        "cci_bull":            cci_val > 0,
        "cci_oversold":        cci_val < -100,
        "cci_overbought":      cci_val > 100,

        # ─ Williams %R
        "willr":               willr_val,
        "willr_oversold":      willr_val < -80,
        "willr_overbought":    willr_val > -20,

        # ─ Body ratio (candle momentum)
        "strong_body":         f("body_ratio") > 0.6,
    }
