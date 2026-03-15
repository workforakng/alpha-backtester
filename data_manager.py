import os
import time
import logging
import pandas as pd
import yfinance as yf
from config import DATA_DIR, STALE_HOURS, TICKERS, DATA_INTERVAL, DATA_PERIOD

os.makedirs(DATA_DIR, exist_ok=True)
logger = logging.getLogger(__name__)


def _cache_path(ticker, period, interval):
    safe = ticker.replace("^", "IDX_").replace(".", "_")
    return os.path.join(DATA_DIR, f"{safe}_{period}_{interval}.csv")


def _is_stale(path):
    if not os.path.exists(path):
        return True
    age_hours = (time.time() - os.path.getmtime(path)) / 3600
    return age_hours > STALE_HOURS


def load_ticker(ticker, force_refresh=False, period=DATA_PERIOD, interval=DATA_INTERVAL):
    path = _cache_path(ticker, period, interval)
    if not force_refresh and not _is_stale(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if len(df) > 10:
                logger.debug("Loaded %s from cache: rows=%s period=%s interval=%s", ticker, len(df), period, interval)
                return df
        except Exception:
            pass
    try:
        df = yf.download(
            ticker, period=period, interval=interval,
            progress=False, auto_adjust=True,
        )
        if df is None or df.empty:
            return None
        # flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(df) < 30:
            return None
        df.to_csv(path)
        logger.debug("Downloaded %s from yfinance: rows=%s period=%s interval=%s", ticker, len(df), period, interval)
        return df
    except Exception as e:
        print(f"[data_manager] Error loading {ticker}: {e}")
        return None


def load_all_tickers(force_refresh=False, tickers=None,
                     period=DATA_PERIOD, interval=DATA_INTERVAL):
    tickers = tickers or TICKERS
    result = {}
    for t in tickers:
        df = load_ticker(t, force_refresh=force_refresh,
                         period=period, interval=interval)
        if df is not None:
            result[t] = df
        else:
            print(f"[data_manager] Skipped {t} — no data")
    return result


def clear_old_data():
    if not os.path.exists(DATA_DIR):
        return
    for f in os.listdir(DATA_DIR):
        if f.endswith(".csv"):
            os.remove(os.path.join(DATA_DIR, f))


def get_data_summary(all_data):
    if not all_data:
        return "[data_manager] No market data loaded."
    lines = ["\n[data_manager] Market data summary"]
    for ticker, df in all_data.items():
        start = pd.Timestamp(df.index.min())
        end = pd.Timestamp(df.index.max())
        lines.append(
            f"  - {ticker:<12} rows={len(df):>5} start={start} end={end} close={float(df['Close'].iloc[-1]):,.2f}"
        )
    return "\n".join(lines)
