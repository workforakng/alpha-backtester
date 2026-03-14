import os, time, logging
import pandas as pd
import yfinance as yf
from config import TICKERS, DATA_DIR, DATA_INTERVAL, DATA_PERIOD, STALE_HOURS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def _csv_path(ticker):
    os.makedirs(DATA_DIR, exist_ok=True)
    safe = ticker.replace(".", "_").replace("^", "IDX_")
    return os.path.join(DATA_DIR, f"{safe}_{DATA_INTERVAL}.csv")

def _is_stale(path):
    if not os.path.exists(path): return True
    return (time.time() - os.path.getmtime(path)) > STALE_HOURS * 3600

def clear_old_data():
    if not os.path.isdir(DATA_DIR): return
    removed = sum(1 for f in os.listdir(DATA_DIR) if f.endswith(".csv") and not os.remove(os.path.join(DATA_DIR, f)))
    logger.info(f"Cleared {removed} cached CSVs.")

def _download(ticker):
    logger.info(f"Downloading {ticker}...")
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval=DATA_INTERVAL,
                         auto_adjust=True, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df.index = pd.to_datetime(df.index, utc=True)
        logger.info(f"{ticker}: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"{ticker} download failed: {e}")
        return pd.DataFrame()

def load_ticker_data(ticker, force_refresh=False):
    path = _csv_path(ticker)
    if not force_refresh and not _is_stale(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            return df
        except: pass
    df = _download(ticker)
    if not df.empty: df.to_csv(path)
    return df

def load_all_tickers(force_refresh=False):
    data = {}
    for t in TICKERS:
        df = load_ticker_data(t, force_refresh)
        if not df.empty and len(df) >= 60:
            data[t] = df
        else:
            logger.warning(f"{t}: insufficient data, skipping.")
    return data
