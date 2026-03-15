# ───────────────────────────────────────────────────────────────────
# Alpha Backtester — Central Configuration
# ───────────────────────────────────────────────────────────────────

# ── Universe
TICKERS = ["RELIANCE.NS", "TCS.NS", "^NSEI"]

# ── Capital
INITIAL_WALLET        = 100_000.0
TRADE_ALLOCATION      = 0.08          # fraction of wallet per trade (Kelly-capped)
MAX_CONCURRENT_TRADES = 3

# ── Execution costs
SLIPPAGE_PCT          = 0.0005
BROKERAGE_PCT         = 0.0003

# ── Option pricing (Black-Scholes)
OPTION_PREMIUM_PCT    = 0.015
RISK_FREE_RATE        = 0.065
DAYS_TO_EXPIRY        = 7
IMPLIED_VOLATILITY    = 0.20

# ── Risk management
TRAILING_STOP_PCT     = 0.015        # trailing stop from peak PnL
HARD_STOP_PCT         = 0.025        # hard stop-loss from entry cost
PROFIT_TARGET_PCT     = 0.045        # take-profit from entry cost
MAX_DRAWDOWN_PCT      = 0.12         # halt trading if wallet drops >12% from peak
COOLDOWN_CANDLES      = 3            # candles to wait after a stopped trade
MAX_TRADES_PER_TICKER = 8            # max closed trades per ticker per session (anti-overtrading)

# ── Kelly Criterion
KELLY_FRACTION        = 0.25         # fractional Kelly (quarter-Kelly = conservative)
KELLY_MIN_ALLOC       = 0.03         # never bet less than 3% per trade
KELLY_MAX_ALLOC       = 0.12         # never bet more than 12% per trade

# ── Regime filter (ADX)
ADX_PERIOD            = 14
ADX_THRESHOLD         = 20           # only trade when ADX > 20 (trending market)
REGIME_FILTER_ENABLED = True

# ── ATR-based dynamic stops
ATR_PERIOD            = 14
ATR_STOP_MULT         = 1.5          # stop = entry ± ATR_STOP_MULT * ATR
ATR_TARGET_MULT       = 2.5          # target = entry ± ATR_TARGET_MULT * ATR
ATR_STOPS_ENABLED     = True

# ── Indicators — MACD
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9

# ── Indicators — Supertrend
SUPERTREND_PERIOD = 10
SUPERTREND_MULT   = 3.0

# ── Indicators — Moving Averages
SMA_50   = 50
SMA_100  = 100
SMA_200  = 200
EMA_15   = 15
EMA_19   = 19
EMA_20   = 20
EMA_50   = 50

# ── Indicators — RSI
RSI_PERIOD     = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD   = 30

# ── Indicators — Bollinger Bands
BB_PERIOD = 20
BB_STD    = 2.0

# ── Indicators — VWAP
VWAP_ENABLED = True

# ── Indicators — Stochastic RSI
STOCH_K      = 14
STOCH_D      = 3
STOCH_SMOOTH = 3

# ── Data
DATA_INTERVAL    = "1m"
DATA_PERIOD      = "5d"
DATA_DIR         = "data"
STALE_HOURS      = 6

# ── Simulation
TICKS_PER_CANDLE = 60
GBM_SIGMA_SCALE  = 0.3
