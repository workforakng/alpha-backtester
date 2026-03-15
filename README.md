# ⚡ Alpha Backtester

A real-time quantitative options backtesting platform built with Flask + Socket.IO for the web dashboard and a CLI simulation engine for terminal-first workflows.

Alpha Backtester loads historical OHLCV data, interpolates candles into fine-grained ticks, runs a multi-signal confluence strategy, applies institutional-style risk controls, and streams live performance updates.

---

## 🚀 Highlights

- **Real-time web dashboard** (Flask-SocketIO + Chart.js)
- **CLI simulation mode** with rich terminal dashboard
- **Confluence strategy engine** (MACD, Supertrend, SMA/EMA structure, RSI, Stoch RSI, VWAP, Bollinger, OBV, CCI, Williams %R)
- **Risk management stack**
  - Kelly-adjusted position sizing
  - ATR-based dynamic stop/target
  - Trailing stop + hard stop
  - Per-ticker cooldown and anti-overtrading limits
  - Portfolio drawdown guard
- **Multi-wallet comparison tool** (₹50K / ₹1L / ₹2L / ₹5L)
- **Cached market data** with auto-refresh staleness handling
- **Railway/Gunicorn deployment ready**

---

## 🧩 Signals & Algorithms Used

### Signal/Indicator Inputs (Confluence Engine)

The strategy in [`evaluate_confluence()`](strategy.py:40) uses these signal groups:

1. **Regime / Trend Strength**
   - ADX trending gate (`trending`, `strong_trend`)
   - Directional bias from DI (`di_bull`)

2. **MACD Cluster**
   - MACD bullish/bearish crossover
   - MACD histogram sign
   - MACD histogram acceleration

3. **Supertrend Cluster**
   - Supertrend bullish/bearish state
   - Supertrend flip up/down

4. **SMA Structure**
   - Price vs SMA50, SMA100, SMA200
   - SMA50 above/below SMA200 (Golden/Death cross state)
   - SMA50 crossing SMA200 trigger

5. **EMA Structure**
   - EMA15/EMA19 crossover
   - EMA20/EMA50 crossover
   - EMA19 above/below EMA50 state

6. **RSI Cluster**
   - RSI overbought/oversold extremes
   - RSI bull/bear zone
   - RSI crossing 50 up/down

7. **Stochastic RSI Cluster**
   - %K/%D bullish/bearish cross
   - Overbought/oversold zones

8. **Bollinger Bands Cluster**
   - Upper/lower band touch
   - Bandwidth expansion confirmation

9. **VWAP Cluster**
   - Price above/below VWAP state
   - VWAP cross up/down

10. **Volume / Flow**
   - OBV vs OBV-EMA (accumulation/distribution)

11. **Oscillators**
   - CCI sign + oversold/overbought
   - Williams %R overbought/oversold

12. **Candlestick Momentum**
   - Body ratio / strong-body momentum filter

> Notes:
> - Confluence score max is **30**, entry threshold is **60%** (18).
> - Signal outputs: `CALL`, `PUT`, or no-trade.

### Core Algorithms & Models

- **Candle-to-tick interpolation** via geometric-Brownian-style path generation in [`interpolate_candle_ticks()`](engine.py:90)
- **Time-normalized tick stream** in [`candle_stream()`](engine.py:109)
- **Option premium estimation** with Black–Scholes in [`black_scholes_premium()`](engine.py:21)
- **Position sizing** via fractional Kelly criterion in [`kelly_allocation()`](engine.py:51) and [`calculate_position_size()`](engine.py:71)
- **Risk controls**
  - ATR-based stop/target in [`TradeRecord.__init__()`](engine.py:135)
  - Trailing stop + hard stop in [`TradeRecord.is_stopped()`](engine.py:169)
  - Profit target checks in [`TradeRecord.is_target_hit()`](engine.py:183)
  - Cooldown / trade limits / drawdown guard in [`TickerStrategy.process_tick()`](strategy.py:252)
- **Execution model** includes slippage and brokerage impacts in [`TradeRecord.compute_pnl()`](engine.py:186)
- **Performance analytics** in [`compute_metrics()`](engine.py:208): Sharpe, Sortino, Profit Factor, Max Drawdown, Calmar, Expectancy

---

## 📁 Project Structure

```text
alpha-backtester/
├── app.py                # Flask app + Socket.IO simulation service
├── main.py               # CLI simulation entrypoint
├── strategy.py           # Ticker strategy + confluence scoring
├── indicators.py         # Indicator computations + signal extraction
├── engine.py             # Tick interpolation, option pricing, trade record, metrics
├── data_manager.py       # yfinance loading + CSV cache
├── config.py             # Global parameters
├── templates/
│   └── index.html        # Web dashboard UI
├── requirements.txt
├── Procfile
├── railway.json
├── setup.sh              # Termux-oriented setup script
└── .gitignore
```

---

## 🧠 How It Works

1. **Data loading**
   - Fetches ticker OHLCV from `yfinance`
   - Caches CSV under `data/`
   - Reloads from cache unless stale (`STALE_HOURS`) or force refresh

2. **Tick simulation**
   - Each candle is interpolated into `TICKS_PER_CANDLE` synthetic ticks (GBM-style)
   - Strategy evaluates and manages trades on every tick

3. **Trade lifecycle**
   - Signal confluence produces CALL/PUT/None
   - Position size determined via Kelly-adjusted allocation
   - Exits happen via target, stop, trailing stop, or signal flip

4. **Live updates**
   - Backend emits `dashboard_update`, `new_trade` / `trade_batch`, `status`, and `sim_done`
   - Frontend updates stats tables, trade log, signal breakdown, and equity chart

5. **Post-run summary**
   - Final trade statistics + equity curve
   - Optional wallet comparison results for multiple capital sizes

---

## 🛠️ Installation

### Standard Linux/macOS/WSL setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Termux setup (Android)

Use the provided script:

```bash
bash setup.sh
source venv/bin/activate
```

---

## ▶️ Running the Project

### 1) Web Dashboard (recommended)

```bash
python3 app.py
```

Open in browser:

- `http://localhost:5000`

### 2) CLI Backtest

```bash
python3 main.py
```

Optional flags:

```bash
python3 main.py --refresh
python3 main.py --clear-data
```

---

## ⚙️ Configuration

Primary strategy/system parameters live in `config.py`.

Key groups:

- **Universe:** `TICKERS`
- **Capital/risk:** `INITIAL_WALLET`, `TRADE_ALLOCATION`, `MAX_DRAWDOWN_PCT`
- **Execution costs:** `SLIPPAGE_PCT`, `BROKERAGE_PCT`
- **Option pricing:** `RISK_FREE_RATE`, `IMPLIED_VOLATILITY`, `DAYS_TO_EXPIRY`
- **Stops/targets:** `TRAILING_STOP_PCT`, `HARD_STOP_PCT`, `PROFIT_TARGET_PCT`, ATR parameters
- **Regime filters:** ADX settings
- **Simulation:** `TICKS_PER_CANDLE`, `SIMULATION_DELAY_MS`

Tune these conservatively and validate on out-of-sample data.

---

## 🔌 Socket.IO Event Contract

Frontend listens for:

- `status` → lifecycle and warnings/errors
- `dashboard_update` → wallet/equity, PnL, ticker stats
- `new_trade` and `trade_batch` → closed-trade log updates
- `sim_done` → final summary + equity curve
- `comparison_done` → wallet comparison table/chart

Backend emits from simulation threads in `app.py`.

---

## 📊 Wallet Comparison

`run_comparison` executes the same strategy for predefined wallet sizes and returns:

- final wallet
- net P&L
- return %
- trades
- win rate
- profitability flag

Useful for checking capital efficiency and scaling behavior.

---

## 🚢 Deployment

This repo is configured for Gunicorn + Eventlet:

- `Procfile`
- `railway.json`

Default start command:

```bash
gunicorn app:app --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 120
```

---

## 🧪 Development Notes

- Data source: Yahoo Finance (`yfinance`) reliability may vary by ticker/interval
- High-speed simulation can produce large event volumes; batch/throttle paths are implemented in the app/frontend flow
- Cached data can be cleared from UI or by deleting `data/`

---

## 🧯 Troubleshooting

### `python: command not found`
Use `python3` instead.

### No data loaded
- Check ticker symbols and period/interval combinations
- Test internet connectivity
- Try force refresh and/or clear cache

### Websocket UI not updating
- Ensure app is started via `python3 app.py`
- Check browser console for Socket.IO errors
- Verify proxy/load-balancer websocket compatibility in deployment

---

## 📝 License

No license file is currently included. Add one (e.g., MIT) before public distribution.

---

## 🙌 Acknowledgements

- Flask
- Flask-SocketIO
- Chart.js
- pandas / numpy / scipy
- yfinance
