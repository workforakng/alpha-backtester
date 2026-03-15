import os
import json
import threading
import time
import logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from config import INITIAL_WALLET, TICKS_PER_CANDLE, TICKERS
from data_manager import load_all_tickers, clear_old_data
from strategy import TickerStrategy
from engine import candle_stream

logging.basicConfig(
    level=os.environ.get("ALPHA_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "alpha_backtester_secret")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ── Global simulation state ────────────────────────────────────────────────
sim_state = {
    "running":       False,
    "wallet":        INITIAL_WALLET,
    "tick_count":    0,
    "trade_log":     [],
    "ticker_stats":  {},
    "start_time":    None,
    "thread":        None,
    "speed":         1,       # ticks emitted per eventlet yield (1=normal, 10=fast, 100=turbo)
    "equity_curve":  [],      # [{t, v}] sampled every N ticks
    "strategy_results": [],   # multi-wallet comparison results
}


def _reset_state(initial_wallet):
    sim_state.update({
        "running":          False,
        "wallet":           initial_wallet,
        "tick_count":       0,
        "trade_log":        [],
        "ticker_stats":     {},
        "start_time":       None,
        "equity_curve":     [],
        "strategy_results": [],
    })


STRATEGY_WALLETS = [50_000, 100_000, 200_000, 500_000]


def _run_simulation(params):
    force_refresh  = params.get("force_refresh", False)
    custom_tickers = params.get("tickers", TICKERS)
    initial_wallet = float(params.get("wallet", INITIAL_WALLET))
    speed          = max(1, int(params.get("speed", 1)))   # 1/10/50/100
    period         = params.get("period", "5d")
    interval       = params.get("interval", "1m")

    sim_state["speed"] = speed
    _reset_state(initial_wallet)
    sim_state["running"]    = True
    sim_state["start_time"] = time.time()

    socketio.emit("status", {"msg": "📡 Loading market data...", "type": "info"})

    all_data = load_all_tickers(
        force_refresh=force_refresh,
        tickers=custom_tickers,
        period=period,
        interval=interval,
    )
    logger.info(
        "Web simulation requested: tickers=%s wallet=%.2f speed=%s period=%s interval=%s",
        custom_tickers, initial_wallet, speed, period, interval,
    )
    if not all_data:
        socketio.emit("status", {"msg": "❌ No data loaded. Check tickers/date range.", "type": "error"})
        sim_state["running"] = False
        return

    # Init per-ticker stats
    for ticker in all_data:
        sim_state["ticker_stats"][ticker] = {
            "current_price": float(all_data[ticker]["Close"].iloc[-1]),
            "open_pnl":      0.0,
            "realised_pnl":  0.0,
            "open_trades":   0,
            "closed_trades": 0,
            "last_signal":   "—",
            "confluence_score": "—",
            "signal_reasons":   [],
        }

    strategies     = {t: TickerStrategy(t, df) for t, df in all_data.items()}
    current_prices = {t: float(df["Close"].iloc[-1]) for t, df in all_data.items()}
    wallet         = float(initial_wallet)
    portfolio_peak = wallet

    iterators = {t: iter(candle_stream(df, TICKS_PER_CANDLE)) for t, df in all_data.items()}
    active    = set(iterators.keys())

    socketio.emit("status", {"msg": f"🚀 Simulation started — {len(strategies)} tickers | speed x{speed}", "type": "success"})
    emit_every    = max(1, TICKS_PER_CANDLE // 4)
    equity_every  = TICKS_PER_CANDLE * 5
    tick_batch    = 0

    while active and sim_state["running"]:
        for ticker in list(active):
            try:
                ts, price, candle_idx = next(iterators[ticker])
            except StopIteration:
                strategies[ticker].force_close_all(current_prices.get(ticker, 0),
                                                    pd.Timestamp.now(tz="UTC"))
                active.discard(ticker)
                continue

            current_prices[ticker] = price
            strat  = strategies[ticker]
            new_t, closed_t, signal = strat.process_tick(ts, price, candle_idx, wallet, portfolio_peak)

            for trade in new_t:
                wallet -= trade.cost_basis()
                logger.debug(
                    "Web wallet debit: ticker=%s dir=%s cost=%.2f wallet=%.2f",
                    trade.ticker, trade.direction, trade.cost_basis(), wallet,
                )

            for trade in closed_t:
                wallet += trade.cost_basis() + trade.pnl
                logger.debug(
                    "Web wallet credit: ticker=%s reason=%s credit=%.2f pnl=%.2f wallet=%.2f",
                    trade.ticker, trade.exit_reason, trade.cost_basis() + trade.pnl, trade.pnl, wallet,
                )
                log_entry = {
                    "ticker":    trade.ticker,
                    "direction": trade.direction,
                    "entry":     f"₹{trade.entry_price:,.2f}",
                    "exit":      f"₹{trade.exit_price:,.2f}" if trade.exit_price else "—",
                    "pnl":       round(trade.pnl, 2),
                    "status":    trade.status,
                    "time":      str(trade.exit_time)[:16] if trade.exit_time else "—",
                    "exit_reason": getattr(trade, 'exit_reason', '—'),
                }
                sim_state["trade_log"].insert(0, log_entry)
                sim_state["trade_log"] = sim_state["trade_log"][:100]
                socketio.emit("new_trade", log_entry)

            # Update stats
            stats = sim_state["ticker_stats"][ticker]
            stats["current_price"] = price
            stats["open_pnl"]      = round(strat.get_open_pnl(price), 2)
            stats["realised_pnl"]  = round(strat.get_total_realised_pnl(), 2)
            stats["open_trades"]   = len(strat.open_trades)
            stats["closed_trades"] = len(strat.closed_trades)
            if signal and signal.direction:
                stats["last_signal"]      = f"{'▲' if signal.direction == 'CALL' else '▼'} {signal.direction} ({signal.score:.0f}/{signal.max_score:.0f})"
                stats["confluence_score"] = f"{round(signal.score/signal.max_score*100)}%"
                stats["signal_strength"]  = signal.strength
                stats["signal_reasons"]   = signal.reasons[:8]

            sim_state["tick_count"] += 1
            sim_state["wallet"]      = wallet
            tick_batch              += 1
            portfolio_equity = wallet + sum(s.get_open_cost_basis() + s.get_open_pnl(current_prices.get(tk, 0.0)) for tk, s in strategies.items())
            portfolio_peak = max(portfolio_peak, portfolio_equity)

            # equity curve sample
            if sim_state["tick_count"] % equity_every == 0:
                equity_value = wallet + sum(
                    s.get_open_cost_basis() + s.get_open_pnl(current_prices.get(tk, 0.0))
                    for tk, s in strategies.items()
                )
                sim_state["equity_curve"].append({
                    "t": sim_state["tick_count"],
                    "v": round(equity_value, 2)
                })

            if tick_batch >= speed:
                tick_batch = 0
                if sim_state["tick_count"] % emit_every == 0:
                    _emit_dashboard(strategies, current_prices, wallet, initial_wallet)
                import eventlet; eventlet.sleep(0)

    # Force close all on end
    for ticker, strat in strategies.items():
        strat.force_close_all(current_prices.get(ticker, 0), pd.Timestamp.now(tz="UTC"))

    # Recalculate wallet from all realised PnL
    total_realised = sum(t.pnl for s in strategies.values() for t in s.closed_trades)
    wallet = initial_wallet + total_realised
    logger.info(
        "Web wallet reconciled: initial=%.2f closed_trades=%s final=%.2f",
        initial_wallet,
        sum(len(s.closed_trades) for s in strategies.values()),
        wallet,
    )

    sim_state["running"] = False
    sim_state["wallet"]  = wallet
    sim_state["equity_curve"].append({"t": sim_state["tick_count"], "v": round(wallet, 2)})
    _emit_dashboard(strategies, current_prices, wallet, initial_wallet)
    socketio.emit("status", {"msg": "✅ Simulation complete!", "type": "success"})

    summary = _build_summary(strategies, initial_wallet)
    summary["equity_curve"] = sim_state["equity_curve"]
    socketio.emit("sim_done", summary)


def _run_multi_wallet_comparison(params):
    """Run simulation with multiple wallet sizes and compare results."""
    results = []
    custom_tickers = params.get("tickers", TICKERS)
    period   = params.get("period", "5d")
    interval = params.get("interval", "1m")

    all_data = load_all_tickers(force_refresh=False, tickers=custom_tickers,
                                period=period, interval=interval)
    if not all_data:
        socketio.emit("status", {"msg": "❌ No data for comparison.", "type": "error"})
        return

    socketio.emit("status", {"msg": "🔬 Running multi-wallet strategy comparison...", "type": "info"})

    for wallet_size in STRATEGY_WALLETS:
        strategies = {t: TickerStrategy(t, df) for t, df in all_data.items()}
        current_prices = {t: float(df["Close"].iloc[-1]) for t, df in all_data.items()}
        wallet = float(wallet_size)
        portfolio_peak = wallet
        iterators = {t: iter(candle_stream(df, TICKS_PER_CANDLE)) for t, df in all_data.items()}
        active = set(iterators.keys())

        while active:
            for ticker in list(active):
                try:
                    ts, price, candle_idx = next(iterators[ticker])
                except StopIteration:
                    strategies[ticker].force_close_all(current_prices.get(ticker, 0),
                                                       pd.Timestamp.now(tz="UTC"))
                    active.discard(ticker)
                    continue
                current_prices[ticker] = price
                strat = strategies[ticker]
                new_t, closed_t, signal = strat.process_tick(ts, price, candle_idx, wallet, portfolio_peak)
                for t in new_t: wallet -= t.cost_basis()
                for t in closed_t: wallet += t.cost_basis() + t.pnl
                portfolio_equity = wallet + sum(s.get_open_cost_basis() + s.get_open_pnl(current_prices.get(tk, 0.0)) for tk, s in strategies.items())
                portfolio_peak = max(portfolio_peak, portfolio_equity)
            import eventlet; eventlet.sleep(0)

        for ticker, strat in strategies.items():
            strat.force_close_all(current_prices.get(ticker, 0), pd.Timestamp.now(tz="UTC"))

        all_closed = [t for s in strategies.values() for t in s.closed_trades]
        wins = [t for t in all_closed if t.status == "WIN"]
        total_pnl = sum(t.pnl for t in all_closed)
        final_wallet = wallet_size + total_pnl

        results.append({
            "wallet":       wallet_size,
            "final":        round(final_wallet, 2),
            "net_pnl":      round(total_pnl, 2),
            "pct":          round((total_pnl / wallet_size) * 100, 2),
            "trades":       len(all_closed),
            "win_rate":     round(len(wins) / max(len(all_closed), 1) * 100, 1),
            "profitable":   total_pnl > 0,
        })
        socketio.emit("status", {"msg": f"  ✓ ₹{wallet_size:,.0f} wallet done — P&L: {'+' if total_pnl>=0 else ''}₹{total_pnl:,.0f}", "type": "info"})

    sim_state["strategy_results"] = results
    socketio.emit("comparison_done", results)
    socketio.emit("status", {"msg": "🏆 Strategy comparison complete!", "type": "success"})


def _emit_dashboard(strategies, current_prices, wallet, initial_wallet):
    all_closed = [t for s in strategies.values() for t in s.closed_trades]
    wins       = [t for t in all_closed if t.status == "WIN"]
    total_pnl  = sum(t.pnl for t in all_closed)
    open_pnl   = sum(s.get_open_pnl(current_prices.get(tk, 0)) for tk, s in strategies.items())
    open_cost  = sum(s.get_open_cost_basis() for s in strategies.values())
    elapsed    = round(time.time() - sim_state["start_time"], 1) if sim_state["start_time"] else 0

    socketio.emit("dashboard_update", {
        "wallet":        round(wallet, 2),
        "equity":        round(wallet + open_cost + open_pnl, 2),
        "initial_wallet": initial_wallet,
        "net_pnl":       round(total_pnl + open_pnl, 2),
        "realised_pnl":  round(total_pnl, 2),
        "open_pnl":      round(open_pnl, 2),
        "wallet_pct":    round((((wallet + open_cost + open_pnl) - initial_wallet) / initial_wallet) * 100, 2),
        "tick_count":    sim_state["tick_count"],
        "win_rate":      round(len(wins) / max(len(all_closed), 1) * 100, 1),
        "total_trades":  len(all_closed),
        "elapsed":       elapsed,
        "ticker_stats":  sim_state["ticker_stats"],
    })


def _build_summary(strategies, initial_wallet):
    all_closed = [t for s in strategies.values() for t in s.closed_trades]
    wins   = [t for t in all_closed if t.status == "WIN"]
    losses = [t for t in all_closed if t.status in ("LOSS", "STOPPED")]
    return {
        "total":          len(all_closed),
        "wins":           len(wins),
        "losses":         len(losses),
        "net_pnl":        round(sum(t.pnl for t in all_closed), 2),
        "win_rate":       round(len(wins) / max(len(all_closed), 1) * 100, 1),
        "avg_win":        round(np.mean([t.pnl for t in wins])   if wins   else 0, 2),
        "avg_loss":       round(np.mean([t.pnl for t in losses]) if losses else 0, 2),
        "initial_wallet": initial_wallet,
        "best_trade":     round(max((t.pnl for t in all_closed), default=0), 2),
        "worst_trade":    round(min((t.pnl for t in all_closed), default=0), 2),
    }


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           tickers=TICKERS,
                           initial_wallet=INITIAL_WALLET)

@app.route("/api/status")
def api_status():
    return jsonify({"running": sim_state["running"], "ticks": sim_state["tick_count"]})


# ── SocketIO Events ────────────────────────────────────────────────────────
@socketio.on("start_sim")
def handle_start(data):
    if sim_state["running"]:
        emit("status", {"msg": "⚠️ Simulation already running.", "type": "warning"})
        return
    t = threading.Thread(target=_run_simulation, args=(data,), daemon=True)
    sim_state["thread"] = t
    t.start()

@socketio.on("stop_sim")
def handle_stop():
    sim_state["running"] = False
    emit("status", {"msg": "🛑 Simulation stopped by user.", "type": "warning"})

@socketio.on("clear_data")
def handle_clear():
    clear_old_data()
    emit("status", {"msg": "🗑️ Cache cleared.", "type": "info"})

@socketio.on("run_comparison")
def handle_comparison(data):
    if sim_state["running"]:
        emit("status", {"msg": "⚠️ Stop current simulation first.", "type": "warning"})
        return
    t = threading.Thread(target=_run_multi_wallet_comparison, args=(data,), daemon=True)
    t.start()

@socketio.on("connect")
def handle_connect():
    emit("status", {"msg": "🔌 Connected to Alpha Backtester.", "type": "success"})
    if sim_state["ticker_stats"]:
        emit("dashboard_update", {
            "wallet":         round(sim_state["wallet"], 2),
            "equity":         round(sim_state["wallet"], 2),
            "initial_wallet": INITIAL_WALLET,
            "net_pnl":        0, "realised_pnl": 0, "open_pnl": 0,
            "wallet_pct":     0, "tick_count":   sim_state["tick_count"],
            "win_rate":       0, "total_trades": 0, "elapsed": 0,
            "ticker_stats":   sim_state["ticker_stats"],
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
