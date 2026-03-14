import os
import json
import threading
import time
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from config import INITIAL_WALLET, TICKS_PER_CANDLE, TICKERS
from data_manager import load_all_tickers, clear_old_data
from strategy import TickerStrategy
from engine import candle_stream

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
}


def _reset_state():
    sim_state.update({
        "running":      False,
        "wallet":       INITIAL_WALLET,
        "tick_count":   0,
        "trade_log":    [],
        "ticker_stats": {},
        "start_time":   None,
    })


def _run_simulation(force_refresh=False):
    _reset_state()
    sim_state["running"]    = True
    sim_state["start_time"] = time.time()

    socketio.emit("status", {"msg": "📡 Loading market data...", "type": "info"})

    all_data = load_all_tickers(force_refresh=force_refresh)
    if not all_data:
        socketio.emit("status", {"msg": "❌ No data loaded. Check tickers.", "type": "error"})
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
        }

    strategies     = {t: TickerStrategy(t, df) for t, df in all_data.items()}
    current_prices = {t: float(df["Close"].iloc[-1]) for t, df in all_data.items()}
    wallet         = float(INITIAL_WALLET)

    iterators = {t: iter(candle_stream(df, TICKS_PER_CANDLE)) for t, df in all_data.items()}
    active    = set(iterators.keys())

    socketio.emit("status", {"msg": f"🚀 Simulation started — {len(strategies)} tickers", "type": "success"})
    emit_every = max(1, TICKS_PER_CANDLE // 3)

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
            new_t, closed_t, signal = strat.process_tick(ts, price, candle_idx, wallet)

            for trade in new_t:
                wallet -= trade.cost_basis()

            for trade in closed_t:
                wallet += trade.cost_basis() + trade.pnl
                log_entry = {
                    "ticker":    trade.ticker,
                    "direction": trade.direction,
                    "entry":     f"₹{trade.entry_price:,.2f}",
                    "exit":      f"₹{trade.exit_price:,.2f}" if trade.exit_price else "—",
                    "pnl":       round(trade.pnl, 2),
                    "status":    trade.status,
                    "time":      str(trade.exit_time)[:16] if trade.exit_time else "—",
                }
                sim_state["trade_log"].insert(0, log_entry)
                sim_state["trade_log"] = sim_state["trade_log"][:50]
                socketio.emit("new_trade", log_entry)

            # Update stats
            stats = sim_state["ticker_stats"][ticker]
            stats["current_price"] = price
            stats["open_pnl"]      = round(strat.get_open_pnl(price), 2)
            stats["realised_pnl"]  = round(strat.get_total_realised_pnl(), 2)
            stats["open_trades"]   = len(strat.open_trades)
            stats["closed_trades"] = len(strat.closed_trades)
            if signal and signal.direction:
                stats["last_signal"] = f"{'▲' if signal.direction == 'CALL' else '▼'} {signal.direction} ({signal.score}/4)"

            sim_state["tick_count"] += 1
            sim_state["wallet"]      = wallet

            if sim_state["tick_count"] % emit_every == 0:
                _emit_dashboard(strategies, current_prices, wallet)

        eventlet_sleep()

    # Force close all on end
    for ticker, strat in strategies.items():
        strat.force_close_all(current_prices.get(ticker, 0), pd.Timestamp.now(tz="UTC"))
        wallet += strat.get_total_realised_pnl()

    sim_state["running"] = False
    sim_state["wallet"]  = wallet
    _emit_dashboard(strategies, current_prices, wallet)
    socketio.emit("status", {"msg": "✅ Simulation complete!", "type": "success"})
    socketio.emit("sim_done", _build_summary(strategies))


def _emit_dashboard(strategies, current_prices, wallet):
    all_closed = [t for s in strategies.values() for t in s.closed_trades]
    wins       = [t for t in all_closed if t.status == "WIN"]
    total_pnl  = sum(t.pnl for t in all_closed)
    open_pnl   = sum(s.get_open_pnl(current_prices.get(tk, 0)) for tk, s in strategies.items())
    elapsed    = round(time.time() - sim_state["start_time"], 1) if sim_state["start_time"] else 0

    socketio.emit("dashboard_update", {
        "wallet":       round(wallet, 2),
        "net_pnl":      round(total_pnl + open_pnl, 2),
        "realised_pnl": round(total_pnl, 2),
        "open_pnl":     round(open_pnl, 2),
        "wallet_pct":   round(((wallet - INITIAL_WALLET) / INITIAL_WALLET) * 100, 2),
        "tick_count":   sim_state["tick_count"],
        "win_rate":     round(len(wins) / max(len(all_closed), 1) * 100, 1),
        "total_trades": len(all_closed),
        "elapsed":      elapsed,
        "ticker_stats": sim_state["ticker_stats"],
    })


def _build_summary(strategies):
    all_closed = [t for s in strategies.values() for t in s.closed_trades]
    wins   = [t for t in all_closed if t.status == "WIN"]
    losses = [t for t in all_closed if t.status in ("LOSS", "STOPPED")]
    return {
        "total":    len(all_closed),
        "wins":     len(wins),
        "losses":   len(losses),
        "net_pnl":  round(sum(t.pnl for t in all_closed), 2),
        "win_rate": round(len(wins) / max(len(all_closed), 1) * 100, 1),
        "avg_win":  round(np.mean([t.pnl for t in wins]) if wins else 0, 2),
        "avg_loss": round(np.mean([t.pnl for t in losses]) if losses else 0, 2),
    }


def eventlet_sleep():
    import eventlet
    eventlet.sleep(0)


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", tickers=TICKERS, initial_wallet=INITIAL_WALLET)

@app.route("/api/status")
def api_status():
    return jsonify({"running": sim_state["running"], "ticks": sim_state["tick_count"]})


# ── SocketIO Events ────────────────────────────────────────────────────────
@socketio.on("start_sim")
def handle_start(data):
    if sim_state["running"]:
        emit("status", {"msg": "⚠️ Simulation already running.", "type": "warning"})
        return
    force = data.get("force_refresh", False)
    t = threading.Thread(target=_run_simulation, args=(force,), daemon=True)
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

@socketio.on("connect")
def handle_connect():
    emit("status", {"msg": "🔌 Connected to Alpha Backtester.", "type": "success"})
    if sim_state["ticker_stats"]:
        emit("dashboard_update", {
            "wallet":       round(sim_state["wallet"], 2),
            "net_pnl":      0, "realised_pnl": 0, "open_pnl": 0,
            "wallet_pct":   0, "tick_count":   sim_state["tick_count"],
            "win_rate":     0, "total_trades": 0, "elapsed": 0,
            "ticker_stats": sim_state["ticker_stats"],
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
