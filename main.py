# main.py — Entry point with real-time terminal dashboard

import os
import sys
import time
import signal
import textwrap
import logging
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

from config import (
    INITIAL_WALLET, SIMULATION_DELAY_MS, TICKS_PER_CANDLE,
    TRADE_LOG_SIZE, TICKERS
)
from data_manager import load_all_tickers, clear_old_data, get_data_summary
from strategy import TickerStrategy
from engine import candle_stream

logging.basicConfig(
    level=os.environ.get("ALPHA_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── ANSI Color Codes ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def clr(text, color): return f"{color}{text}{RESET}"
def bold(text):       return f"{BOLD}{text}{RESET}"


# ── Dashboard Renderer ────────────────────────────────────────────────────────

class Dashboard:
    def __init__(self, tickers):
        self.tickers        = tickers
        self.trade_log      = []       # last N closed trades across all tickers
        self.tick_count     = 0
        self.start_time     = time.time()
        self._clear         = "\033[2J\033[H"

    def _bar(self, pct, width=20):
        filled = int(max(0, min(1, pct)) * width)
        bar    = "█" * filled + "░" * (width - filled)
        return bar

    def _pnl_color(self, val):
        if val > 0:  return clr(f"+₹{val:,.2f}", GREEN)
        if val < 0:  return clr(f"-₹{abs(val):,.2f}", RED)
        return clr(f"₹{val:,.2f}", WHITE)

    def _format_trade(self, t):
        tag = clr("▲CALL", GREEN) if t.direction == "CALL" else clr("▼PUT", RED)
        pnl = self._pnl_color(t.pnl)
        st  = clr(t.status, GREEN if t.status == "WIN" else RED)
        ts  = str(t.exit_time)[:16] if t.exit_time else "open"
        return f"  {tag} {t.ticker:<14} @ ₹{t.entry_price:>10,.2f} | {pnl:>16} | {st} | {ts}"

    def render(self, wallet, strategies, current_prices):
        total_open_pnl    = 0.0
        total_open_cost   = 0.0
        total_realised    = 0.0
        open_trade_count  = 0
        closed_trade_count= 0

        per_ticker_lines  = []
        for ticker, strat in strategies.items():
            price = current_prices.get(ticker, 0.0)
            open_pnl = strat.get_open_pnl(price)
            open_cost = strat.get_open_cost_basis()
            real_pnl = strat.get_total_realised_pnl()
            total_open_pnl   += open_pnl
            total_open_cost  += open_cost
            total_realised   += real_pnl
            open_trade_count += len(strat.open_trades)
            closed_trade_count += len(strat.closed_trades)
            signal_info = ""
            if strat.open_trades:
                dirs = ", ".join(
                    (clr("▲", GREEN) if t.direction == "CALL" else clr("▼", RED))
                    for t in strat.open_trades
                )
                signal_info = f"[{dirs}]"
            per_ticker_lines.append(
                f"  {clr(ticker, CYAN):<22} ₹{price:>10,.2f} | "
                f"Open P&L: {self._pnl_color(open_pnl):>16} | "
                f"Realised: {self._pnl_color(real_pnl):>16} {signal_info}"
            )

        net_pnl      = total_realised + total_open_pnl
        wallet_now   = wallet + total_open_cost + total_open_pnl
        wallet_pct   = (net_pnl / INITIAL_WALLET) * 100
        elapsed      = time.time() - self.start_time

        lines = [self._clear]
        lines.append(bold(clr("═" * 72, CYAN)))
        lines.append(bold(clr("  ██████╗ ALPHA BACKTESTER v1.0  ███████╗  Quantitative Engine", CYAN)))
        lines.append(bold(clr("═" * 72, CYAN)))

        # Wallet panel
        wallet_bar = self._bar(max(0, wallet_now / INITIAL_WALLET))
        lines.append(f"\n{bold('  WALLET')}")
        lines.append(f"  Initial  : {clr(f'₹{INITIAL_WALLET:,.2f}', WHITE)}")
        lines.append(f"  Current  : {clr(f'₹{wallet_now:,.2f}', GREEN if wallet_now >= INITIAL_WALLET else RED)}  {wallet_bar}")
        lines.append(f"  Net P&L  : {self._pnl_color(net_pnl)}  ({'+' if wallet_pct >= 0 else ''}{wallet_pct:.2f}%)")
        lines.append(f"  Realised : {self._pnl_color(total_realised)}   Open: {self._pnl_color(total_open_pnl)}")

        # Stats
        win_trades  = [t for s in strategies.values() for t in s.closed_trades if t.status == "WIN"]
        loss_trades = [t for s in strategies.values() for t in s.closed_trades if t.status in ("LOSS", "STOPPED")]
        total_closed = len(win_trades) + len(loss_trades)
        win_rate = (len(win_trades) / total_closed * 100) if total_closed > 0 else 0.0

        lines.append(f"\n{bold('  STATS')}")
        lines.append(
            f"  Ticks: {clr(str(self.tick_count), YELLOW)}   "
            f"Open Trades: {clr(str(open_trade_count), CYAN)}   "
            f"Closed: {clr(str(closed_trade_count), WHITE)}   "
            f"Win Rate: {clr(f'{win_rate:.1f}%', GREEN if win_rate > 50 else RED)}   "
            f"Elapsed: {clr(f'{elapsed:.0f}s', DIM)}"
        )

        # Per-ticker breakdown
        lines.append(f"\n{bold('  TICKERS')}")
        lines.extend(per_ticker_lines)

        # Trade log
        lines.append(f"\n{bold('  RECENT TRADES')}")
        if self.trade_log:
            for t in self.trade_log[-TRADE_LOG_SIZE:]:
                lines.append(self._format_trade(t))
        else:
            lines.append(clr("  No closed trades yet...", DIM))

        lines.append(f"\n{clr('  Press Ctrl+C to stop simulation.', DIM)}")
        print("\n".join(lines), flush=True)


# ── Main Simulation Loop ──────────────────────────────────────────────────────

def run_simulation(force_refresh: bool = False):
    print(clr("\n  Loading market data...", CYAN))

    all_data = load_all_tickers(force_refresh=force_refresh)
    if not all_data:
        print(clr("[ERROR] No data loaded. Check internet connection.", RED))
        sys.exit(1)

    print(get_data_summary(all_data))
    logger.info("Loaded %s tickers for CLI simulation", len(all_data))

    wallet      = float(INITIAL_WALLET)
    strategies  = {ticker: TickerStrategy(ticker, df) for ticker, df in all_data.items()}
    dashboard   = Dashboard(list(strategies.keys()))
    current_prices = {t: float(df["Close"].iloc[-1]) for t, df in all_data.items()}
    portfolio_peak = wallet

    # Build unified tick stream (sorted by timestamp)
    tick_streams = {}
    for ticker, df in all_data.items():
        tick_streams[ticker] = list(candle_stream(df, n_ticks=TICKS_PER_CANDLE))

    # Determine minimum length
    lengths = {t: len(ticks) for t, ticks in tick_streams.items()}
    max_ticks = max(lengths.values())

    print(clr(f"\n  Simulating {max_ticks:,} ticks across {len(strategies)} tickers...\n", CYAN))
    time.sleep(1.0)

    # Interleave ticks across tickers (round-robin)
    iterators = {t: iter(ticks) for t, ticks in tick_streams.items()}
    active    = set(iterators.keys())

    render_every = max(1, TICKS_PER_CANDLE // 2)

    try:
        while active:
            for ticker in list(active):
                try:
                    ts, price, candle_idx = next(iterators[ticker])
                except StopIteration:
                    # Force-close any remaining open positions
                    strategies[ticker].force_close_all(
                        current_prices.get(ticker, 0.0),
                        pd.Timestamp.now(tz="UTC")
                    )
                    active.discard(ticker)
                    continue

                current_prices[ticker] = price
                strat = strategies[ticker]

                new_trades, closed_trades, signal = strat.process_tick(
                    ts, price, candle_idx, wallet, portfolio_peak
                )

                # Adjust wallet for trade costs and P&L
                for trade in new_trades:
                    wallet -= trade.cost_basis()
                    logger.debug(
                        "CLI wallet debit: ticker=%s dir=%s cost=%.2f wallet=%.2f",
                        trade.ticker, trade.direction, trade.cost_basis(), wallet,
                    )

                for trade in closed_trades:
                    wallet += trade.cost_basis() + trade.pnl
                    dashboard.trade_log.append(trade)
                    logger.debug(
                        "CLI wallet credit: ticker=%s reason=%s credit=%.2f pnl=%.2f wallet=%.2f",
                        trade.ticker, trade.exit_reason, trade.cost_basis() + trade.pnl, trade.pnl, wallet,
                    )

                dashboard.tick_count += 1
                portfolio_equity = wallet + sum(s.get_open_cost_basis() + s.get_open_pnl(current_prices.get(tk, 0.0)) for tk, s in strategies.items())
                portfolio_peak = max(portfolio_peak, portfolio_equity)

                if dashboard.tick_count % render_every == 0:
                    dashboard.render(wallet, strategies, current_prices)

                if SIMULATION_DELAY_MS > 0:
                    time.sleep(SIMULATION_DELAY_MS / 1000.0)

    except KeyboardInterrupt:
        print(clr("\n\n  Simulation interrupted by user.", YELLOW))

    # Final close of all open trades
    for ticker, strat in strategies.items():
        closed_before = len(strat.closed_trades)
        strat.force_close_all(current_prices.get(ticker, 0), pd.Timestamp.now(tz="UTC"))
        for t in strat.closed_trades[closed_before:]:
            if t not in dashboard.trade_log:
                dashboard.trade_log.append(t)

    all_closed = [t for s in strategies.values() for t in s.closed_trades]
    wallet = INITIAL_WALLET + sum(t.pnl for t in all_closed)
    logger.info(
        "CLI wallet reconciled: initial=%.2f closed_trades=%s final=%.2f",
        INITIAL_WALLET, len(all_closed), wallet,
    )

    dashboard.render(wallet, strategies, current_prices)
    _print_final_report(wallet, strategies, dashboard)


def _print_final_report(wallet: float, strategies: dict, dashboard: Dashboard):
    print(bold(clr("\n" + "═" * 72, CYAN)))
    print(bold(clr("  FINAL SIMULATION REPORT", CYAN)))
    print(bold(clr("═" * 72, CYAN)))

    all_closed = [t for s in strategies.values() for t in s.closed_trades]
    wins   = [t for t in all_closed if t.status == "WIN"]
    losses = [t for t in all_closed if t.status in ("LOSS", "STOPPED")]
    stopped = [t for t in all_closed if t.status == "STOPPED"]

    total_pnl = sum(t.pnl for t in all_closed)
    win_rate  = len(wins) / max(len(all_closed), 1) * 100
    avg_win   = np.mean([t.pnl for t in wins])   if wins   else 0
    avg_loss  = np.mean([t.pnl for t in losses]) if losses else 0

    print(f"  Total Trades   : {len(all_closed)}")
    print(f"  Wins           : {clr(str(len(wins)), GREEN)}  |  Losses: {clr(str(len(losses)), RED)}  |  Stopped: {clr(str(len(stopped)), YELLOW)}")
    print(f"  Win Rate       : {clr(f'{win_rate:.1f}%', GREEN if win_rate >= 50 else RED)}")
    print(f"  Avg Win        : {clr(f'₹{avg_win:,.2f}', GREEN)}")
    print(f"  Avg Loss       : {clr(f'₹{avg_loss:,.2f}', RED)}")
    print(f"  Net P&L        : {clr(f'₹{total_pnl:+,.2f}', GREEN if total_pnl >= 0 else RED)}")
    print(f"  Final Wallet   : {clr(f'₹{wallet:,.2f}', GREEN if wallet >= INITIAL_WALLET else RED)}")
    print(f"  Return         : {clr(f'{((wallet - INITIAL_WALLET) / INITIAL_WALLET * 100):+.2f}%', GREEN if wallet >= INITIAL_WALLET else RED)}")
    print(bold(clr("═" * 72 + "\n", CYAN)))

    # Save trade log to CSV
    if all_closed:
        import csv
        os.makedirs("results", exist_ok=True)
        fname = f"results/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ticker","direction","entry_time","exit_time",
                              "entry_price","premium","qty","pnl","status"])
            for t in all_closed:
                writer.writerow([t.ticker, t.direction, t.entry_time, t.exit_time,
                                  t.entry_price, t.premium, t.qty, t.pnl, t.status])
        print(clr(f"  Trade log saved → {fname}\n", DIM))


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    force = "--refresh" in sys.argv or "-r" in sys.argv
    if "--clear-data" in sys.argv:
        clear_old_data()
        print(clr("  Data cache cleared.", YELLOW))

    print(bold(clr("""
  ██████╗ ██╗     ██████╗ ██╗  ██╗ █████╗
  ██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗
  ███████║██║     ██████╔╝███████║███████║
  ██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║
  ██║  ██║███████╗██║     ██║  ██║██║  ██║
      BACKTESTER  ─  Quantitative Options Engine
    """, CYAN)))

    run_simulation(force_refresh=force)
