"""Matplotlib plotting helpers for backtest results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from backtester import metrics as metrics_module
from backtester.engine import BacktestResult


def plot_equity_curve(result: BacktestResult, title: str = "Equity Curve", ax=None):
    """Plot the equity curve with an underwater (drawdown) subplot below it."""
    equity = result.equity_curve
    dd = metrics_module.drawdown_series(equity)

    if ax is None:
        fig, (ax_eq, ax_dd) = plt.subplots(
            2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        ax_eq, ax_dd = ax

    ax_eq.plot(equity.index, equity.values, color="#1f77b4", linewidth=1.5, label="Equity")
    ax_eq.axhline(result.initial_capital, color="grey", linestyle="--", linewidth=0.8,
                   label="Initial capital")
    ax_eq.set_title(title)
    ax_eq.set_ylabel("Portfolio value ($)")
    ax_eq.legend(loc="upper left")
    ax_eq.grid(alpha=0.3)

    ax_dd.fill_between(dd.index, dd.values * 100, 0, color="#d62728", alpha=0.4)
    ax_dd.plot(dd.index, dd.values * 100, color="#d62728", linewidth=1.0)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Date")
    ax_dd.grid(alpha=0.3)

    plt.tight_layout()
    return ax_eq.figure


def plot_signals(result: BacktestResult, title: str = "Price & Signals", ax=None):
    """Overlay long/short entry and exit markers on the price series."""
    data = result.data
    trades = result.trade_log

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(data.index, data["close"], color="black", linewidth=1.0, alpha=0.8, label="Close")

    if not trades.empty:
        longs = trades[trades["side"] == "long"]
        shorts = trades[trades["side"] == "short"]

        if not longs.empty:
            ax.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="green",
                       s=70, label="Long entry", zorder=5)
            ax.scatter(longs["exit_date"], longs["exit_price"], marker="v", color="red",
                       s=70, label="Long exit", zorder=5)
        if not shorts.empty:
            ax.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="orange",
                       s=70, label="Short entry", zorder=5)
            ax.scatter(shorts["exit_date"], shorts["exit_price"], marker="^", color="blue",
                       s=70, label="Short exit", zorder=5)

    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return ax.figure


def plot_trade_pnl(result: BacktestResult, title: str = "Trade P&L", ax=None):
    """Bar chart of realized P&L per closed trade, in trade order."""
    trades = result.trade_log.dropna(subset=["pnl"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))

    colors = ["#2ca02c" if p >= 0 else "#d62728" for p in trades["pnl"]]
    ax.bar(range(len(trades)), trades["pnl"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("P&L ($)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    return ax.figure


def plot_backtest_summary(result: BacktestResult, title: str = "Backtest Summary"):
    """Combined figure: equity + underwater, price & signals, and trade P&L."""
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 12), sharex=False,
        gridspec_kw={"height_ratios": [3, 1, 2, 1.5]},
    )
    plot_equity_curve(result, title=f"{title} — Equity", ax=(axes[0], axes[1]))
    plot_signals(result, title=f"{title} — Price & Trades", ax=axes[2])
    plot_trade_pnl(result, title=f"{title} — Trade P&L", ax=axes[3])
    fig.tight_layout()
    return fig
