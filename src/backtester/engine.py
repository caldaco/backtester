"""Event-driven backtesting engine.

The engine walks forward through OHLCV data bar-by-bar, executes trades
implied by a strategy's target-position signal, and tracks portfolio state
through a :class:`~backtester.portfolio.Portfolio`. Signals are shifted by
one bar before being acted on -- a decision computed from bar ``t``'s data
is only ever executed at bar ``t+1`` -- which is the key mechanism that
prevents lookahead bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from backtester import metrics as metrics_module
from backtester.portfolio import Portfolio
from backtester.strategies.base import Strategy


@dataclass
class BacktestResult:
    """Container for everything a backtest run produces."""

    equity_curve: pd.Series
    trade_log: pd.DataFrame
    signals: pd.Series
    positions: pd.Series
    data: pd.DataFrame
    initial_capital: float

    def summary(self, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> dict:
        """Compute the standard performance metrics dict for this run."""
        return metrics_module.compute_metrics(
            self.equity_curve, self.trade_log,
            periods_per_year=periods_per_year, risk_free_rate=risk_free_rate,
        )

    def __repr__(self) -> str:
        return (
            f"BacktestResult(n_bars={len(self.equity_curve)}, "
            f"n_trades={len(self.trade_log)}, "
            f"final_equity={self.equity_curve.iloc[-1]:,.2f})"
            if len(self.equity_curve) else "BacktestResult(empty)"
        )


class BacktestEngine:
    """Runs a single strategy over a single OHLCV series.

    Parameters
    ----------
    data:
        OHLCV DataFrame indexed by date with columns
        ``open, high, low, close, volume`` (see :mod:`backtester.data`).
    strategy:
        A :class:`~backtester.strategies.base.Strategy` instance.
    initial_capital:
        Starting cash.
    commission_bps:
        Commission charged per trade, in basis points of trade notional.
    slippage_bps:
        Adverse price slippage applied per trade, in basis points --
        buys/covers execute at a worse (higher) price, sells/shorts execute
        at a worse (lower) price.
    execution_price:
        Which column of ``data`` trades execute against (``"close"`` by
        default; ``"open"`` is also common to simulate next-bar-open fills).
    allow_short:
        Whether short positions are permitted; only relevant if the
        strategy itself emits -1 signals.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        initial_capital: float = 100_000.0,
        commission_bps: float = 5.0,
        slippage_bps: float = 5.0,
        execution_price: str = "close",
    ):
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"data is missing required columns: {missing}")
        if execution_price not in data.columns:
            raise ValueError(f"execution_price '{execution_price}' not a column in data")

        self.data = data.sort_index()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_rate = commission_bps / 10_000
        self.slippage_rate = slippage_bps / 10_000
        self.execution_price = execution_price

    def _adjusted_price(self, price: float, direction: str) -> float:
        """Apply slippage. direction is 'buy' (worse=higher) or 'sell' (worse=lower)."""
        if direction == "buy":
            return price * (1 + self.slippage_rate)
        return price * (1 - self.slippage_rate)

    def _commission(self, notional: float) -> float:
        return notional * self.commission_rate

    def run(self) -> BacktestResult:
        signals = self.strategy.generate_signals(self.data)
        if not signals.index.equals(self.data.index):
            raise ValueError("strategy signals must be indexed identically to data")

        # Shift by one bar: a signal computed using bar t's data is only
        # acted on starting at bar t+1. This is what prevents lookahead.
        target_position = signals.shift(1).fillna(0).astype(int)

        portfolio = Portfolio(self.initial_capital)
        realized_positions = []
        last_index = len(target_position) - 1

        for i, (date, target) in enumerate(target_position.items()):
            if i == last_index:
                # Force liquidation on the final bar so the equity curve and
                # trade log both reflect fully realized P&L, with no
                # duplicate final timestamp in the equity curve.
                target = 0
            row = self.data.loc[date]
            fill_price = row[self.execution_price]
            current = 0 if portfolio.shares == 0 else (1 if portfolio.shares > 0 else -1)

            if target != current:
                # Close any existing position first.
                if current == 1:
                    exit_price = self._adjusted_price(fill_price, "sell")
                    commission = self._commission(abs(portfolio.shares) * exit_price)
                    portfolio.exit_position(date, exit_price, commission)
                elif current == -1:
                    exit_price = self._adjusted_price(fill_price, "buy")
                    commission = self._commission(abs(portfolio.shares) * exit_price)
                    portfolio.exit_position(date, exit_price, commission)

                # Open the new target position (if not flat) using all
                # available cash (full-allocation sizing).
                if target == 1:
                    entry_price = self._adjusted_price(fill_price, "buy")
                    shares = math.floor(portfolio.cash / (entry_price * (1 + self.commission_rate)))
                    if shares > 0:
                        commission = self._commission(shares * entry_price)
                        portfolio.enter_long(date, entry_price, commission, shares)
                elif target == -1:
                    entry_price = self._adjusted_price(fill_price, "sell")
                    shares = math.floor(portfolio.cash / (entry_price * (1 + self.commission_rate)))
                    if shares > 0:
                        commission = self._commission(shares * entry_price)
                        portfolio.enter_short(date, entry_price, commission, shares)

            portfolio.record_equity(date, row["close"])
            realized_positions.append(0 if portfolio.shares == 0 else (1 if portfolio.shares > 0 else -1))

        positions = pd.Series(realized_positions, index=self.data.index, name="position")

        return BacktestResult(
            equity_curve=portfolio.equity_curve,
            trade_log=portfolio.trade_log,
            signals=signals,
            positions=positions,
            data=self.data,
            initial_capital=self.initial_capital,
        )
