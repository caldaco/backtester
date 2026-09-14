"""Portfolio state tracking: cash, positions, equity curve, and trade log."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Side = Literal["long", "short"]


@dataclass
class Trade:
    """A single round-trip trade (entry to exit)."""

    side: Side
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    entry_commission: float = 0.0
    exit_commission: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def pnl(self) -> float | None:
        if self.is_open:
            return None
        gross = (self.exit_price - self.entry_price) * self.shares
        if self.side == "short":
            gross = -gross
        return gross - self.entry_commission - self.exit_commission

    @property
    def return_pct(self) -> float | None:
        if self.is_open:
            return None
        cost_basis = self.entry_price * abs(self.shares)
        if cost_basis == 0:
            return 0.0
        return self.pnl / cost_basis

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "shares": self.shares,
            "exit_date": self.exit_date,
            "exit_price": self.exit_price,
            "commission": self.entry_commission + self.exit_commission,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
        }


class Portfolio:
    """Tracks cash, a single-asset position, equity history, and closed trades.

    The portfolio supports going long, flat, or (optionally) short a single
    instrument. Position sizing is "full allocation": when entering a
    position the portfolio commits all available cash (subject to
    transaction costs); when exiting it liquidates fully. This keeps the
    engine's accounting simple and deterministic while still exercising
    realistic cash/PnL bookkeeping.
    """

    def __init__(self, initial_capital: float = 100_000.0):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.shares = 0.0  # positive = long, negative = short
        self._open_trade: Trade | None = None

        self.closed_trades: list[Trade] = []
        self._equity_history: list[tuple[pd.Timestamp, float]] = []

    @property
    def position_side(self) -> Side | None:
        if self.shares > 0:
            return "long"
        if self.shares < 0:
            return "short"
        return None

    def market_value(self, price: float) -> float:
        return self.shares * price

    def equity(self, price: float) -> float:
        return self.cash + self.market_value(price)

    def enter_long(self, date: pd.Timestamp, price: float, commission_paid: float, shares: float) -> None:
        cost = shares * price + commission_paid
        self.cash -= cost
        self.shares += shares
        self._open_trade = Trade(
            side="long", entry_date=date, entry_price=price,
            shares=shares, entry_commission=commission_paid,
        )

    def enter_short(self, date: pd.Timestamp, price: float, commission_paid: float, shares: float) -> None:
        proceeds = shares * price - commission_paid
        self.cash += proceeds
        self.shares -= shares
        self._open_trade = Trade(
            side="short", entry_date=date, entry_price=price,
            shares=shares, entry_commission=commission_paid,
        )

    def exit_position(self, date: pd.Timestamp, price: float, commission_paid: float) -> Trade | None:
        if self._open_trade is None or self.shares == 0:
            return None

        shares_held = abs(self.shares)
        if self.position_side == "long":
            proceeds = shares_held * price - commission_paid
            self.cash += proceeds
        else:
            cost = shares_held * price + commission_paid
            self.cash -= cost

        self.shares = 0.0
        trade = self._open_trade
        trade.exit_date = date
        trade.exit_price = price
        trade.exit_commission = commission_paid
        self.closed_trades.append(trade)
        self._open_trade = None
        return trade

    def record_equity(self, date: pd.Timestamp, price: float) -> None:
        self._equity_history.append((date, self.equity(price)))

    @property
    def equity_curve(self) -> pd.Series:
        if not self._equity_history:
            return pd.Series(dtype=float, name="equity")
        dates, values = zip(*self._equity_history)
        return pd.Series(values, index=pd.DatetimeIndex(dates, name="date"), name="equity")

    @property
    def trade_log(self) -> pd.DataFrame:
        rows = [t.to_dict() for t in self.closed_trades]
        columns = ["side", "entry_date", "entry_price", "shares", "exit_date",
                   "exit_price", "commission", "pnl", "return_pct"]
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows, columns=columns)
