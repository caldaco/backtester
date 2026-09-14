"""Base interface all strategies implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Abstract base class for parameterizable trading strategies.

    A strategy consumes an OHLCV DataFrame and produces a *target position*
    signal series aligned to the same index, using values in ``{-1, 0, 1}``
    for short / flat / long. Signals must only use information available up
    to and including the current bar (e.g. rolling windows, not centered or
    forward-looking windows) -- the :class:`~backtester.engine.BacktestEngine`
    additionally shifts signals by one bar before execution so that a signal
    computed using bar ``t``'s close is only acted on at bar ``t+1``,
    eliminating lookahead bias.
    """

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a Series of target positions in {-1, 0, 1} indexed like ``data``."""
        raise NotImplementedError

    def with_params(self, **params) -> "Strategy":
        """Return a new instance of this strategy with updated parameters."""
        merged = {**self.params, **params}
        return type(self)(**merged)

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({param_str})"
