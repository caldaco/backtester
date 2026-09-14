"""Donchian-channel momentum (breakout) strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.strategies.base import Strategy


class MomentumStrategy(Strategy):
    """Classic N-period breakout / channel momentum strategy.

    Goes long when the close makes a new ``lookback``-period high and exits
    to flat (or optionally reverses to short) when the close makes a new
    ``lookback``-period low. This is the "turtle trading" style breakout
    rule.

    Parameters
    ----------
    lookback:
        Number of trailing bars (excluding the current bar) used to compute
        the rolling high/low channel.
    allow_short:
        If True, a new N-period low flips the position to short instead of
        just flattening it.
    """

    def __init__(self, lookback: int = 20, allow_short: bool = False):
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        super().__init__(lookback=lookback, allow_short=allow_short)
        self.lookback = lookback
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        # Exclude the current bar from the channel so the signal reflects a
        # genuine breakout above/below the *prior* N bars.
        rolling_high = close.shift(1).rolling(window=self.lookback, min_periods=self.lookback).max()
        rolling_low = close.shift(1).rolling(window=self.lookback, min_periods=self.lookback).min()

        breakout_up = close > rolling_high
        breakout_down = close < rolling_low

        raw_signal = pd.Series(np.nan, index=data.index)
        raw_signal[breakout_up] = 1
        raw_signal[breakout_down] = -1 if self.allow_short else 0

        signal = raw_signal.ffill().fillna(0)
        return signal.astype(int)
