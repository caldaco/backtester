"""Z-score / Bollinger-Band mean-reversion strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.strategies.base import Strategy


class MeanReversionStrategy(Strategy):
    """Bollinger-Band / z-score mean reversion strategy.

    Computes a rolling mean and standard deviation of the close price and
    trades the resulting z-score: go long when price is oversold (z-score
    below ``-entry_z``) and exit when it reverts back to the mean band
    (``|z| < exit_z``). Optionally goes short when overbought.

    Parameters
    ----------
    window:
        Rolling lookback window for the mean/std (the "Bollinger" window).
    entry_z:
        Z-score threshold that triggers an entry.
    exit_z:
        Z-score threshold (closer to zero) that triggers an exit back to flat.
    allow_short:
        If True, an overbought reading (z-score above ``entry_z``) opens a
        short position instead of staying flat.
    """

    def __init__(self, window: int = 20, entry_z: float = 2.0, exit_z: float = 0.5,
                 allow_short: bool = False):
        if window < 2:
            raise ValueError("window must be >= 2")
        if exit_z >= entry_z:
            raise ValueError("exit_z must be smaller than entry_z")
        super().__init__(window=window, entry_z=entry_z, exit_z=exit_z, allow_short=allow_short)
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        # Use only prior bars (shift(1)) for the rolling stats so today's
        # z-score never uses today's own close in its reference band.
        prior_close = close.shift(1)
        rolling_mean = prior_close.rolling(window=self.window, min_periods=self.window).mean()
        rolling_std = prior_close.rolling(window=self.window, min_periods=self.window).std(ddof=0)

        z_score = (close - rolling_mean) / rolling_std.replace(0, np.nan)

        signal = pd.Series(index=data.index, dtype=float)
        position = 0
        z_values = z_score.to_numpy()
        out = np.zeros(len(z_values))

        for i, z in enumerate(z_values):
            if np.isnan(z):
                out[i] = position
                continue
            if position == 0:
                if z < -self.entry_z:
                    position = 1
                elif self.allow_short and z > self.entry_z:
                    position = -1
            elif position == 1:
                if z > -self.exit_z:
                    position = 0
            elif position == -1:
                if z < self.exit_z:
                    position = 0
            out[i] = position

        return pd.Series(out, index=data.index).astype(int)
