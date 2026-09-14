import numpy as np
import pandas as pd
import pytest

from backtester.data import generate_synthetic_ohlcv


@pytest.fixture
def ohlcv_data() -> pd.DataFrame:
    return generate_synthetic_ohlcv(n_periods=300, seed=7)


@pytest.fixture
def trending_data() -> pd.DataFrame:
    """A deterministic, strictly increasing-then-decreasing price series
    with no noise, useful for asserting exact signal/trade behavior."""
    n = 100
    up = np.linspace(100, 200, n // 2)
    down = np.linspace(200, 120, n - n // 2)
    close = np.concatenate([up, down])
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": np.full(n, 1_000_000.0),
    }, index=pd.DatetimeIndex(dates, name="date"))
    return df
