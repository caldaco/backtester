"""Data loading utilities.

Supports loading OHLCV data from local CSV files and generating synthetic
OHLCV data for examples and tests. A thin, optional wrapper around
``yfinance`` is provided for pulling real market data when the package is
installed; the core framework does not depend on network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def load_csv(path: str, date_column: str = "date") -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    The file must contain a date column plus open/high/low/close/volume
    columns (case-insensitive). Returns a DataFrame indexed by a sorted
    ``DatetimeIndex`` with lowercase column names.

    Parameters
    ----------
    path:
        Path to the CSV file.
    date_column:
        Name of the column containing dates (case-insensitive).

    Raises
    ------
    ValueError
        If any of open/high/low/close/volume columns are missing.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    date_column = date_column.lower()

    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in {path}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required OHLCV columns: {missing}")

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column).sort_index()
    df.index.name = "date"
    return df[list(REQUIRED_COLUMNS)]


def load_yfinance(ticker: str, start: str | None = None, end: str | None = None,
                   interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data for ``ticker`` using the optional ``yfinance`` package.

    Requires ``pip install yfinance``. Provided as a convenience so the
    engine can consume API-sourced data as well as CSV files; not used by
    default so the framework has no hard network dependency.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "yfinance is required for load_yfinance(); install with `pip install yfinance`"
        ) from exc

    raw = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    raw.columns = [str(c).lower() for c in raw.columns]
    raw = raw.rename(columns={"adj close": "adj_close"})
    raw.index.name = "date"
    return raw[list(REQUIRED_COLUMNS)]


def generate_synthetic_ohlcv(
    n_periods: int = 1000,
    start: str = "2018-01-01",
    freq: str = "B",
    start_price: float = 100.0,
    annual_drift: float = 0.06,
    annual_vol: float = 0.25,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data via geometric Brownian motion.

    Useful for examples, tests, and demos where no external data source is
    available. Open/high/low are derived from the close-to-close path with
    a small amount of intraday noise so downstream code exercising all four
    price fields behaves sensibly.
    """
    rng = np.random.default_rng(seed)
    periods_per_year = 252
    dt = 1 / periods_per_year

    drift = (annual_drift - 0.5 * annual_vol**2) * dt
    shocks = rng.normal(loc=drift, scale=annual_vol * np.sqrt(dt), size=n_periods)
    close = start_price * np.exp(np.cumsum(shocks))

    dates = pd.date_range(start=start, periods=n_periods, freq=freq)

    intraday_noise = rng.normal(loc=0.0, scale=annual_vol * np.sqrt(dt) * 0.5, size=n_periods)
    open_ = np.empty(n_periods)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1 + intraday_noise[1:] * 0.3)

    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n_periods)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n_periods)))
    volume = rng.integers(1_000_000, 10_000_000, size=n_periods).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return df
