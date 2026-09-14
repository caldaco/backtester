"""backtester: a lightweight event-driven backtesting framework for systematic trading strategies."""

from backtester.portfolio import Portfolio, Trade
from backtester.engine import BacktestEngine, BacktestResult
from backtester.data import load_csv, generate_synthetic_ohlcv
from backtester import metrics

__version__ = "0.1.0"

__all__ = [
    "Portfolio",
    "Trade",
    "BacktestEngine",
    "BacktestResult",
    "load_csv",
    "generate_synthetic_ohlcv",
    "metrics",
]
