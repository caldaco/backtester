"""Walk-forward validation: rolling in-sample optimization + out-of-sample testing.

Walk-forward validation splits the data into successive
``[train window][test window]`` blocks that slide forward through time. For
each block, the strategy's parameters are grid-searched on the train window
only, the winning parameters are then evaluated on the *immediately
following, non-overlapping* test window, and metrics are reported
separately for train and test. Because each window's strategy signals are
computed strictly within that window's own data slice (never on future
bars, and never sharing data between a window's train and test halves),
this procedure cannot leak information from the future into a decision --
the standard defense against lookahead bias in strategy research.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from backtester import metrics as metrics_module
from backtester.engine import BacktestEngine, BacktestResult
from backtester.strategies.base import Strategy


@dataclass
class WalkForwardWindow:
    """Results for a single train/test block."""

    window_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_metrics: dict
    test_metrics: dict
    test_result: BacktestResult


@dataclass
class WalkForwardResult:
    """Aggregate results across all walk-forward windows."""

    windows: list[WalkForwardWindow]

    def summary_table(self) -> pd.DataFrame:
        """One row per window comparing key train vs. test metrics."""
        rows = []
        for w in self.windows:
            rows.append({
                "window": w.window_index,
                "train_start": w.train_start, "train_end": w.train_end,
                "test_start": w.test_start, "test_end": w.test_end,
                "best_params": w.best_params,
                "train_sharpe": w.train_metrics.get("sharpe_ratio"),
                "test_sharpe": w.test_metrics.get("sharpe_ratio"),
                "train_return": w.train_metrics.get("cumulative_return"),
                "test_return": w.test_metrics.get("cumulative_return"),
                "test_max_drawdown": w.test_metrics.get("max_drawdown"),
                "test_n_trades": w.test_metrics.get("n_trades"),
            })
        return pd.DataFrame(rows)

    def combined_out_of_sample_equity(self) -> pd.Series:
        """Chain each window's test equity curve into one continuous series.

        Each window's test equity is rescaled to start where the previous
        window's test equity ended, so the result approximates the equity
        path of always trading with the most recently walk-forward-optimized
        parameters.
        """
        pieces = []
        running_capital = None
        for w in self.windows:
            eq = w.test_result.equity_curve
            if eq.empty:
                continue
            if running_capital is None:
                scaled = eq
            else:
                scaled = eq / eq.iloc[0] * running_capital
            pieces.append(scaled)
            running_capital = scaled.iloc[-1]
        if not pieces:
            return pd.Series(dtype=float)
        return pd.concat(pieces).sort_index()

    def aggregate_test_metrics(self) -> dict:
        """Mean of each scalar metric across all out-of-sample test windows."""
        if not self.windows:
            return {}
        keys = self.windows[0].test_metrics.keys()
        agg = {}
        for key in keys:
            values = [w.test_metrics[key] for w in self.windows
                      if isinstance(w.test_metrics.get(key), (int, float))]
            agg[key] = sum(values) / len(values) if values else float("nan")
        return agg


def _param_grid_combinations(param_grid: dict) -> list[dict]:
    keys = list(param_grid.keys())
    values_product = itertools.product(*(param_grid[k] for k in keys))
    return [dict(zip(keys, combo)) for combo in values_product]


def walk_forward_validation(
    data: pd.DataFrame,
    strategy_cls: type[Strategy],
    param_grid: dict,
    train_size: int,
    test_size: int,
    step: int | None = None,
    optimize_metric: str = "sharpe_ratio",
    initial_capital: float = 100_000.0,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> WalkForwardResult:
    """Run rolling walk-forward validation over ``data``.

    Parameters
    ----------
    data:
        Full OHLCV history to walk forward across.
    strategy_cls:
        Strategy class (not instance) to optimize; must accept the keys of
        ``param_grid`` as constructor kwargs.
    param_grid:
        Dict mapping parameter name -> list of candidate values. The
        cartesian product is grid-searched on each train window.
    train_size, test_size:
        Number of bars in each train / test window.
    step:
        Bars to advance the window start between iterations. Defaults to
        ``test_size`` (non-overlapping test windows covering the full series).
    optimize_metric:
        Key into :func:`backtester.metrics.compute_metrics` used to rank
        candidate parameters on the train window (higher is better).
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step or test_size
    combos = _param_grid_combinations(param_grid)
    if not combos:
        raise ValueError("param_grid produced no parameter combinations")

    n = len(data)
    windows: list[WalkForwardWindow] = []
    window_index = 0
    start = 0

    while start + train_size + test_size <= n:
        train_slice = data.iloc[start: start + train_size]
        test_slice = data.iloc[start + train_size: start + train_size + test_size]

        best_params, best_score, best_train_metrics = None, float("-inf"), None
        for params in combos:
            strategy = strategy_cls(**params)
            result = BacktestEngine(
                train_slice, strategy, initial_capital=initial_capital,
                commission_bps=commission_bps, slippage_bps=slippage_bps,
            ).run()
            train_metrics = result.summary()
            score = train_metrics.get(optimize_metric, float("-inf"))
            if score is None or pd.isna(score):
                score = float("-inf")
            if score > best_score:
                best_params, best_score, best_train_metrics = params, score, train_metrics

        test_strategy = strategy_cls(**best_params)
        test_result = BacktestEngine(
            test_slice, test_strategy, initial_capital=initial_capital,
            commission_bps=commission_bps, slippage_bps=slippage_bps,
        ).run()
        test_metrics = test_result.summary()

        windows.append(WalkForwardWindow(
            window_index=window_index,
            train_start=train_slice.index[0], train_end=train_slice.index[-1],
            test_start=test_slice.index[0], test_end=test_slice.index[-1],
            best_params=best_params,
            train_metrics=best_train_metrics,
            test_metrics=test_metrics,
            test_result=test_result,
        ))

        window_index += 1
        start += step

    return WalkForwardResult(windows=windows)
