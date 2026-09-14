import pandas as pd

from backtester.strategies import MomentumStrategy
from backtester.validation import walk_forward_validation


def test_walk_forward_validation_produces_expected_number_of_windows(ohlcv_data):
    train_size, test_size = 100, 50
    result = walk_forward_validation(
        ohlcv_data, MomentumStrategy,
        param_grid={"lookback": [10, 20]},
        train_size=train_size, test_size=test_size,
    )
    n = len(ohlcv_data)
    expected_windows = (n - train_size - test_size) // test_size + 1
    assert len(result.windows) == expected_windows


def test_walk_forward_windows_are_non_overlapping_and_time_ordered(ohlcv_data):
    result = walk_forward_validation(
        ohlcv_data, MomentumStrategy,
        param_grid={"lookback": [10, 20]},
        train_size=100, test_size=50,
    )
    for w in result.windows:
        assert w.train_start < w.train_end < w.test_start <= w.test_end
        assert w.train_end < w.test_start  # no overlap between train and test

    for prev, nxt in zip(result.windows, result.windows[1:]):
        assert prev.test_start < nxt.test_start


def test_walk_forward_best_params_selected_from_grid(ohlcv_data):
    result = walk_forward_validation(
        ohlcv_data, MomentumStrategy,
        param_grid={"lookback": [10, 20, 30]},
        train_size=120, test_size=40,
    )
    for w in result.windows:
        assert w.best_params["lookback"] in (10, 20, 30)


def test_walk_forward_summary_table_has_train_and_test_columns(ohlcv_data):
    result = walk_forward_validation(
        ohlcv_data, MomentumStrategy,
        param_grid={"lookback": [15]},
        train_size=100, test_size=50,
    )
    table = result.summary_table()
    assert {"train_sharpe", "test_sharpe", "train_return", "test_return"}.issubset(table.columns)
    assert len(table) == len(result.windows)


def test_combined_out_of_sample_equity_is_continuous(ohlcv_data):
    result = walk_forward_validation(
        ohlcv_data, MomentumStrategy,
        param_grid={"lookback": [15]},
        train_size=100, test_size=50,
    )
    combined = result.combined_out_of_sample_equity()
    assert len(combined) > 0
    assert combined.index.is_monotonic_increasing
