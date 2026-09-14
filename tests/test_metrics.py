import numpy as np
import pandas as pd
import pytest

from backtester import metrics


def _equity(values, start="2020-01-01"):
    dates = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=dates, name="equity")


def test_cumulative_return_basic():
    eq = _equity([100, 110, 121])
    assert metrics.cumulative_return(eq) == pytest.approx(0.21)


def test_cumulative_return_empty_or_single_point():
    assert metrics.cumulative_return(_equity([100])) == 0.0


def test_annualized_return_matches_cagr_formula():
    # Doubling over exactly one year (252 bars) should annualize to ~100%.
    eq = _equity([100] + [100 * (2 ** (i / 252)) for i in range(1, 253)])
    ann = metrics.annualized_return(eq, periods_per_year=252)
    assert ann == pytest.approx(1.0, abs=0.02)


def test_sharpe_ratio_zero_for_flat_equity():
    eq = _equity([100] * 50)
    assert metrics.sharpe_ratio(eq) == 0.0


def test_sharpe_ratio_positive_for_steadily_rising_equity():
    eq = _equity([100 * (1.001 ** i) for i in range(100)])
    assert metrics.sharpe_ratio(eq) > 0


def test_max_drawdown_detects_known_decline():
    eq = _equity([100, 120, 90, 95, 130])
    result = metrics.max_drawdown(eq)
    assert result["max_drawdown"] == pytest.approx(90 / 120 - 1)
    assert result["peak_date"] == eq.index[1]
    assert result["trough_date"] == eq.index[2]
    assert result["recovery_date"] == eq.index[4]


def test_max_drawdown_no_recovery_within_sample():
    eq = _equity([100, 120, 90, 95])
    result = metrics.max_drawdown(eq)
    assert result["recovery_date"] is None


def test_max_drawdown_duration_counts_bars_underwater():
    eq = _equity([100, 120, 90, 95, 110, 130])
    # Underwater at indices 2,3,4 (below the peak of 120) => 3 bars.
    assert metrics.max_drawdown_duration(eq) == 3


def test_win_rate_and_profit_factor():
    trades = pd.DataFrame({"pnl": [100, -50, 200, -25], "return_pct": [0.1, -0.05, 0.2, -0.02]})
    assert metrics.win_rate(trades) == pytest.approx(0.5)
    assert metrics.profit_factor(trades) == pytest.approx(300 / 75)


def test_win_rate_empty_trade_log():
    trades = pd.DataFrame(columns=["pnl", "return_pct"])
    assert metrics.win_rate(trades) == 0.0
    assert metrics.profit_factor(trades) == 0.0


def test_profit_factor_no_losses_is_infinite():
    trades = pd.DataFrame({"pnl": [10, 20], "return_pct": [0.1, 0.2]})
    assert metrics.profit_factor(trades) == float("inf")


def test_compute_metrics_returns_all_expected_keys():
    eq = _equity([100, 105, 102, 110, 108])
    trades = pd.DataFrame({
        "side": ["long"], "entry_date": [eq.index[0]], "entry_price": [100],
        "shares": [10], "exit_date": [eq.index[-1]], "exit_price": [108],
        "commission": [1.0], "pnl": [79.0], "return_pct": [0.079],
    })
    result = metrics.compute_metrics(eq, trades)
    expected_keys = {
        "cumulative_return", "annualized_return", "annualized_volatility",
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "max_drawdown_peak_date",
        "max_drawdown_trough_date", "max_drawdown_recovery_date",
        "max_drawdown_duration_bars", "n_trades", "win_rate", "profit_factor",
        "avg_trade_return", "trade_return_std", "trade_return_skew",
        "trade_return_kurtosis", "best_trade_return", "worst_trade_return",
    }
    assert expected_keys.issubset(result.keys())
    assert result["n_trades"] == 1
