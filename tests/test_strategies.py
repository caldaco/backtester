import numpy as np
import pandas as pd
import pytest

from backtester.strategies import MomentumStrategy, MeanReversionStrategy


def test_momentum_signal_is_flat_during_warmup(trending_data):
    strat = MomentumStrategy(lookback=10)
    signals = strat.generate_signals(trending_data)
    assert (signals.iloc[:10] == 0).all()


def test_momentum_goes_long_on_uptrend(trending_data):
    strat = MomentumStrategy(lookback=5)
    signals = strat.generate_signals(trending_data)
    # Strictly rising prices should trigger a long breakout well before the peak.
    assert (signals.iloc[20:45] == 1).all()


def test_momentum_only_uses_past_data_no_lookahead(trending_data):
    strat = MomentumStrategy(lookback=10)
    full_signals = strat.generate_signals(trending_data)

    # Truncating the future must not change any already-computed past signal.
    truncated = trending_data.iloc[:60]
    truncated_signals = strat.generate_signals(truncated)
    pd.testing.assert_series_equal(
        full_signals.iloc[:60], truncated_signals, check_names=False,
    )


def test_momentum_signals_in_valid_range(ohlcv_data):
    strat = MomentumStrategy(lookback=20, allow_short=True)
    signals = strat.generate_signals(ohlcv_data)
    assert set(signals.unique()).issubset({-1, 0, 1})
    assert signals.index.equals(ohlcv_data.index)


def test_momentum_rejects_bad_lookback():
    with pytest.raises(ValueError):
        MomentumStrategy(lookback=1)


def test_mean_reversion_enters_long_on_extreme_dip():
    n = 60
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # Small oscillation gives the rolling window nonzero variance; a sustained
    # level shift down at index 40 should trigger a clear long entry once the
    # rolling window (built only from prior bars) still reflects the old level.
    oscillation = np.array([0.5 if i % 2 == 0 else -0.5 for i in range(n)])
    close = np.where(np.arange(n) < 40, 100.0, 70.0) + oscillation
    df = pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close,
        "volume": np.full(n, 1e6),
    }, index=pd.DatetimeIndex(dates, name="date"))

    strat = MeanReversionStrategy(window=20, entry_z=2.0, exit_z=0.5)
    signals = strat.generate_signals(df)
    assert signals.iloc[40] == 1  # enters as soon as the level shift is observed


def test_mean_reversion_no_lookahead(ohlcv_data):
    strat = MeanReversionStrategy(window=20)
    full_signals = strat.generate_signals(ohlcv_data)
    truncated_signals = strat.generate_signals(ohlcv_data.iloc[:100])
    pd.testing.assert_series_equal(
        full_signals.iloc[:100], truncated_signals, check_names=False,
    )


def test_mean_reversion_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        MeanReversionStrategy(window=20, entry_z=1.0, exit_z=2.0)


def test_strategy_with_params_creates_new_instance():
    strat = MomentumStrategy(lookback=20)
    other = strat.with_params(lookback=40)
    assert other.lookback == 40
    assert strat.lookback == 20
    assert other is not strat
