import pandas as pd
import pytest

from backtester.data import generate_synthetic_ohlcv, load_csv


def test_generate_synthetic_ohlcv_shape_and_columns():
    df = generate_synthetic_ohlcv(n_periods=250, seed=1)
    assert len(df) == 250
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)


def test_generate_synthetic_ohlcv_is_deterministic_with_seed():
    a = generate_synthetic_ohlcv(n_periods=100, seed=99)
    b = generate_synthetic_ohlcv(n_periods=100, seed=99)
    pd.testing.assert_frame_equal(a, b)


def test_generate_synthetic_ohlcv_high_low_consistency():
    df = generate_synthetic_ohlcv(n_periods=200, seed=3)
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["low"] <= df["high"]).all()


def test_load_csv_roundtrip(tmp_path):
    df = generate_synthetic_ohlcv(n_periods=50, seed=5)
    out = tmp_path / "sample.csv"
    df.reset_index().to_csv(out, index=False)

    loaded = load_csv(str(out))
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert len(loaded) == 50
    pd.testing.assert_index_equal(loaded.index, df.index)


def test_load_csv_missing_columns_raises(tmp_path):
    out = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2020-01-01"], "close": [100.0]}).to_csv(out, index=False)
    with pytest.raises(ValueError, match="missing required OHLCV columns"):
        load_csv(str(out))


def test_load_csv_missing_date_column_raises(tmp_path):
    out = tmp_path / "bad2.csv"
    pd.DataFrame({
        "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1],
    }).to_csv(out, index=False)
    with pytest.raises(ValueError, match="Date column"):
        load_csv(str(out))
