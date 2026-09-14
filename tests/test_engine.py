import numpy as np
import pandas as pd
import pytest

from backtester.engine import BacktestEngine
from backtester.strategies import MomentumStrategy
from backtester.strategies.base import Strategy


class AlwaysLongStrategy(Strategy):
    """Trivial strategy: long on every bar. Used to test engine mechanics
    in isolation from real signal logic."""

    def __init__(self):
        super().__init__()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=data.index)


class FlipFlopStrategy(Strategy):
    """Alternates long/flat every other bar to exercise repeated entries/exits."""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        values = [1 if i % 4 < 2 else 0 for i in range(len(data))]
        return pd.Series(values, index=data.index)


def test_engine_rejects_missing_columns():
    bad_data = pd.DataFrame({"close": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match="missing required columns"):
        BacktestEngine(bad_data, AlwaysLongStrategy())


def test_engine_signal_is_shifted_one_bar_no_lookahead(trending_data):
    engine = BacktestEngine(trending_data, AlwaysLongStrategy(),
                             commission_bps=0, slippage_bps=0)
    result = engine.run()
    # Position at bar 0 must be flat: the raw signal at bar 0 (target=1) can only
    # be acted on starting bar 1 once it has been shifted forward.
    assert result.positions.iloc[0] == 0
    assert result.positions.iloc[1] == 1


def test_engine_produces_equity_curve_same_length_as_data(ohlcv_data):
    engine = BacktestEngine(ohlcv_data, MomentumStrategy(lookback=20))
    result = engine.run()
    assert len(result.equity_curve) == len(ohlcv_data)
    assert result.equity_curve.index.equals(ohlcv_data.index)


def test_engine_final_position_is_flat_after_forced_liquidation(trending_data):
    engine = BacktestEngine(trending_data, AlwaysLongStrategy())
    result = engine.run()
    assert result.positions.iloc[-1] == 0
    assert len(result.trade_log) >= 1
    assert result.trade_log.iloc[-1]["exit_date"] == trending_data.index[-1]


def test_engine_zero_cost_long_only_matches_buy_and_hold_direction(trending_data):
    engine = BacktestEngine(trending_data, AlwaysLongStrategy(),
                             commission_bps=0, slippage_bps=0, initial_capital=100_000)
    result = engine.run()
    final_equity = result.equity_curve.iloc[-1]
    # Price rises then falls but ends above the start, so a fully-invested,
    # zero-cost long should also end up with a net gain.
    assert final_equity > 100_000


def test_engine_commissions_reduce_equity_relative_to_zero_cost(trending_data):
    cheap = BacktestEngine(trending_data, FlipFlopStrategy(),
                            commission_bps=0, slippage_bps=0).run()
    expensive = BacktestEngine(trending_data, FlipFlopStrategy(),
                                commission_bps=50, slippage_bps=50).run()
    assert expensive.equity_curve.iloc[-1] < cheap.equity_curve.iloc[-1]


def test_engine_trade_log_has_matching_entries_and_exits(ohlcv_data):
    engine = BacktestEngine(ohlcv_data, MomentumStrategy(lookback=15))
    result = engine.run()
    trades = result.trade_log
    if len(trades) > 0:
        assert trades["exit_date"].notna().all()
        assert (trades["exit_date"] >= trades["entry_date"]).all()


def test_engine_shares_never_negative_cash_long_only(ohlcv_data):
    engine = BacktestEngine(ohlcv_data, MomentumStrategy(lookback=10),
                             commission_bps=10, slippage_bps=10)
    result = engine.run()
    assert (result.equity_curve > 0).all()
