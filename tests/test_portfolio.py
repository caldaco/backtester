import pandas as pd
import pytest

from backtester.portfolio import Portfolio


def test_initial_state():
    p = Portfolio(initial_capital=100_000)
    assert p.cash == 100_000
    assert p.shares == 0
    assert p.position_side is None
    assert p.equity(price=50) == 100_000


def test_invalid_initial_capital_raises():
    with pytest.raises(ValueError):
        Portfolio(initial_capital=0)


def test_enter_long_updates_cash_and_shares():
    p = Portfolio(initial_capital=10_000)
    p.enter_long(pd.Timestamp("2020-01-01"), price=100, commission_paid=5, shares=90)
    assert p.shares == 90
    assert p.cash == 10_000 - (90 * 100) - 5
    assert p.position_side == "long"


def test_exit_long_records_trade_with_correct_pnl():
    p = Portfolio(initial_capital=10_000)
    p.enter_long(pd.Timestamp("2020-01-01"), price=100, commission_paid=5, shares=90)
    trade = p.exit_position(pd.Timestamp("2020-01-05"), price=110, commission_paid=5)

    assert trade is not None
    assert trade.side == "long"
    expected_pnl = (110 - 100) * 90 - 5 - 5
    assert trade.pnl == pytest.approx(expected_pnl)
    assert p.shares == 0
    assert p.position_side is None
    assert len(p.closed_trades) == 1


def test_enter_short_and_exit_profit_when_price_falls():
    p = Portfolio(initial_capital=10_000)
    p.enter_short(pd.Timestamp("2020-01-01"), price=100, commission_paid=0, shares=50)
    assert p.shares == -50
    assert p.position_side == "short"

    trade = p.exit_position(pd.Timestamp("2020-01-10"), price=80, commission_paid=0)
    assert trade.pnl == pytest.approx((100 - 80) * 50)
    assert trade.pnl > 0


def test_exit_with_no_open_position_returns_none():
    p = Portfolio(initial_capital=10_000)
    assert p.exit_position(pd.Timestamp("2020-01-01"), price=100, commission_paid=0) is None


def test_equity_curve_and_trade_log_accumulate():
    p = Portfolio(initial_capital=10_000)
    p.record_equity(pd.Timestamp("2020-01-01"), price=100)
    p.enter_long(pd.Timestamp("2020-01-02"), price=100, commission_paid=0, shares=50)
    p.record_equity(pd.Timestamp("2020-01-03"), price=105)
    p.exit_position(pd.Timestamp("2020-01-04"), price=105, commission_paid=0)
    p.record_equity(pd.Timestamp("2020-01-04"), price=105)

    curve = p.equity_curve
    assert len(curve) == 3
    assert curve.iloc[-1] == pytest.approx(p.cash)

    log = p.trade_log
    assert len(log) == 1
    assert log.iloc[0]["side"] == "long"
