"""Performance metrics for equity curves and trade logs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def periodic_returns(equity_curve: pd.Series) -> pd.Series:
    """Simple period-over-period returns of an equity curve."""
    return equity_curve.pct_change().dropna()


def cumulative_return(equity_curve: pd.Series) -> float:
    """Total return over the full period, e.g. 0.25 == +25%."""
    if len(equity_curve) < 2:
        return 0.0
    return equity_curve.iloc[-1] / equity_curve.iloc[0] - 1


def annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR implied by the equity curve's total return and length."""
    if len(equity_curve) < 2:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    n_periods = len(equity_curve) - 1
    if total_return <= 0 or n_periods == 0:
        return -1.0
    years = n_periods / periods_per_year
    return total_return ** (1 / years) - 1 if years > 0 else 0.0


def annualized_volatility(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    returns = periodic_returns(equity_curve)
    if len(returns) < 2:
        return 0.0
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0,
                  periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio computed from period returns.

    ``risk_free_rate`` is an annual rate; it is converted to a per-period
    rate before being subtracted from each period's return.
    """
    returns = periodic_returns(equity_curve)
    if len(returns) < 2:
        return 0.0
    period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - period_rf
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return (excess.mean() / std) * np.sqrt(periods_per_year)


def sortino_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0,
                   periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio (downside-deviation-adjusted return)."""
    returns = periodic_returns(equity_curve)
    if len(returns) < 2:
        return 0.0
    period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - period_rf
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if not downside_std:
        return 0.0
    return (excess.mean() / downside_std) * np.sqrt(periods_per_year)


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak at every point (<= 0)."""
    if equity_curve.empty:
        return equity_curve.copy()
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1


def max_drawdown(equity_curve: pd.Series) -> dict:
    """Maximum peak-to-trough decline plus the dates it occurred and recovered.

    Returns a dict with keys: ``max_drawdown`` (negative float),
    ``peak_date``, ``trough_date``, ``recovery_date`` (None if never
    recovered within the sample), and ``duration_days`` (peak to trough).
    """
    dd = drawdown_series(equity_curve)
    if dd.empty:
        return {"max_drawdown": 0.0, "peak_date": None, "trough_date": None,
                "recovery_date": None, "duration_days": 0}

    trough_date = dd.idxmin()
    max_dd = dd.loc[trough_date]

    peak_date = equity_curve.loc[:trough_date].idxmax()
    peak_value = equity_curve.loc[peak_date]

    post_trough = equity_curve.loc[trough_date:]
    recovered = post_trough[post_trough >= peak_value]
    recovery_date = recovered.index[0] if len(recovered) > 0 else None

    return {
        "max_drawdown": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "duration_days": (trough_date - peak_date).days if peak_date is not None else 0,
    }


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Longest number of bars spent below a prior equity peak (recovered or not)."""
    if equity_curve.empty:
        return 0
    running_max = equity_curve.cummax()
    underwater = equity_curve < running_max

    longest = current = 0
    for is_underwater in underwater:
        if is_underwater:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def win_rate(trade_log: pd.DataFrame) -> float:
    """Fraction of closed trades with positive P&L."""
    closed = trade_log.dropna(subset=["pnl"])
    if closed.empty:
        return 0.0
    return (closed["pnl"] > 0).sum() / len(closed)


def profit_factor(trade_log: pd.DataFrame) -> float:
    """Gross profit / gross loss across closed trades (inf if no losses)."""
    closed = trade_log.dropna(subset=["pnl"])
    if closed.empty:
        return 0.0
    gross_profit = closed.loc[closed["pnl"] > 0, "pnl"].sum()
    gross_loss = -closed.loc[closed["pnl"] < 0, "pnl"].sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def return_distribution(trade_log: pd.DataFrame) -> dict:
    """Summary stats of per-trade returns: mean, std, skew, kurtosis, best, worst."""
    closed = trade_log.dropna(subset=["return_pct"])
    if closed.empty:
        return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurtosis": 0.0,
                "best": 0.0, "worst": 0.0, "n_trades": 0}
    returns = closed["return_pct"]
    return {
        "mean": returns.mean(),
        "std": returns.std(ddof=1) if len(returns) > 1 else 0.0,
        "skew": returns.skew() if len(returns) > 2 else 0.0,
        "kurtosis": returns.kurtosis() if len(returns) > 3 else 0.0,
        "best": returns.max(),
        "worst": returns.min(),
        "n_trades": len(returns),
    }


def compute_metrics(equity_curve: pd.Series, trade_log: pd.DataFrame,
                     periods_per_year: int = 252, risk_free_rate: float = 0.0) -> dict:
    """Compute the full standard metrics suite for a backtest run."""
    dd = max_drawdown(equity_curve)
    dist = return_distribution(trade_log)

    return {
        "cumulative_return": cumulative_return(equity_curve),
        "annualized_return": annualized_return(equity_curve, periods_per_year),
        "annualized_volatility": annualized_volatility(equity_curve, periods_per_year),
        "sharpe_ratio": sharpe_ratio(equity_curve, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(equity_curve, risk_free_rate, periods_per_year),
        "max_drawdown": dd["max_drawdown"],
        "max_drawdown_peak_date": dd["peak_date"],
        "max_drawdown_trough_date": dd["trough_date"],
        "max_drawdown_recovery_date": dd["recovery_date"],
        "max_drawdown_duration_bars": max_drawdown_duration(equity_curve),
        "n_trades": len(trade_log),
        "win_rate": win_rate(trade_log),
        "profit_factor": profit_factor(trade_log),
        "avg_trade_return": dist["mean"],
        "trade_return_std": dist["std"],
        "trade_return_skew": dist["skew"],
        "trade_return_kurtosis": dist["kurtosis"],
        "best_trade_return": dist["best"],
        "worst_trade_return": dist["worst"],
    }
