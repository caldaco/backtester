"""Example usage: run both strategies over sample data, print metrics, save plots.

Run from the project root with the virtual environment active:

    python examples/run_backtest.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless-safe; comment out to use an interactive backend

from backtester.data import load_csv, generate_synthetic_ohlcv
from backtester.engine import BacktestEngine
from backtester.strategies import MomentumStrategy, MeanReversionStrategy
from backtester.validation import walk_forward_validation
from backtester.visualization import plot_backtest_summary

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_ohlcv.csv")


def print_metrics(name: str, summary: dict) -> None:
    print(f"\n--- {name} ---")
    print(f"  Cumulative return:     {summary['cumulative_return']:.2%}")
    print(f"  Annualized return:     {summary['annualized_return']:.2%}")
    print(f"  Annualized volatility: {summary['annualized_volatility']:.2%}")
    print(f"  Sharpe ratio:          {summary['sharpe_ratio']:.2f}")
    print(f"  Sortino ratio:         {summary['sortino_ratio']:.2f}")
    print(f"  Max drawdown:          {summary['max_drawdown']:.2%}")
    print(f"  Max DD duration (bars):{summary['max_drawdown_duration_bars']}")
    print(f"  Number of trades:      {summary['n_trades']}")
    print(f"  Win rate:              {summary['win_rate']:.2%}")
    print(f"  Profit factor:         {summary['profit_factor']:.2f}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(DATA_PATH):
        data = load_csv(DATA_PATH)
        print(f"Loaded {len(data)} bars of sample data from {DATA_PATH}")
    else:
        data = generate_synthetic_ohlcv(n_periods=1000, seed=42)
        print(f"Generated {len(data)} bars of synthetic data")

    strategies = {
        "Momentum (20-day breakout)": MomentumStrategy(lookback=20),
        "Mean Reversion (20-day z-score)": MeanReversionStrategy(window=20, entry_z=2.0, exit_z=0.5),
    }

    for name, strategy in strategies.items():
        engine = BacktestEngine(
            data, strategy,
            initial_capital=100_000,
            commission_bps=5,   # 5 bps ~ 0.05% per trade
            slippage_bps=5,     # 5 bps adverse slippage per trade
        )
        result = engine.run()
        summary = result.summary()
        print_metrics(name, summary)

        fig = plot_backtest_summary(result, title=name)
        out_path = os.path.join(OUTPUT_DIR, f"{name.split()[0].lower()}_summary.png")
        fig.savefig(out_path, dpi=120)
        print(f"  Saved plot -> {out_path}")

    # --- Walk-forward validation example (momentum lookback grid search) ---
    print("\n--- Walk-Forward Validation: Momentum lookback ---")
    wf_result = walk_forward_validation(
        data, MomentumStrategy,
        param_grid={"lookback": [10, 20, 30, 50]},
        train_size=180, test_size=60,
        optimize_metric="sharpe_ratio",
    )
    table = wf_result.summary_table()
    print(table.to_string(index=False))

    agg = wf_result.aggregate_test_metrics()
    print(f"\nMean out-of-sample Sharpe across windows: {agg.get('sharpe_ratio', float('nan')):.2f}")
    print(f"Mean out-of-sample return per window:      {agg.get('cumulative_return', float('nan')):.2%}")


if __name__ == "__main__":
    main()
