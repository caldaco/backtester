"""Regenerate the sample OHLCV dataset committed under data/sample_ohlcv.csv."""

import os

from backtester.data import generate_synthetic_ohlcv

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_ohlcv.csv")


def main() -> None:
    df = generate_synthetic_ohlcv(
        n_periods=1000, start="2018-01-01", start_price=100.0,
        annual_drift=0.08, annual_vol=0.28, seed=42,
    )
    df.round(4).to_csv(OUT_PATH)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
