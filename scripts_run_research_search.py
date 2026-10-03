"""Research evolutionary search script using a synthetic sine-wave price feed.

Generates a 500-bar synthetic dataset with realistic multi-frequency price dynamics,
runs an NSGA-II search, and produces reports. The synthetic data generator is also
available as a reusable fixture for testing (see tests/conftest.py).

Paths are resolved relative to this script — works on any OS.

Usage:
    python scripts_run_research_search.py [--output-dir RUNS_DIR] [--run-id RUN_ID]
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import sizon

_HERE = Path(__file__).resolve().parent


def generate_synthetic_csv(output_path: Path, n_bars: int = 500, seed_price: float = 100.0) -> Path:
    """Write a synthetic OHLCV CSV with multi-frequency price dynamics.

    The price model combines a linear trend with two sine waves at different
    frequencies to simulate realistic-looking but fully controlled market data.

    Args:
        output_path: Where to write the CSV. Parent directory must exist.
        n_bars: Number of bars to generate.
        seed_price: Starting price level.

    Returns:
        Path to the written CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prev = seed_price
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        for i in range(n_bars):
            close = seed_price + 0.04 * i + 2.5 * math.sin(i / 9) + 1.2 * math.sin(i / 31)
            open_ = prev
            high = max(open_, close) + 0.5 + 0.1 * abs(math.sin(i))
            low = min(open_, close) - 0.5 - 0.1 * abs(math.cos(i))
            volume = 1000 + 100 * abs(math.sin(i / 13)) + 10 * (i % 7)
            # Timestamp: daily bars starting 2024-01-01
            day = (i // 24) + 1
            hour = i % 24
            ts = f"2024-{day // 30 + 1:02d}-{day % 30 + 1:02d}T{hour:02d}:00:00"
            writer.writerow([
                ts, "BTC",
                round(open_, 6), round(high, 6), round(low, 6),
                round(close, 6), round(volume, 6),
            ])
            prev = close
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NSGA-II research search on synthetic sine-wave data"
    )
    parser.add_argument("--output-dir", default=str(_HERE / "runs"), help="Output directory")
    parser.add_argument("--run-id", default="nsga2_research", help="Run identifier")
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bars", type=int, default=500, help="Number of synthetic bars")
    parser.add_argument(
        "--data-output",
        default=None,
        help="Where to write the synthetic CSV (default: tmp dir inside output-dir)",
    )
    args = parser.parse_args()

    # Write synthetic data to a temporary location (never overwrite examples/)
    out_dir = Path(args.output_dir)
    data_path = Path(args.data_output) if args.data_output else out_dir / "synthetic_research.csv"
    generate_synthetic_csv(data_path, n_bars=args.n_bars)

    feed = sizon.DataFeed.from_csv(str(data_path))
    engine = sizon.Engine(
        feed,
        population=args.population,
        generations=args.generations,
        seed=args.seed,
        output_dir=str(out_dir),
        run_id=args.run_id,
        robustness_simulations=10,
    )
    best = engine.run()
    run_dir = engine.store.path
    sizon.build_report(run_dir)
    sizon.build_top_report(run_dir, top_n=15)
    records = [
        json.loads(p.read_text()) for p in sorted((run_dir / "strategies").glob("*.json"))
    ]
    ranked = sizon.compare(records)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "rows": len(feed.rows),
                "strategies_saved": len(records),
                "unique_strategies": len(ranked),
                "top": ranked[:5],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
