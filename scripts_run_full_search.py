"""Full evolutionary search script.

Runs a complete NSGA-II search on the sample data feed and generates reports.
Paths are resolved relative to this script, so it works on any OS.

Usage:
    python scripts_run_full_search.py [--output-dir RUNS_DIR] [--run-id RUN_ID]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import sizon

_HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full NSGA-II search on sample data")
    parser.add_argument("--output-dir", default=str(_HERE / "runs"), help="Output directory")
    parser.add_argument("--run-id", default="nsga2_full", help="Run identifier")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data",
        default=str(_HERE / "examples" / "sample.csv"),
        help="Path to market data CSV",
    )
    args = parser.parse_args()

    feed = sizon.DataFeed.from_csv(args.data)
    engine = sizon.Engine(
        feed,
        population=args.population,
        generations=args.generations,
        seed=args.seed,
        output_dir=args.output_dir,
        run_id=args.run_id,
        robustness_simulations=30,
    )
    best = engine.run()
    run_dir = engine.store.path
    report = sizon.build_report(run_dir)
    top_report = sizon.build_top_report(run_dir, top_n=10)
    records = [
        json.loads(p.read_text()) for p in sorted((run_dir / "strategies").glob("*.json"))
    ]
    ranked = sizon.compare(records)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "strategies_saved": len(records),
                "report": str(report),
                "top_report": str(top_report),
                "best": ranked[:3],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
