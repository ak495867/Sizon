"""Sizon CLI: Comprehensive terminal toolkit for evolutionary alpha research."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from sizon.core.strategy import genome_from_dict
from sizon.data.data import DataFeed
from sizon.platform.studio_server import serve
from sizon.research.ensemble import AlphaEnsemble
from sizon.research.evolution import Engine
from sizon.research.factor_analytics import FactorAnalytics
from sizon.simulation.orderbook import microstructure_gauntlet


def main():
    parser = argparse.ArgumentParser(
        prog="sizon", description="Institutional Evolutionary Alpha Research Platform"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. validate
    p_validate = sub.add_parser("validate", help="Validate market data feed integrity")
    p_validate.add_argument("path", help="Path to CSV or Parquet market data file")

    # 2. run
    p_run = sub.add_parser("run", help="Run multi-objective evolutionary alpha search")
    p_run.add_argument("path", help="Path to market data feed")
    p_run.add_argument("--output-dir", default="runs")
    p_run.add_argument("--run-id")
    p_run.add_argument("--population", type=int, default=20)
    p_run.add_argument("--generations", type=int, default=5)
    p_run.add_argument("--seed", type=int, default=7)
    p_run.add_argument("--crossover", type=float, default=0.7)
    p_run.add_argument("--mutation", type=float, default=0.3)
    p_run.add_argument("--workers", type=int, default=1)

    # 3. ensemble
    p_ens = sub.add_parser(
        "ensemble",
        help="Combine Pareto-front alphas into a Risk-Parity portfolio",
    )
    p_ens.add_argument(
        "run_dir",
        help="Path to experiment run directory containing strategies",
    )
    p_ens.add_argument(
        "data_path", help="Path to market data feed for portfolio evaluation"
    )
    p_ens.add_argument(
        "--method",
        choices=[
            "risk_parity",
            "min_variance",
            "inverse_variance",
            "equal_weight",
        ],
        default="risk_parity",
    )
    p_ens.add_argument("--max-corr", type=float, default=0.65)
    p_ens.add_argument("--top-n", type=int, default=10)

    # 4. factor
    p_factor = sub.add_parser(
        "factor",
        help="Generate Alphalens-grade Factor Analytics (Rank IC & Quantiles)",
    )
    p_factor.add_argument("run_dir", help="Path to experiment run directory")
    p_factor.add_argument("data_path", help="Path to market data feed")
    p_factor.add_argument("--quantiles", type=int, default=5)

    # 5. gauntlet
    p_gauntlet = sub.add_parser(
        "gauntlet",
        help="Stress test strategies against order-book microstructure",
    )
    p_gauntlet.add_argument("run_dir", help="Path to experiment run directory")
    p_gauntlet.add_argument("data_path", help="Path to market data feed")
    p_gauntlet.add_argument("--top-n", type=int, default=5)

    # 6. studio
    p_studio = sub.add_parser("studio", help="Launch Sizon Studio web dashboard")
    p_studio.add_argument("--directory", default="runs")
    p_studio.add_argument("--port", type=int, default=8765)
    p_studio.add_argument("--host", default="127.0.0.1")

    args = parser.parse_args()

    if args.command == "validate":
        feed = DataFeed.from_path(args.path)
        print(json.dumps(feed.validate(), indent=2))
        return

    if args.command == "run":
        feed = DataFeed.from_path(args.path)
        engine = Engine(
            feed,
            population=args.population,
            generations=args.generations,
            seed=args.seed,
            output_dir=args.output_dir,
            run_id=args.run_id,
            crossover_rate=args.crossover,
            mutation_rate=args.mutation,
            workers=args.workers,
        )
        best = engine.run()
        print(
            json.dumps(
                {
                    "run_id": engine.store.run_id,
                    "run_path": str(engine.store.path),
                    "best_train_metrics": best.scores if best else {},
                    "strategies_saved": args.population * args.generations,
                },
                indent=2,
            )
        )
        return

    if args.command == "ensemble":
        run_path = Path(args.run_dir)
        feed = DataFeed.from_path(args.data_path)
        strat_files = sorted((run_path / "strategies").glob("*.json"))
        records = [json.loads(sf.read_text()) for sf in strat_files]
        records.sort(
            key=lambda r: r.get("test_metrics", {}).get("sharpe", 0), reverse=True
        )
        genomes = [genome_from_dict(r["genome"]) for r in records[: args.top_n]]
        ens = AlphaEnsemble(method=args.method, max_correlation=args.max_corr)
        result = ens.fit(feed, genomes)
        print(
            json.dumps(
                {
                    "ensemble_method": args.method,
                    "metrics": result.metrics,
                    "weights": result.weights,
                    "diversification_ratio": result.diversification_ratio,
                },
                indent=2,
            )
        )
        return

    if args.command == "factor":
        run_path = Path(args.run_dir)
        feed = DataFeed.from_path(args.data_path)
        strat_files = sorted((run_path / "strategies").glob("*.json"))
        if not strat_files:
            print(json.dumps({"error": "No strategies found in run directory"}))
            return
        records = [json.loads(sf.read_text()) for sf in strat_files]
        records.sort(
            key=lambda r: r.get("test_metrics", {}).get("sharpe", 0), reverse=True
        )
        top_genome = genome_from_dict(records[0]["genome"])
        analytics = FactorAnalytics.analyze(
            feed, top_genome, quantiles=args.quantiles
        )
        print(json.dumps(analytics, indent=2))
        return

    if args.command == "gauntlet":
        run_path = Path(args.run_dir)
        feed = DataFeed.from_path(args.data_path)
        strat_files = sorted((run_path / "strategies").glob("*.json"))
        records = [json.loads(sf.read_text()) for sf in strat_files]
        records.sort(
            key=lambda r: r.get("test_metrics", {}).get("sharpe", 0), reverse=True
        )
        genomes = [genome_from_dict(r["genome"]) for r in records[: args.top_n]]
        gauntlet_results = microstructure_gauntlet(feed, genomes)
        print(
            json.dumps(
                {
                    "strategies_tested": len(genomes),
                    "passed_gauntlet": sum(1 for g in gauntlet_results if g["passed"]),
                    "results": gauntlet_results,
                },
                indent=2,
            )
        )
        return

    if args.command == "studio":
        print(
            f"Starting Sizon Studio at http://{args.host}:{args.port} (Serving {args.directory})..."
        )
        server = serve(directory=args.directory, host=args.host, port=args.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()
            print("Studio server stopped.")


if __name__ == "__main__":
    main()
