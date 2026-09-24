from pathlib import Path
import sizon

root = Path("/home/ubuntu/sizon/runs")
run_id = "nsga2_full_20260919"
feed = sizon.DataFeed.from_csv("/home/ubuntu/sizon/examples/sample.csv")
engine = sizon.Engine(
    feed,
    population=20,
    generations=5,
    seed=42,
    output_dir=root,
    run_id=run_id,
    robustness_simulations=30,
)
best = engine.run()
run_dir = engine.store.path
report = sizon.build_report(run_dir)
top_report = sizon.build_top_report(run_dir, top_n=10)
records = [
    __import__("json").loads(p.read_text())
    for p in sorted((run_dir / "strategies").glob("*.json"))
]
ranked = sizon.compare(records)
print(
    {
        "run_dir": str(run_dir),
        "strategies_saved": len(records),
        "top_report": str(top_report),
        "best": ranked[:3],
    }
)
