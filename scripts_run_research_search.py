from pathlib import Path
import csv
import math
import json
import sizon

base = Path("/home/ubuntu/sizon")
data_path = base / "examples" / "research_sample.csv"
with data_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    prev = 100.0
    for i in range(500):
        close = 100 + 0.04 * i + 2.5 * math.sin(i / 9) + 1.2 * math.sin(i / 31)
        open_ = prev
        high = max(open_, close) + 0.5 + 0.1 * abs(math.sin(i))
        low = min(open_, close) - 0.5 - 0.1 * abs(math.cos(i))
        volume = 1000 + 100 * abs(math.sin(i / 13)) + 10 * (i % 7)
        writer.writerow(
            [
                f"2024-01-{(i//24)+1:03d}T{(i%24):02d}:00:00",
                "BTC",
                round(open_, 6),
                round(high, 6),
                round(low, 6),
                round(close, 6),
                round(volume, 6),
            ]
        )
        prev = close

feed = sizon.DataFeed.from_csv(data_path)
out = base / "runs"
run_id = "nsga2_research_20260920"
engine = sizon.Engine(
    feed,
    population=30,
    generations=8,
    seed=42,
    output_dir=out,
    run_id=run_id,
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
    )
)
