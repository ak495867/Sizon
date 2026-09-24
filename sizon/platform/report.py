"""Human-readable run reporting."""
from __future__ import annotations
import json,html
from pathlib import Path
from sizon.research.comparison import compare
def build_report(run_dir):
    run_dir=Path(run_dir); summary=json.loads((run_dir/"summary.json").read_text()); manifest=json.loads((run_dir/"manifest.json").read_text()); records=[]
    for path in sorted((run_dir/"strategies").glob("*.json")): records.append(json.loads(path.read_text()))
    rows="\n".join(f"<tr><td>{html.escape(r['strategy_id'])}</td><td>{r['generation']}</td><td>{r['test_metrics'].get('sharpe',0):.4f}</td><td>{r['test_metrics'].get('max_drawdown',0):.4f}</td></tr>" for r in records)
    content=f"<html><body><h1>Sizon Research Report</h1><pre>{html.escape(json.dumps(manifest,indent=2))}</pre><p>Strategies evaluated: {len(records)}</p><table><tr><th>ID</th><th>Generation</th><th>Test Sharpe</th><th>Test Drawdown</th></tr>{rows}</table></body></html>"
    (run_dir/"report.html").write_text(content); (run_dir/"report.md").write_text(f"# Sizon Research Report\n\nStrategies evaluated: {len(records)}\n\n```json\n{json.dumps(summary,indent=2)}\n```\n")
    return run_dir/"report.html"

def build_top_report(run_dir, top_n=10):
    run_dir=Path(run_dir); records=[json.loads(path.read_text()) for path in sorted((run_dir/"strategies").glob("*.json"))]
    ranked=compare(records)[:top_n]
    rows="\n".join(f"<tr><td>{html.escape(str(r['strategy_id']))}</td><td>{html.escape(str(r['expression']))}</td><td>{r['test_sharpe']:.4f}</td><td>{r['test_drawdown']:.4f}</td><td>{r['robustness_score']:.2f}</td></tr>" for r in ranked)
    content=f"<html><head><meta charset='utf-8'><title>Sizon Top Strategies</title><style>body{{font-family:system-ui;margin:2rem}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:.5rem;text-align:left}}th{{background:#f3f4f6}}</style></head><body><h1>Sizon Top Strategies</h1><p>Ranked by multi-objective robustness score, test Sharpe, and test drawdown. Candidates: {len(records)}. Showing: {len(ranked)}.</p><table><tr><th>Strategy</th><th>Expression</th><th>Test Sharpe</th><th>Test Drawdown</th><th>Robustness</th></tr>{rows}</table></body></html>"
    target=run_dir/"top_strategies.html"; target.write_text(content); return target
