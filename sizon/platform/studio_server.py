import json
import os
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from sizon.platform.production import HealthMonitor


class StudioHandler(SimpleHTTPRequestHandler):
    monitor = HealthMonitor()
    runs_dir = Path("runs")

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._send_json(self.monitor.health())
            return

        if path == "/api/runs":
            runs = []
            target_dirs = [self.runs_dir, Path(".")]
            seen = set()
            for r_dir in target_dirs:
                if not r_dir.exists():
                    continue
                for p in sorted(r_dir.iterdir()):
                    if (
                        p.is_dir()
                        and (p / "summary.json").exists()
                        and p.name not in seen
                    ):
                        seen.add(p.name)
                        try:
                            summary = json.loads((p / "summary.json").read_text())
                            strategies_count = summary.get(
                                "strategies_saved",
                                len(list((p / "strategies").glob("*.json"))),
                            )
                            has_rep = (
                                p / "top_strategies.html"
                            ).exists() or (p / "report.html").exists()
                            rep_url = (
                                f"/{p.name}/top_strategies.html"
                                if (p / "top_strategies.html").exists()
                                else f"/{p.name}/report.html"
                            )
                            runs.append(
                                {
                                    "run_id": summary.get("run_id", p.name),
                                    "strategies_saved": strategies_count,
                                    "best_train_metrics": summary.get(
                                        "best_train_metrics", {}
                                    ),
                                    "has_report": has_rep,
                                    "report_url": rep_url,
                                }
                            )
                        except Exception:
                            continue
            self._send_json(runs)
            return

        if path.startswith("/api/runs/"):
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 3:
                run_id = parts[2]
                sub = parts[3] if len(parts) > 3 else "summary"
                for r_dir in [self.runs_dir, Path(".")]:
                    target = r_dir / run_id
                    if target.is_dir() and (target / "summary.json").exists():
                        try:
                            if sub == "pareto":
                                strat_files = sorted(
                                    (target / "strategies").glob("*.json")
                                )
                                pareto = []
                                for sf in strat_files[:200]:
                                    rec = json.loads(sf.read_text())
                                    pareto.append(
                                        {
                                            "id": rec.get(
                                                "strategy_id", sf.stem
                                            ),
                                            "expression": rec.get(
                                                "expression", ""
                                            ),
                                            "sharpe": rec.get(
                                                "test_metrics", {}
                                            ).get("sharpe", 0.0),
                                            "drawdown": rec.get(
                                                "test_metrics", {}
                                            ).get("max_drawdown", 0.0),
                                            "complexity": rec.get(
                                                "complexity", 1
                                            ),
                                        }
                                    )
                                self._send_json(pareto)
                                return

                            summary = json.loads(
                                (target / "summary.json").read_text()
                            )
                            manifest = (
                                json.loads(
                                    (target / "manifest.json").read_text()
                                )
                                if (target / "manifest.json").exists()
                                else {}
                            )
                            strat_files = sorted(
                                (target / "strategies").glob("*.json")
                            )
                            loaded = [
                                json.loads(sf.read_text())
                                for sf in strat_files[:30]
                            ]
                            loaded.sort(
                                key=lambda x: x.get("test_metrics", {}).get(
                                    "sharpe", 0
                                ),
                                reverse=True,
                            )
                            self._send_json(
                                {
                                    "summary": summary,
                                    "manifest": manifest,
                                    "top_strategies": loaded[:5],
                                }
                            )
                            return
                        except Exception as exc:
                            self._send_json({"error": str(exc)}, status=500)
                            return
                self._send_json({"error": f"run {run_id} not found"}, status=404)
                return

        if path in {"/", "/index.html"}:
            for ui_path in (
                Path("studio-ui/index.html"),
                Path("../studio-ui/index.html"),
            ):
                if ui_path.exists():
                    content = ui_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return

        return super().do_GET()


def serve(directory="runs", host="127.0.0.1", port=8765):
    directory = str(Path(directory).resolve())
    os.chdir(directory)
    return ThreadingHTTPServer((host, port), StudioHandler)
