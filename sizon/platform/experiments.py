"""Filesystem experiment store: every candidate is persisted, good or bad."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from sizon.core.strategy import StrategyRecord

class ExperimentStore:
    def __init__(self, root: str | Path = "runs", run_id: str | None = None):
        self.root = Path(root); self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.root / self.run_id; self.strategies = self.path / "strategies"
        self.strategies.mkdir(parents=True, exist_ok=True)
    def write_manifest(self, manifest: dict): self._write(self.path / "manifest.json", manifest)
    def save(self, record: StrategyRecord) -> Path:
        payload = record.as_dict()
        target = self.strategies / f"{record.strategy_id}_g{record.generation}_i{record.index}.json"
        self._write(target, payload)
        with open(self.path / "strategies.jsonl", "a", encoding="utf-8") as fh: fh.write(json.dumps(payload, sort_keys=True) + "\n")
        return target
    def finalize(self, summary: dict): self._write(self.path / "summary.json", summary)
    @staticmethod
    def _write(path: Path, payload: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh: json.dump(payload, fh, indent=2, sort_keys=True, default=str)
