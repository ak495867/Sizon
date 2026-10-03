"""Filesystem experiment store: every candidate is persisted, good or bad."""

from __future__ import annotations
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from sizon.core.strategy import StrategyRecord


class ExperimentStore:
    def __init__(self, root: str | Path = "runs", run_id: str | None = None):
        self.root = Path(root)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.root / self.run_id
        self.strategies = self.path / "strategies"
        self.strategies.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write_manifest(self, manifest: dict):
        self._write(self.path / "manifest.json", manifest)

    def save(self, record: StrategyRecord) -> Path:
        payload = record.as_dict()
        target = (
            self.strategies
            / f"{record.strategy_id}_g{record.generation}_i{record.index}.json"
        )
        with self._lock:
            self._write(target, payload)
            with open(self.path / "strategies.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, sort_keys=True) + "\n")
        return target

    def finalize(self, summary: dict):
        self._write(self.path / "summary.json", summary)

    @staticmethod
    def _write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        tmp_path.replace(path)  # atomic on POSIX, best-effort on Windows
