"""Local-first market data loading, quality checks, and reproducibility manifests."""

from __future__ import annotations
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REQUIRED = ("timestamp", "symbol", "open", "high", "low", "close", "volume")
PRICE_FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class DataQualityReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self):
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
            "ok": not self.errors,
        }


@dataclass
class DataFeed:
    rows: list[dict]
    source: str = "memory"
    timezone: str = "UTC"

    @classmethod
    def from_csv(cls, path: str | Path) -> "DataFeed":
        path = Path(path)
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        return cls._validated(rows, str(path))

    @classmethod
    def from_parquet(cls, path: str | Path) -> "DataFeed":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Parquet support requires: pip install sizon[parquet]"
            ) from exc
        return cls._validated(pq.read_table(path).to_pylist(), str(path))

    @classmethod
    def from_path(cls, path: str | Path) -> "DataFeed":
        path = Path(path)
        if path.suffix.lower() == ".parquet":
            return cls.from_parquet(path)
        if path.suffix.lower() == ".csv":
            return cls.from_csv(path)
        raise ValueError(f"Unsupported data format: {path.suffix}")

    @classmethod
    def _validated(cls, rows: Iterable[dict], source: str) -> "DataFeed":
        rows = list(rows)
        if not rows:
            raise ValueError("Data feed is empty")
        missing = set(REQUIRED) - set(rows[0])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        normalized = []
        for n, row in enumerate(rows, 1):
            item = dict(row)
            try:
                for key in PRICE_FIELDS:
                    item[key] = float(item[key])
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid numeric value on row {n}: {exc}") from exc
            item["timestamp"] = str(item["timestamp"])
            item["symbol"] = str(item["symbol"])
            normalized.append(item)
        feed = cls(normalized, source=source)
        report = feed.quality_report()
        if report.errors:
            raise ValueError("Data quality errors: " + "; ".join(report.errors))
        return feed

    def quality_report(self) -> DataQualityReport:
        report = DataQualityReport(checks={"rows": len(self.rows)})
        seen = set()
        previous_by_symbol = {}
        for i, row in enumerate(self.rows, 1):
            key = (row["symbol"], row["timestamp"])
            if key in seen:
                report.errors.append(f"duplicate timestamp for {key[0]} at {key[1]}")
            seen.add(key)
            o, h, l, c, v = (row[k] for k in PRICE_FIELDS)
            if h < max(o, c) or l > min(o, c) or h < l:
                report.errors.append(f"OHLC inconsistency on row {i}")
            if any(not math.isfinite(x) for x in (o, h, l, c, v)):
                report.errors.append(f"non-finite value on row {i}")
            if v < 0:
                report.errors.append(f"negative volume on row {i}")
            prev = previous_by_symbol.get(row["symbol"])
            if prev and row["timestamp"] < prev:
                report.warnings.append(f"timestamps are not sorted for {row['symbol']}")
            previous_by_symbol[row["symbol"]] = row["timestamp"]
        report.checks.update(
            {
                "symbols": len({r["symbol"] for r in self.rows}),
                "duplicates": len(self.rows) - len(seen),
            }
        )
        return report

    def validate(self) -> dict:
        return {
            "rows": len(self.rows),
            "symbols": sorted({r["symbol"] for r in self.rows}),
            "columns": sorted(self.rows[0]),
            "quality": self.quality_report().as_dict(),
        }

    def column(self, name: str) -> list[float]:
        return [float(r[name]) for r in self.rows]

    def split(self, fraction: float = 0.7) -> tuple["DataFeed", "DataFeed"]:
        cut = max(1, min(len(self.rows) - 1, int(len(self.rows) * fraction)))
        return DataFeed(self.rows[:cut], self.source, self.timezone), DataFeed(
            self.rows[cut:], self.source, self.timezone
        )

    def manifest(self) -> dict:
        payload = json.dumps(self.rows, sort_keys=True, default=str).encode()
        return {
            "source": self.source,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": len(self.rows),
            "symbols": sorted({r["symbol"] for r in self.rows}),
            "timezone": self.timezone,
            "columns": sorted(self.rows[0]),
        }
