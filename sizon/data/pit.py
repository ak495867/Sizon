"""Point-in-time data contracts that prevent future metadata leakage."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CorporateAction:
    effective_at: str
    symbol: str
    split_factor: float = 1.0
    cash_dividend: float = 0.0


@dataclass(frozen=True)
class FuturesRoll:
    effective_at: str
    root: str
    old_contract: str
    new_contract: str
    adjustment: float = 0.0


@dataclass
class PointInTimeDataset:
    rows: list[dict]
    actions: list[CorporateAction] | None = None
    rolls: list[FuturesRoll] | None = None
    as_of: str | None = None

    def visible_rows(self, as_of: str | None = None):
        cutoff = as_of or self.as_of
        if not cutoff:
            return list(self.rows)
        return [
            r
            for r in self.rows
            if str(r.get("available_at", r.get("timestamp"))) <= cutoff
        ]

    def apply_actions(self, rows=None):
        out = [dict(r) for r in (rows or self.visible_rows())]
        for action in self.actions or []:
            for row in out:
                if (
                    row.get("symbol") == action.symbol
                    and row.get("timestamp", "") < action.effective_at
                ):
                    for key in ("open", "high", "low", "close"):
                        row[key] = (
                            float(row[key]) * action.split_factor - action.cash_dividend
                        )
        return out

    def manifest(self):
        return {
            "point_in_time": True,
            "as_of": self.as_of,
            "rows": len(self.rows),
            "actions": len(self.actions or []),
            "rolls": len(self.rolls or []),
        }


class ConnectorRegistry:
    def __init__(self):
        self._connectors = {}

    def register(self, name, loader, metadata=None):
        self._connectors[name] = {"loader": loader, "metadata": metadata or {}}

    def load(self, name, **kwargs):
        if name not in self._connectors:
            raise KeyError(name)
        return self._connectors[name]["loader"](**kwargs)

    def names(self):
        return sorted(self._connectors)
