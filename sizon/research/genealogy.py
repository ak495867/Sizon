"""Strategy lineage tracking and population-level analytics."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import Counter


@dataclass
class GenealogyEvent:
    child_id: str
    parent_ids: list[str]
    generation: int
    operation: str


class Genealogy:
    def __init__(self):
        self.events = []

    def add(self, child_id, parent_ids, generation, operation):
        self.events.append(GenealogyEvent(child_id, parent_ids, generation, operation))

    def ancestors(self, strategy_id):
        seen = set()
        frontier = [strategy_id]
        while frontier:
            current = frontier.pop()
            for e in self.events:
                if e.child_id == current and current not in seen:
                    seen.add(current)
                    frontier.extend(e.parent_ids)
        return sorted(seen)

    def as_dict(self):
        return [asdict(e) for e in self.events]


def population_analytics(records):
    records = list(records)
    return {
        "strategies": len(records),
        "generations": len({r.get("generation") for r in records}),
        "mean_test_sharpe": sum(
            r.get("test_metrics", {}).get("sharpe", 0) for r in records
        )
        / max(1, len(records)),
        "primitive_usage": Counter(
            r.get("genome", {}).get("name", "binary") for r in records
        ),
        "survived_positive_test": sum(
            r.get("test_metrics", {}).get("sharpe", 0) > 0 for r in records
        ),
    }
