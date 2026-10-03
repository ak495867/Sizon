"""Small deterministic NSGA-II implementation for strategy score dictionaries."""

from __future__ import annotations
import json
from pathlib import Path
from sizon.core.strategy import genome_to_dict, genome_from_dict


def dominates(a, b):
    return (
        a["sharpe"] >= b["sharpe"]
        and a["max_drawdown"] <= b["max_drawdown"]
        and a["complexity"] <= b["complexity"]
        and (
            a["sharpe"] > b["sharpe"]
            or a["max_drawdown"] < b["max_drawdown"]
            or a["complexity"] < b["complexity"]
        )
    )


def fronts(candidates):
    result = []
    remaining = list(candidates)
    while remaining:
        front = [
            c
            for c in remaining
            if not any(d is not c and dominates(d.scores, c.scores) for d in remaining)
        ]
        result.append(front)
        remaining = [c for c in remaining if c not in front]
    return result


def crowding(front: list) -> dict[int, float]:
    """Return crowding distance keyed by candidate's position index in front."""
    if not front:
        return {}
    distance: dict[int, float] = {i: 0.0 for i in range(len(front))}
    for key, reverse in (("sharpe", True), ("max_drawdown", False), ("complexity", False)):
        order = sorted(range(len(front)), key=lambda i: front[i].scores[key], reverse=reverse)
        distance[order[0]] = distance[order[-1]] = float("inf")
        span = abs(front[order[0]].scores[key] - front[order[-1]].scores[key]) or 1.0
        for rank in range(1, len(order) - 1):
            i = order[rank]
            prev_i = order[rank - 1]
            next_i = order[rank + 1]
            distance[i] += abs(front[prev_i].scores[key] - front[next_i].scores[key]) / span
    return distance


class NSGA2:
    def select(self, candidates, size):
        chosen = []
        for front in fronts(candidates):
            if len(chosen) + len(front) <= size:
                chosen.extend(front)
            else:
                cd = crowding(front)
                # sort by crowding distance (higher = more spread out = better)
                remaining_needed = size - len(chosen)
                sorted_front = sorted(range(len(front)), key=lambda i: cd.get(i, 0.0), reverse=True)
                chosen.extend(front[i] for i in sorted_front[:remaining_needed])
                break
        return chosen

    def pareto_front(self, candidates):
        return fronts(candidates)[0] if candidates else []


def save_checkpoint(path, population, generation, seed):
    Path(path).write_text(
        json.dumps(
            {
                "version": 1,
                "generation": generation,
                "seed": seed,
                "population": [genome_to_dict(g) for g in population],
            },
            indent=2,
        )
    )


def load_checkpoint(path):
    payload = json.loads(Path(path).read_text())
    payload["population"] = [genome_from_dict(g) for g in payload["population"]]
    return payload
