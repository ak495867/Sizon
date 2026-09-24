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


def crowding(front):
    if not front:
        return {}
    distance = {id(c): 0.0 for c in front}
    for key, reverse in (
        ("sharpe", True),
        ("max_drawdown", False),
        ("complexity", False),
    ):
        ordered = sorted(front, key=lambda c: c.scores[key], reverse=reverse)
        distance[id(ordered[0])] = distance[id(ordered[-1])] = float("inf")
        span = abs(ordered[0].scores[key] - ordered[-1].scores[key]) or 1
        for i in range(1, len(ordered) - 1):
            distance[id(ordered[i])] += (
                abs(ordered[i - 1].scores[key] - ordered[i + 1].scores[key]) / span
            )
    return distance


class NSGA2:
    def select(self, candidates, size):
        chosen = []
        for front in fronts(candidates):
            if len(chosen) + len(front) <= size:
                chosen.extend(front)
            else:
                chosen.extend(
                    sorted(front, key=lambda c: crowding(front)[id(c)], reverse=True)[
                        : size - len(chosen)
                    ]
                )
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
