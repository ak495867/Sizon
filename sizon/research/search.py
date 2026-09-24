"""Search utilities for multi-objective evolutionary research."""

from __future__ import annotations
import json
import random
from pathlib import Path
from sizon.core.expression import Node, Binary


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


def pareto_front(candidates):
    return [
        c
        for c in candidates
        if not any(d is not c and dominates(d.scores, c.scores) for d in candidates)
    ]


def structural_crossover(a: Node, b: Node, rng=None) -> Node:
    rng = rng or random.Random(7)
    if isinstance(a, Binary) and isinstance(b, Binary):
        return (
            Binary(a.op, a.left, b.right)
            if rng.random() < 0.5
            else Binary(b.op, b.left, a.right)
        )
    return a


def bloat_penalty(node: Node, max_complexity=25):
    return max(0, node.complexity() - max_complexity)


def diversity_score(genomes):
    return len({repr(g) for g in genomes}) / max(1, len(genomes))


def save_checkpoint(path, population, generation, seed):
    Path(path).write_text(
        json.dumps(
            {
                "generation": generation,
                "seed": seed,
                "population_size": len(population),
            },
            indent=2,
        )
    )


def load_checkpoint(path):
    return json.loads(Path(path).read_text())
