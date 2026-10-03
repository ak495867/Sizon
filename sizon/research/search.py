"""Search utilities for multi-objective evolutionary research."""

from __future__ import annotations
import random
from sizon.core.expression import Node, Binary
from sizon.core.strategy import strategy_id as _sid
from sizon.research.nsga2 import dominates, fronts as _nsga_fronts
from sizon.research.nsga2 import save_checkpoint, load_checkpoint  # noqa: F401


def pareto_front(candidates):
    """Return non-dominated candidates from a flat list of score dicts."""
    return [
        c for c in candidates
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


def bloat_penalty(node: Node, max_complexity: int = 25) -> int:
    """Return penalty (>=0) for genomes exceeding max_complexity nodes."""
    return max(0, node.complexity() - max_complexity)


def diversity_score(genomes) -> float:
    """Fraction of structurally unique genomes in the population."""
    unique = {_sid(g) for g in genomes}
    return len(unique) / max(1, len(genomes))
