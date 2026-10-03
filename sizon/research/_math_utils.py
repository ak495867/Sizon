"""Shared mathematical utilities for pure-Python research calculations."""

from __future__ import annotations
import math


def _clean_pairs(
    x: list[float], y: list[float]
) -> tuple[list[float], list[float]]:
    pairs = [
        (a, b)
        for a, b in zip(x, y)
        if not (math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b))
    ]
    if not pairs:
        return [], []
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: list[float], mean_val: float | None = None) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values) if mean_val is None else mean_val
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(max(0.0, var))


def _correlation(x: list[float], y: list[float]) -> float:
    cx, cy = _clean_pairs(x, y)
    n = len(cx)
    if n < 2:
        return 0.0
    mx, my = _mean(cx), _mean(cy)
    sx, sy = _std(cx, mx), _std(cy, my)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    cov = sum((cx[i] - mx) * (cy[i] - my) for i in range(n)) / (n - 1)
    return max(-1.0, min(1.0, cov / (sx * sy)))
