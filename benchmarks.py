"""Small benchmark harness for measuring primitive and evaluation throughput."""

from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter


@dataclass
class BenchmarkResult:
    name: str
    seconds: float
    operations: int

    @property
    def ops_per_second(self):
        return self.operations / max(self.seconds, 1e-12)


def benchmark(name, fn, operations=1):
    start = perf_counter()
    fn()
    elapsed = perf_counter() - start
    return BenchmarkResult(name, elapsed, operations)


def enforce_budget(result, max_seconds):
    return {
        "passed": result.seconds <= max_seconds,
        "name": result.name,
        "seconds": result.seconds,
        "budget": max_seconds,
        "ops_per_second": result.ops_per_second,
    }
