"""Known-answer/reference harness for research statistics.

The harness compares Sizon outputs to supplied reference values. It does not
claim a result is academically validated until a fixture passes.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    actual: float
    expected: float
    tolerance: float = 1e-6

    def passes(self):
        return abs(self.actual - self.expected) <= self.tolerance


def validate_cases(cases):
    cases = list(cases)
    results = [
        {
            "name": c.name,
            "actual": c.actual,
            "expected": c.expected,
            "passed": c.passes(),
        }
        for c in cases
    ]
    return {"passed": all(r["passed"] for r in results), "cases": results}


def fixture_constant_returns():
    return ReferenceCase("constant_returns_mean", 1.0, 1.0)
