"""Dependency-light statistical diagnostics for research claims."""

from __future__ import annotations
import math
import random


def _median(values: list) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def bootstrap_mean(values: list[float], simulations: int = 1000, seed: int = 7) -> dict:
    values = list(values)
    if not values:
        return {}
    rng = random.Random(seed)
    means = []
    for _ in range(simulations):
        means.append(sum(rng.choice(values) for _ in values) / max(1, len(values)))
    means.sort()
    return {
        "mean": sum(values) / max(1, len(values)),
        "lower_95": means[int(0.025 * len(means))],
        "upper_95": means[int(0.975 * len(means)) - 1],
        "simulations": simulations,
    }


def deflated_sharpe_ratio(
    sharpe: float, trials: int, expected_mean: float = 0.0, variance: float = 1.0
) -> float:
    # Expected maximum SR = expected_mean + sqrt(variance * 2 * log(trials))
    # Approximation of Euler-Mascheroni formula
    expected_max = expected_mean + math.sqrt(variance * 2 * math.log(max(1, trials)))
    return sharpe - expected_max


def probability_of_backtest_overfitting(sharpes: list[float]) -> dict:
    sharpes = list(sharpes)
    ifrom = max(sharpes) if sharpes else 0
    median = _median(sharpes)
    return {
        "trials": len(sharpes),
        "best_sharpe": ifrom,
        "median_sharpe": median,
        "estimated_overfit_probability": sum(s <= 0 for s in sharpes)
        / max(1, len(sharpes)),
    }


def multiple_testing_adjust(pvalues: list[float], method: str = "bonferroni") -> list[float]:
    pvalues = list(pvalues)
    n = len(pvalues)
    return (
        [min(1, p * n) for p in pvalues]
        if method == "bonferroni"
        else [min(1, p * n / max(1, (i + 1))) for i, p in enumerate(sorted(pvalues))]
    )


def white_reality_check(returns_matrix: list[list[float]], simulations: int = 500, seed: int = 7) -> dict:
    matrix = [list(x) for x in returns_matrix]
    best = max((sum(x) / max(1, len(x)) for x in matrix), default=0)
    rng = random.Random(seed)
    null = []
    for _ in range(simulations):
        null.append(
            max(
                (sum(rng.choice(x) for _ in x) / max(1, len(x)) for x in matrix),
                default=0,
            )
        )
    return {
        "best_mean_return": best,
        "p_value": sum(v >= best for v in null) / max(1, len(null)),
        "simulations": simulations,
    }


def spa_test(returns_matrix: list[list[float]], simulations: int = 500, seed: int = 7) -> dict:
    result = white_reality_check(returns_matrix, simulations, seed)
    result["method"] = "superior_predictive_ability"
    return result


def regime_labels(returns: list[float], window: int = 20) -> list[str]:
    if not returns:
        return []
    out = []
    for i in range(len(returns)):
        chunk = returns[max(0, i - window + 1) : i + 1]
        vol = (
            math.sqrt(
                sum((x - sum(chunk) / len(chunk)) ** 2 for x in chunk)
                / max(1, len(chunk))
            )
            if chunk
            else 0
        )
        mean = sum(chunk) / max(1, len(chunk))
        out.append(
            "high_vol"
            if vol > abs(mean) * 2
            else "uptrend" if mean > 0 else "downtrend"
        )
    return out
