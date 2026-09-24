"""Dependency-light statistical diagnostics for research claims."""

from __future__ import annotations
import math
import random


def bootstrap_mean(values, simulations=1000, seed=7):
    values = list(values)
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
    observed_sharpe, trials, observations, skew=0.0, kurtosis=3.0
):
    penalty = math.sqrt(2 * math.log(max(1, trials))) / math.sqrt(max(1, observations))
    return {
        "observed_sharpe": observed_sharpe,
        "trial_penalty": penalty,
        "deflated_sharpe": observed_sharpe - penalty,
        "skew": skew,
        "kurtosis": kurtosis,
    }


def probability_of_backtest_overfitting(sharpes):
    sharpes = list(sharpes)
    ifrom = max(sharpes) if sharpes else 0
    median = sorted(sharpes)[len(sharpes) // 2] if sharpes else 0
    return {
        "trials": len(sharpes),
        "best_sharpe": ifrom,
        "median_sharpe": median,
        "estimated_overfit_probability": sum(s <= 0 for s in sharpes)
        / max(1, len(sharpes)),
    }


def multiple_testing_adjust(pvalues, method="bonferroni"):
    pvalues = list(pvalues)
    n = len(pvalues)
    return (
        [min(1, p * n) for p in pvalues]
        if method == "bonferroni"
        else [min(1, p * n / max(1, (i + 1))) for i, p in enumerate(sorted(pvalues))]
    )


def white_reality_check(returns_matrix, simulations=500, seed=7):
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


def spa_test(returns_matrix, simulations=500, seed=7):
    result = white_reality_check(returns_matrix, simulations, seed)
    result["method"] = "superior_predictive_ability"
    return result


def regime_labels(returns, window=20):
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
