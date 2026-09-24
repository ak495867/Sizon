"""Alphalens-grade Factor Analytics: Rank IC, IC Information Ratio, Quantiles, and Decay."""

from __future__ import annotations
import math
from sizon.core.expression import Context, Node
from sizon.data.data import DataFeed


def _rank(values: list[float]) -> list[float]:
    """Compute fractional ranks for a sequence of numeric values."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and math.isclose(
            indexed[j + 1][1], indexed[i][1], rel_tol=1e-9, abs_tol=1e-12
        ):
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation between two equal-length arrays."""
    valid = [
        (a, b)
        for a, b in zip(x, y)
        if not (math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b))
    ]
    if len(valid) < 3:
        return 0.0
    vx = [item[0] for item in valid]
    vy = [item[1] for item in valid]
    rx = _rank(vx)
    ry = _rank(vy)
    n = len(valid)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    denom = n * (n**2 - 1)
    return max(-1.0, min(1.0, 1.0 - (6.0 * d_sq) / denom)) if denom > 0 else 0.0


def compute_rank_ic(
    signals: list[float],
    forward_returns: list[float],
    chunk_size: int = 20,
) -> dict:
    """Compute Information Coefficient (Rank IC) statistics."""
    valid_pairs = [
        (s, r)
        for s, r in zip(signals, forward_returns)
        if not (math.isnan(s) or math.isnan(r))
    ]
    if len(valid_pairs) < chunk_size:
        # Full period correlation if chunking is not feasible
        sig = [p[0] for p in valid_pairs]
        ret = [p[1] for p in valid_pairs]
        ic = spearman_rank_correlation(sig, ret)
        return {
            "mean_ic": ic,
            "std_ic": 0.0,
            "ic_ir": 0.0,
            "t_stat": 0.0,
            "positive_ic_rate": 1.0 if ic > 0 else 0.0,
            "samples": len(valid_pairs),
        }

    # Rolling window IC evaluation
    ics = []
    step = max(1, chunk_size // 2)
    for i in range(0, len(valid_pairs) - chunk_size + 1, step):
        chunk = valid_pairs[i : i + chunk_size]
        sig = [p[0] for p in chunk]
        ret = [p[1] for p in chunk]
        ics.append(spearman_rank_correlation(sig, ret))

    mean_ic = sum(ics) / max(1, len(ics))
    var_ic = sum((x - mean_ic) ** 2 for x in ics) / max(1, len(ics) - 1)
    std_ic = math.sqrt(var_ic)
    t_stat = (mean_ic / (std_ic / math.sqrt(len(ics)))) if std_ic > 0 else 0.0
    ic_ir = (mean_ic / std_ic * math.sqrt(252)) if std_ic > 0 else 0.0
    pos_rate = sum(1 for x in ics if x > 0) / max(1, len(ics))

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ic_ir": ic_ir,
        "t_stat": t_stat,
        "positive_ic_rate": pos_rate,
        "samples": len(valid_pairs),
    }


def quantile_spread(
    signals: list[float],
    forward_returns: list[float],
    quantiles: int = 5,
) -> dict:
    """Bin signals into quantiles and evaluate return spread and monotonicity."""
    valid_pairs = [
        (s, r)
        for s, r in zip(signals, forward_returns)
        if not (math.isnan(s) or math.isnan(r))
    ]
    if len(valid_pairs) < quantiles * 2:
        return {
            "quantiles": {},
            "spread": 0.0,
            "monotonicity_score": 0.0,
        }

    valid_pairs.sort(key=lambda x: x[0])
    bin_size = len(valid_pairs) // quantiles
    q_returns = {}
    for q in range(quantiles):
        start = q * bin_size
        end = len(valid_pairs) if q == quantiles - 1 else (q + 1) * bin_size
        chunk_rets = [p[1] for p in valid_pairs[start:end]]
        q_returns[f"Q{q+1}"] = sum(chunk_rets) / max(1, len(chunk_rets))

    q_vals = list(q_returns.values())
    monotonicity = spearman_rank_correlation(
        list(range(1, quantiles + 1)), q_vals
    )
    spread = q_vals[-1] - q_vals[0]

    return {
        "quantiles": q_returns,
        "spread": spread,
        "monotonicity_score": monotonicity,
    }


def factor_autocorrelation(
    signals: list[float], max_lags: int = 10
) -> list[float]:
    """Compute factor rank persistence across consecutive lags."""
    valid = [s for s in signals if not math.isnan(s)]
    autocorrs = [1.0]
    for lag in range(1, max_lags + 1):
        if len(valid) <= lag:
            break
        s1 = valid[:-lag]
        s2 = valid[lag:]
        autocorrs.append(spearman_rank_correlation(s1, s2))
    return autocorrs


class FactorAnalytics:
    """Comprehensive factor evaluation suite for alpha signals."""

    @staticmethod
    def analyze(
        feed: DataFeed,
        genome: Node,
        horizons: tuple[int, ...] = (1, 5, 10),
        quantiles: int = 5,
    ) -> dict:
        close = feed.column("close")
        ctx = Context(
            {k: feed.column(k) for k in ("close", "high", "low", "open", "volume")}
        )
        signal = genome.evaluate(ctx)

        results = {
            "strategy_expression": repr(genome),
            "horizons": {},
            "autocorrelation": factor_autocorrelation(signal, max_lags=5),
        }

        for h in horizons:
            # Forward return: R_{t+h} = (Close_{t+h} / Close_t) - 1
            fwd = [math.nan] * len(close)
            for i in range(len(close) - h):
                if close[i] > 0:
                    fwd[i] = close[i + h] / close[i] - 1.0

            ic_stats = compute_rank_ic(signal, fwd)
            q_stats = quantile_spread(signal, fwd, quantiles=quantiles)
            results["horizons"][f"h{h}"] = {
                "rank_ic": ic_stats,
                "quantile_spread": q_stats,
            }

        return results
