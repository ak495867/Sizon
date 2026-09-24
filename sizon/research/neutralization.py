"""Factor neutralization and Gram-Schmidt signal orthogonalization."""

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


def neutralize_market(
    signal: list[float], market_returns: list[float]
) -> list[float]:
    """OLS residualization: strips out linear market exposure from raw alpha signal."""
    cx, cy = _clean_pairs(signal, market_returns)
    if len(cx) < 3:
        return list(signal)

    mx, my = _mean(cx), _mean(cy)
    var_m = sum((y - my) ** 2 for y in cy) / (len(cy) - 1)
    if var_m <= 1e-12:
        return list(signal)

    cov = sum((cx[i] - mx) * (cy[i] - my) for i in range(len(cx))) / (
        len(cx) - 1
    )
    beta = cov / var_m
    alpha = mx - beta * my

    neutralized = []
    for s, m in zip(signal, market_returns):
        if math.isnan(s) or math.isnan(m):
            neutralized.append(math.nan)
        else:
            resid = s - (alpha + beta * m)
            neutralized.append(resid)

    # Standardize output signal
    valid = [x for x in neutralized if not math.isnan(x)]
    if valid:
        m_res = _mean(valid)
        s_res = _std(valid, m_res) or 1.0
        return [
            (x - m_res) / s_res if not math.isnan(x) else math.nan
            for x in neutralized
        ]
    return neutralized


def orthogonalize_signal(
    candidate_signal: list[float], basis_signals: list[list[float]]
) -> list[float]:
    """Gram-Schmidt projection: extracts the component orthogonal to existing library signals."""
    current = list(candidate_signal)
    for basis in basis_signals:
        cx, cb = _clean_pairs(current, basis)
        if len(cx) < 3:
            continue
        dot_product = sum(cx[i] * cb[i] for i in range(len(cx)))
        norm_sq = sum(cb[i] ** 2 for i in range(len(cb)))
        if norm_sq <= 1e-12:
            continue
        proj_scalar = dot_product / norm_sq

        # Subtract projection
        for i in range(len(current)):
            if not (math.isnan(current[i]) or math.isnan(basis[i])):
                current[i] -= proj_scalar * basis[i]

    # Standardize residual signal
    valid = [x for x in current if not math.isnan(x)]
    if valid:
        m = _mean(valid)
        s = _std(valid, m) or 1.0
        return [(x - m) / s if not math.isnan(x) else math.nan for x in current]
    return current


def signal_novelty_score(
    candidate_signal: list[float], library_signals: list[list[float]]
) -> float:
    """Computes orthogonality score (1.0 = completely novel, 0.0 = exact duplicate)."""
    if not library_signals:
        return 1.0
    max_abs_corr = 0.0
    for basis in library_signals:
        corr = abs(_correlation(candidate_signal, basis))
        if corr > max_abs_corr:
            max_abs_corr = corr
    return max(0.0, 1.0 - max_abs_corr)
