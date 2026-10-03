"""Population comparison and robustness scoring."""

from __future__ import annotations


def robustness_score(record):
    """
    Calculate a robustness score (0-100) based on:
    - 50%: Survival rate under increased transaction costs.
    - 30%: Stability under parameter perturbation.
    - 20%: 5th percentile Monte Carlo return being above -1 (i.e. not ruined).
    """
    robust = record.get("robustness") or {}
    
    cost_data = robust.get("cost_sensitivity") or {}
    cost = cost_data.get("scenarios") or []
    
    param_data = robust.get("parameter_sensitivity") or {}
    param = param_data.get("scenarios") or []
    
    mc_data = robust.get("monte_carlo") or {}
    quantiles = mc_data.get("quantiles") or {}
    total_return_q = quantiles.get("total_return") or {}
    mc = total_return_q.get("p05", 0)

    cost_survival = sum(s.get("metrics", {}).get("sharpe", 0) > 0 for s in cost) / max(
        1, len(cost)
    )
    param_stability = sum(
        1 for s in param if s.get("metrics", {}).get("sharpe", 0) > 0
    ) / max(1, len(param))
    return max(
        0,
        min(
            100, 50 * cost_survival + 30 * param_stability + 20 * (1 if mc > -1 else 0)
        ),
    )


def compare(records):
    rows = {}
    for r in records:
        row = {
            "strategy_id": r.get("strategy_id"),
            "expression": r.get("expression"),
            "test_sharpe": r.get("test_metrics", {}).get("sharpe", 0),
            "test_drawdown": r.get("test_metrics", {}).get("max_drawdown", 0),
            "robustness_score": robustness_score(r),
        }
        if row["strategy_id"] not in rows or (
            row["robustness_score"],
            row["test_sharpe"],
        ) > (
            rows[row["strategy_id"]]["robustness_score"],
            rows[row["strategy_id"]]["test_sharpe"],
        ):
            rows[row["strategy_id"]] = row
    return sorted(
        rows.values(),
        key=lambda x: (x["robustness_score"], x["test_sharpe"]),
        reverse=True,
    )
