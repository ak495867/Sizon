"""Alpha ensembling and multi-strategy portfolio optimization."""

from __future__ import annotations
from dataclasses import dataclass
import math
from sizon.core.expression import Node
from sizon.data.data import DataFeed
from sizon.simulation.backtest import BacktestResult, run_backtest
from sizon.simulation.execution import ExecutionModel, DEFAULT_EXECUTION


@dataclass
class EnsembleMember:
    strategy_id: str
    genome: Node
    weight: float
    standalone_sharpe: float


@dataclass
class EnsembleResult:
    equity: list[float]
    returns: list[float]
    weights: dict[str, float]
    metrics: dict[str, float]
    diversification_ratio: float
    members: list[EnsembleMember]


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: list[float], mean_val: float | None = None) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values) if mean_val is None else mean_val
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(max(0.0, var))


def _correlation(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    mx, my = _mean(x[:n]), _mean(y[:n])
    sx, sy = _std(x[:n], mx), _std(y[:n], my)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)
    return max(-1.0, min(1.0, cov / (sx * sy)))


def _covariance_matrix(matrix: list[list[float]]) -> list[list[float]]:
    k = len(matrix)
    cov = [[0.0] * k for _ in range(k)]
    means = [_mean(row) for row in matrix]
    n = len(matrix[0]) if k > 0 else 0
    if n < 2:
        return cov
    for i in range(k):
        for j in range(i, k):
            c = (
                sum(
                    (matrix[i][t] - means[i]) * (matrix[j][t] - means[j])
                    for t in range(n)
                )
                / (n - 1)
            )
            cov[i][j] = c
            cov[j][i] = c
    return cov


def _invert_matrix_ridge(
    cov: list[list[float]], ridge: float = 1e-4
) -> list[list[float]]:
    """Invert symmetric matrix with ridge shrinkage via Gaussian elimination."""
    n = len(cov)
    # Augmented matrix [A + ridge*I | I]
    aug = []
    for i in range(n):
        row = [cov[i][j] + (ridge if i == j else 0.0) for j in range(n)]
        row += [1.0 if i == j else 0.0 for j in range(n)]
        aug.append(row)

    # Forward elimination with partial pivoting
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            pivot = 1e-12
        for j in range(2 * n):
            aug[col][j] /= pivot
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

    inv = [row[n:] for row in aug]
    return inv


def _risk_parity_weights(
    cov: list[list[float]], max_iter: int = 100
) -> list[float]:
    """Iterative Equal Risk Contribution (Risk Parity) solver."""
    n = len(cov)
    if n == 0:
        return []
    # Initial weights proportional to inverse volatility
    vols = [math.sqrt(max(1e-12, cov[i][i])) for i in range(n)]
    inv_vols = [1.0 / v for v in vols]
    s = sum(inv_vols) or 1.0
    w = [v / s for v in inv_vols]

    for _ in range(max_iter):
        # Marginal risk contribution: (cov * w)_i
        mrc = [sum(cov[i][j] * w[j] for j in range(n)) for i in range(n)]
        # Target update: w_i proportional to 1 / mrc_i
        w_new = [w[i] / max(1e-12, mrc[i]) for i in range(n)]
        s_new = sum(w_new) or 1.0
        w = [x / s_new for x in w_new]
    return w


def _min_variance_weights(
    cov: list[list[float]], ridge: float = 1e-4
) -> list[float]:
    """Minimum variance weights via regularized inverse covariance."""
    n = len(cov)
    if n == 0:
        return []
    inv = _invert_matrix_ridge(cov, ridge)
    raw = [sum(inv[i]) for i in range(n)]
    # Project to long-only simplex
    non_neg = [max(0.0, x) for x in raw]
    s = sum(non_neg)
    if s <= 1e-12:
        return [1.0 / n] * n
    return [x / s for x in non_neg]


def prune_correlated_strategies(
    candidates_with_returns: list[tuple[any, list[float]]],
    max_correlation: float = 0.65,
) -> list[int]:
    """Greedy correlation filter: retains strategies whose |corr| <= max_correlation."""
    accepted_indices = []
    for idx, (_, rets_a) in enumerate(candidates_with_returns):
        redundant = False
        for acc_idx in accepted_indices:
            _, rets_b = candidates_with_returns[acc_idx]
            corr = abs(_correlation(rets_a, rets_b))
            if corr > max_correlation:
                redundant = True
                break
        if not redundant:
            accepted_indices.append(idx)
    return accepted_indices


class AlphaEnsemble:
    """Ensemble combiner for multi-strategy alpha portfolios."""

    def __init__(
        self,
        method: str = "risk_parity",
        max_correlation: float = 0.65,
        ridge: float = 1e-4,
    ):
        self.method = method
        self.max_correlation = max_correlation
        self.ridge = ridge

    def fit(
        self,
        feed: DataFeed,
        genomes: list[Node],
        execution: ExecutionModel | None = None,
    ) -> EnsembleResult:
        execution = execution or DEFAULT_EXECUTION
        if not genomes:
            raise ValueError("No genomes provided to AlphaEnsemble")

        # 1. Backtest each candidate
        evaluated = []
        for i, g in enumerate(genomes):
            res: BacktestResult = run_backtest(feed, g, execution)
            strat_id = f"alpha_{i:03d}"
            evaluated.append((strat_id, g, res))

        # Rank candidates by standalone Sharpe descending
        evaluated.sort(key=lambda item: item[2].sharpe, reverse=True)

        # 2. Prune correlated strategies
        cand_rets = [(item, item[2].returns[1:]) for item in evaluated]
        keep_indices = prune_correlated_strategies(
            cand_rets, max_correlation=self.max_correlation
        )
        selected = [evaluated[idx] for idx in keep_indices]

        k = len(selected)
        returns_matrix = [item[2].returns[1:] for item in selected]
        cov = _covariance_matrix(returns_matrix)

        # 3. Compute optimal weights
        if self.method == "risk_parity":
            weights_list = _risk_parity_weights(cov)
        elif self.method == "min_variance":
            weights_list = _min_variance_weights(cov, ridge=self.ridge)
        elif self.method == "inverse_variance":
            inv_vars = [1.0 / max(1e-12, cov[i][i]) for i in range(k)]
            s = sum(inv_vars) or 1.0
            weights_list = [v / s for v in inv_vars]
        else:  # equal_weight
            weights_list = [1.0 / k] * k

        # 4. Construct ensemble portfolio returns
        n_bars = len(returns_matrix[0]) if k > 0 else 0
        ensemble_returns = [0.0]
        for t in range(n_bars):
            r_t = sum(weights_list[i] * returns_matrix[i][t] for i in range(k))
            ensemble_returns.append(r_t)

        equity = [1.0]
        for r in ensemble_returns[1:]:
            equity.append(equity[-1] * (1.0 + r))

        # 5. Compute combined metrics
        r_slice = ensemble_returns[1:]
        m_r = _mean(r_slice)
        s_r = _std(r_slice, m_r)
        ens_sharpe = (math.sqrt(252) * m_r / s_r) if s_r > 0 else 0.0

        peak = equity[0]
        worst = 0.0
        for val in equity:
            peak = max(peak, val)
            worst = min(worst, val / peak - 1)
        ens_dd = abs(worst)

        # Diversification Ratio: weighted average volatility / portfolio volatility
        standalone_vols = [math.sqrt(max(1e-12, cov[i][i])) for i in range(k)]
        weighted_vol = sum(weights_list[i] * standalone_vols[i] for i in range(k))
        port_vol = s_r
        div_ratio = (weighted_vol / port_vol) if port_vol > 0 else 1.0

        members = [
            EnsembleMember(
                strategy_id=item[0],
                genome=item[1],
                weight=weights_list[i],
                standalone_sharpe=item[2].sharpe,
            )
            for i, item in enumerate(selected)
        ]

        metrics = {
            "sharpe": ens_sharpe,
            "max_drawdown": ens_dd,
            "total_return": equity[-1] - 1.0,
            "selected_alphas": len(selected),
            "original_alphas": len(genomes),
            "diversification_ratio": div_ratio,
        }

        weights_dict = {
            member.strategy_id: member.weight for member in members
        }
        return EnsembleResult(
            equity,
            ensemble_returns,
            weights_dict,
            metrics,
            div_ratio,
            members,
        )
