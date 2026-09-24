from __future__ import annotations
import math
from sizon.core.expression import Binary, Primitive, example_genome
from sizon.data.data import DataFeed
from sizon.platform.deployment import KillSwitch
from sizon.platform.security import AuditLog
from sizon.research.ensemble import AlphaEnsemble, prune_correlated_strategies
from sizon.research.factor_analytics import (
    FactorAnalytics,
    compute_rank_ic,
    factor_autocorrelation,
    quantile_spread,
    spearman_rank_correlation,
)
from sizon.research.neutralization import (
    neutralize_market,
    orthogonalize_signal,
    signal_novelty_score,
)
from sizon.simulation.live import LiveBar, PaperTradingRunner


def test_spearman_and_rank_ic():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    # Perfect positive rank correlation
    assert math.isclose(spearman_rank_correlation(x, y), 1.0, abs_tol=1e-6)

    # Perfect inverse rank correlation
    inv_y = [50.0, 40.0, 30.0, 20.0, 10.0]
    assert math.isclose(spearman_rank_correlation(x, inv_y), -1.0, abs_tol=1e-6)

    # Test Rank IC computation
    sig = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    fwd = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    stats = compute_rank_ic(sig, fwd, chunk_size=5)
    assert stats["mean_ic"] > 0.9
    assert stats["positive_ic_rate"] == 1.0


def test_quantile_spread_and_decay():
    sig = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    fwd = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    q_stats = quantile_spread(sig, fwd, quantiles=4)
    assert q_stats["spread"] > 0
    assert math.isclose(q_stats["monotonicity_score"], 1.0, abs_tol=1e-6)

    decay = factor_autocorrelation(sig, max_lags=3)
    assert len(decay) == 4
    assert decay[0] == 1.0


def test_factor_analytics_full_pipeline():
    feed = DataFeed.from_csv("examples/sample.csv")
    genome = example_genome()
    report = FactorAnalytics.analyze(feed, genome, horizons=(1, 3), quantiles=3)
    assert "horizons" in report
    assert "h1" in report["horizons"]
    assert "h3" in report["horizons"]
    assert "rank_ic" in report["horizons"]["h1"]


def test_market_neutralization_and_orthogonalization():
    market = [0.01, -0.02, 0.03, -0.01, 0.02, 0.04, -0.03, 0.01, 0.02, -0.01]
    # Pure market-tracking alpha
    signal = [2.0 * m + 0.001 for m in market]
    neutral = neutralize_market(signal, market)
    assert len(neutral) == len(signal)
    # Correlation with market should be virtually 0
    valid_pairs = [(n, m) for n, m in zip(neutral, market) if not math.isnan(n)]
    vn = [p[0] for p in valid_pairs]
    vm = [p[1] for p in valid_pairs]
    cov = sum(
        (vn[i] - sum(vn) / len(vn)) * (vm[i] - sum(vm) / len(vm))
        for i in range(len(vn))
    )
    assert abs(cov) < 1e-4

    # Gram-Schmidt Orthogonalization
    b1 = [1.0, 0.0, 0.0, 1.0]
    cand = [1.0, 2.0, 0.0, 1.0]
    ortho = orthogonalize_signal(cand, [b1])
    assert len(ortho) == len(cand)

    # Signal Novelty Score
    nov = signal_novelty_score([1.0, 2.0, 3.0], [[1.0, 2.0, 3.0]])
    assert math.isclose(nov, 0.0, abs_tol=1e-5)
    nov_diff = signal_novelty_score([1.0, -1.0, 1.0], [[1.0, 2.0, 3.0]])
    assert nov_diff > 0.0


def test_alpha_ensemble_portfolio():
    feed = DataFeed.from_csv("examples/sample.csv")
    g1 = Primitive("RSI", 14)
    g2 = Primitive("SMA", 5)
    g3 = Binary("-", Primitive("EMA", 3), Primitive("SMA", 10))

    # Test correlation pruning
    mock_rets = [
        ("s1", [0.01, 0.02, -0.01]),
        ("s2", [0.01, 0.02, -0.01]),  # duplicate
        ("s3", [-0.02, 0.01, 0.03]),
    ]
    kept = prune_correlated_strategies(mock_rets, max_correlation=0.9)
    assert len(kept) == 2  # duplicate s2 should be pruned

    # Test Risk Parity Ensemble
    ens = AlphaEnsemble(method="risk_parity")
    res = ens.fit(feed, [g1, g2, g3])
    assert len(res.members) >= 1
    assert "sharpe" in res.metrics
    assert res.diversification_ratio >= 1.0
    assert len(res.equity) == len(feed.rows)


def test_live_paper_trading_daemon():
    genome = Primitive("SMA", 3)
    audit = AuditLog()
    switch = KillSwitch()
    runner = PaperTradingRunner(
        genome=genome, audit_log=audit, kill_switch=switch, buffer_size=10
    )

    feed = DataFeed.from_csv("examples/sample.csv")
    step_results = []
    for r in feed.rows:
        bar = LiveBar(
            timestamp=r["timestamp"],
            symbol=r["symbol"],
            open=r["open"],
            high=r["high"],
            low=r["low"],
            close=r["close"],
            volume=r["volume"],
        )
        res = runner.step(bar)
        step_results.append(res)

    assert any(s["status"] in {"filled", "partial", "no_change"} for s in step_results)
    snap = runner.snapshot()
    assert snap["audit_chain_verified"]
    assert not snap["kill_switch_active"]

    # Test Kill Switch Tripping
    switch.trip("latency_spike")
    blocked = runner.step(
        LiveBar("2026-01-01", "BTC", 100, 105, 95, 102, 1000)
    )
    assert blocked["status"] == "rejected"
    assert "kill switch" in blocked["reason"]
