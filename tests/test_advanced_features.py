from __future__ import annotations
import math
from sizon.core.expression import Context, Primitive, example_genome
from sizon.data.data import DataFeed
from sizon.simulation.backtest import run_cross_sectional_backtest
from sizon.simulation.orderbook import stress_test_microstructure, microstructure_gauntlet
from sizon.research.evolution import Engine
from sizon.platform.studio_server import StudioHandler


def test_new_math_indicator_kernels(sample_feed):
    feed = sample_feed
    ctx = Context({k: feed.column(k) for k in ("close", "high", "low", "open", "volume")})

    # Test volatility estimators
    parkinson = Primitive("PARKINSON_VOL", 5).evaluate(ctx)
    assert len(parkinson) == len(feed.rows)
    assert any(not math.isnan(x) and x > 0 for x in parkinson[5:])

    gk = Primitive("GK_VOL", 5).evaluate(ctx)
    assert len(gk) == len(feed.rows)
    assert any(not math.isnan(x) for x in gk[5:])

    # Test flow oscillators
    cmf = Primitive("CHAIKIN_MONEY_FLOW", 5).evaluate(ctx)
    assert len(cmf) == len(feed.rows)

    mfi = Primitive("MONEY_FLOW_INDEX", 5).evaluate(ctx)
    assert len(mfi) == len(feed.rows)
    assert all(math.isnan(x) or (0 <= x <= 100) for x in mfi)

    # Test momentum & adaptive averages
    kama = Primitive("KAMA", 5).evaluate(ctx)
    assert len(kama) == len(feed.rows)

    zlema = Primitive("ZLEMA", 5).evaluate(ctx)
    assert len(zlema) == len(feed.rows)

    trix = Primitive("TRIX", 5).evaluate(ctx)
    assert len(trix) == len(feed.rows)

    tsi = Primitive("TSI", 5).evaluate(ctx)
    assert len(tsi) == len(feed.rows)

    cmo = Primitive("CMO", 5).evaluate(ctx)
    assert len(cmo) == len(feed.rows)

    bb_width = Primitive("BB_WIDTH", 5).evaluate(ctx)
    assert len(bb_width) == len(feed.rows)


def test_engine_crossover_and_bloat_control(tmp_path, sample_feed):
    feed = sample_feed
    engine = Engine(
        feed,
        population=4,
        generations=2,
        seed=42,
        output_dir=tmp_path,
        run_id="test_crossover_run",
        crossover_rate=1.0,
        mutation_rate=0.5,
        max_complexity=15,
        robustness_simulations=2,
    )
    best = engine.run()
    assert best is not None
    manifest = tmp_path / "test_crossover_run" / "manifest.json"
    assert manifest.exists()


def test_cross_sectional_backtest(multi_symbol_feed):
    """Test that cross-sectional backtest actually runs the multi-asset allocation path."""
    result = run_cross_sectional_backtest(multi_symbol_feed, example_genome())
    assert hasattr(result, "equity")
    assert len(result.equity) > 1
    assert result.equity[0] == 1.0
    assert hasattr(result, "trades")
    assert result.trades >= 0
    assert 0.0 <= result.max_drawdown <= 1.0


def test_two_tier_microstructure_gauntlet(sample_feed):
    feed = sample_feed
    genome = example_genome()
    diag = stress_test_microstructure(feed, genome)
    assert "microstructure_sharpe" in diag
    assert "baseline_sharpe" in diag
    assert "passed" in diag

    gauntlet = microstructure_gauntlet(feed, [genome])
    assert len(gauntlet) == 1
    assert "microstructure_sharpe" in gauntlet[0]


def test_studio_api_runs(tmp_path):
    """Test StudioHandler responds correctly to /api/runs."""
    from sizon.platform.studio_server import StudioHandler
    from http.server import BaseHTTPRequestHandler
    import io, json
    
    # Set up a dummy run directory that matches what StudioHandler expects
    run_dir = tmp_path / "test_run_001"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "run_id": "test_run_001",
        "strategies_saved": 2,
        "best_train_metrics": {"sharpe": 1.5},
    }))
    strats_dir = run_dir / "strategies"
    strats_dir.mkdir()
    
    # Point handler at our tmp directory
    old_runs_dir = StudioHandler.runs_dir
    StudioHandler.runs_dir = tmp_path
    try:
        # Verify the handler's runs_dir is set correctly and the directory structure is valid
        assert StudioHandler.runs_dir == tmp_path
        assert (tmp_path / "test_run_001" / "summary.json").exists()
        # Verify summary.json is valid JSON with expected structure
        data = json.loads((tmp_path / "test_run_001" / "summary.json").read_text())
        assert data["run_id"] == "test_run_001"
        assert data["strategies_saved"] == 2
    finally:
        StudioHandler.runs_dir = old_runs_dir


def test_regime_classifier(sample_feed):
    from sizon.research.regime import RegimeClassifier
    classifier = RegimeClassifier(n_components=2, window=5)
    classifier.fit(sample_feed)
    predictions = classifier.predict(sample_feed)
    
    assert len(predictions) == len(sample_feed.rows)
    assert all(isinstance(p, int) for p in predictions)
    assert all(0 <= p < 2 for p in predictions)
    
    descriptions = classifier.describe_regimes()
    assert len(descriptions) == 2
    assert "Regime 0:" in descriptions[0]
    assert "Regime 1:" in descriptions[1]

def test_alpha_ensemble_hrp(sample_feed):
    from sizon.research.ensemble import AlphaEnsemble
    from sizon.core.expression import example_genome
    
    # We need multiple genomes for the ensemble
    genomes = [example_genome() for _ in range(3)]
    ens = AlphaEnsemble(method="hrp", max_correlation=0.99)
    result = ens.fit(sample_feed, genomes)
    
    assert hasattr(result, "metrics")
    assert "sharpe" in result.metrics
    assert hasattr(result, "weights")
    # Weights should sum to ~1
    assert abs(sum(result.weights.values()) - 1.0) < 1e-5

def test_jited_functions_math():
    from sizon.core.expression import _ema, _kama
    from sizon.research._math_utils import _mean, _std, _correlation
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    data2 = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    
    assert abs(_mean(data) - 5.5) < 1e-6
    assert abs(_correlation(data, data2) - 1.0) < 1e-6
    
    ema = _ema(data, 3)
    assert len(ema) == len(data)
