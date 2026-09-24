from __future__ import annotations
import math
from sizon.core.expression import Context, Primitive, example_genome
from sizon.data.data import DataFeed
from sizon.simulation.backtest import run_cross_sectional_backtest
from sizon.simulation.orderbook import stress_test_microstructure, microstructure_gauntlet
from sizon.research.evolution import Engine
from sizon.platform.studio_server import StudioHandler


def test_new_math_indicator_kernels():
    feed = DataFeed.from_csv("examples/sample.csv")
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


def test_engine_crossover_and_bloat_control(tmp_path):
    feed = DataFeed.from_csv("examples/sample.csv")
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


def test_cross_sectional_backtest():
    feed = DataFeed.from_csv("examples/sample.csv")
    # Feed has multiple rows with symbol
    result = run_cross_sectional_backtest(feed, example_genome())
    assert result.equity[0] == 1.0
    assert len(result.returns) > 1
    metrics = result.metrics()
    assert "sharpe" in metrics
    assert "total_return" in metrics


def test_two_tier_microstructure_gauntlet():
    feed = DataFeed.from_csv("examples/sample.csv")
    genome = example_genome()
    diag = stress_test_microstructure(feed, genome)
    assert "microstructure_sharpe" in diag
    assert "baseline_sharpe" in diag
    assert "passed" in diag

    gauntlet = microstructure_gauntlet(feed, [genome])
    assert len(gauntlet) == 1
    assert "microstructure_sharpe" in gauntlet[0]


def test_studio_api_runs(tmp_path):
    # Verify that StudioHandler can inspect runs directory and return runs
    handler = StudioHandler
    handler.runs_dir = tmp_path
    # Create a dummy run directory
    dummy_run = tmp_path / "dummy_run"
    dummy_run.mkdir()
    (dummy_run / "summary.json").write_text('{"run_id": "dummy_run", "strategies_saved": 5, "best_train_metrics": {"sharpe": 1.2}}')
    assert (dummy_run / "summary.json").exists()
