import json
from sizon import (
    DataFeed,
    Engine,
    ExecutionModel,
    example_genome,
    genome_from_dict,
    genome_to_dict,
    run_backtest,
    walk_forward,
    purged_cross_validation,
    robustness_suite,
)


def test_feed_and_quality_report(sample_feed):
    feed = sample_feed
    report = feed.quality_report()
    assert feed.validate()["rows"] == 20
    assert report.ok


def test_expression_round_trip_and_execution_costs(sample_feed):
    feed = sample_feed
    genome = example_genome()
    assert genome_to_dict(genome) == genome_to_dict(
        genome_from_dict(genome_to_dict(genome))
    )
    cheap = run_backtest(
        feed, genome, ExecutionModel(commission_bps=0, spread_bps=0, slippage_bps=0)
    )
    expensive = run_backtest(
        feed, genome, ExecutionModel(commission_bps=50, spread_bps=50, slippage_bps=50)
    )
    assert expensive.costs_paid >= cheap.costs_paid


def test_walk_forward_and_purged_cv_have_boundaries(sample_feed):
    feed = sample_feed
    genome = example_genome()
    wf = walk_forward(feed, genome, n_splits=2, test_size=0.2)
    cv = purged_cross_validation(feed, genome, n_splits=3, purge_bars=4, embargo_bars=2)
    assert wf["method"] == "walk_forward" and cv["method"] == "purged_cross_validation"
    assert cv["purge_bars"] == 4
    assert all(f["test_start"] >= 0 for f in cv["fold_details"])


def test_robustness_suite_has_all_sensitivity_families(sample_feed):
    feed = sample_feed
    report = robustness_suite(feed, example_genome(), simulations=10, seed=3)
    assert set(report) == {"cost_sensitivity", "parameter_sensitivity", "monte_carlo"}
    assert len(report["cost_sensitivity"]["scenarios"]) == 5
    assert len(report["parameter_sensitivity"]["scenarios"]) == 7


def test_engine_persists_every_candidate_and_all_research_tests(tmp_path, sample_feed):
    feed = sample_feed
    Engine(
        feed,
        population=2,
        generations=2,
        seed=11,
        output_dir=tmp_path,
        run_id="test-run",
        robustness_simulations=5,
    ).run()
    run = tmp_path / "test-run"
    records = list((run / "strategies").glob("*.json"))
    assert len(records) == 4
    sample = json.loads(records[0].read_text())
    assert sample["train_metrics"] and sample["test_metrics"]
    assert sample["execution"]
    assert "walk_forward" in sample["validation"]
    assert "cost_sensitivity" in sample["robustness"]
    assert len((run / "strategies.jsonl").read_text().splitlines()) == 4


def test_bad_ohlc_is_rejected(tmp_path):
    import pytest
    path = tmp_path / "bad.csv"
    path.write_text(
        "timestamp,symbol,open,high,low,close,volume\n2026-01-01,X,10,8,9,10,1\n"
    )
    with pytest.raises(ValueError, match="OHLC"):
        DataFeed.from_csv(path)


def test_alt_data_loaded_and_evaluated(tmp_path):
    from sizon.core.expression import Primitive, Binary
    path = tmp_path / "alt.csv"
    path.write_text(
        "timestamp,symbol,open,high,low,close,volume,funding_rate,imbalance,sentiment\n"
        "2026-01-01,X,10,12,9,11,100,0.01,0.5,0.9\n"
        "2026-01-02,X,11,13,10,12,200,0.02,-0.1,0.8\n"
        "2026-01-03,X,12,14,11,13,300,0.03,0.0,0.7\n"
    )
    feed = DataFeed.from_csv(path)
    ctx = DataFeed.to_context(feed)
    assert ctx.columns["funding_rate"] == [0.01, 0.02, 0.03]
    
    funding_node = Primitive("FUNDING_RATE")
    imbalance_node = Primitive("ORDERBOOK_IMBALANCE")
    assert funding_node.evaluate(ctx) == [0.01, 0.02, 0.03]
    assert imbalance_node.evaluate(ctx) == [0.5, -0.1, 0.0]
