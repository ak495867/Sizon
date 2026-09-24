import json
from sizon import DataFeed,Engine,ExecutionModel,example_genome,genome_from_dict,genome_to_dict,run_backtest,walk_forward,purged_cross_validation,robustness_suite

def test_feed_and_quality_report():
    feed=DataFeed.from_csv("examples/sample.csv"); report=feed.quality_report(); assert feed.validate()["rows"]==20; assert report.ok

def test_expression_round_trip_and_execution_costs():
    feed=DataFeed.from_csv("examples/sample.csv"); genome=example_genome(); assert genome_to_dict(genome)==genome_to_dict(genome_from_dict(genome_to_dict(genome)))
    cheap=run_backtest(feed,genome,ExecutionModel(commission_bps=0,spread_bps=0,slippage_bps=0)); expensive=run_backtest(feed,genome,ExecutionModel(commission_bps=50,spread_bps=50,slippage_bps=50)); assert expensive.costs_paid>=cheap.costs_paid

def test_walk_forward_and_purged_cv_have_boundaries():
    feed=DataFeed.from_csv("examples/sample.csv"); genome=example_genome(); wf=walk_forward(feed,genome,n_splits=2,test_size=.2); cv=purged_cross_validation(feed,genome,n_splits=3,purge_bars=4,embargo_bars=2)
    assert wf["method"]=="walk_forward" and cv["method"]=="purged_cross_validation"; assert cv["purge_bars"]==4; assert all(f["test_start"]>=0 for f in cv["fold_details"])

def test_robustness_suite_has_all_sensitivity_families():
    feed=DataFeed.from_csv("examples/sample.csv"); report=robustness_suite(feed,example_genome(),simulations=10,seed=3)
    assert set(report)=={"cost_sensitivity","parameter_sensitivity","monte_carlo"}; assert len(report["cost_sensitivity"]["scenarios"])==5; assert len(report["parameter_sensitivity"]["scenarios"])==7

def test_engine_persists_every_candidate_and_all_research_tests(tmp_path):
    feed=DataFeed.from_csv("examples/sample.csv"); Engine(feed,population=2,generations=2,seed=11,output_dir=tmp_path,run_id="test-run",robustness_simulations=5).run(); run=tmp_path/"test-run"; records=list((run/"strategies").glob("*.json")); assert len(records)==4
    sample=json.loads(records[0].read_text()); assert sample["train_metrics"] and sample["test_metrics"]; assert sample["execution"]; assert "walk_forward" in sample["validation"]; assert "cost_sensitivity" in sample["robustness"]; assert len((run/"strategies.jsonl").read_text().splitlines())==4

def test_bad_ohlc_is_rejected(tmp_path):
    path=tmp_path/"bad.csv"; path.write_text("timestamp,symbol,open,high,low,close,volume\n2026-01-01,X,10,8,9,10,1\n")
    try: DataFeed.from_csv(path); assert False
    except ValueError as exc: assert "OHLC" in str(exc)
