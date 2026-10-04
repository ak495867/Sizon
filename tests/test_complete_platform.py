import json
from sizon import (
    ALL_PRIMITIVES,
    DataFeed,
    Engine,
    ExperimentQuery,
    Order,
    PaperBroker,
    PaperOrder,
    PortfolioConfig,
    Primitive,
    ValueType,
    bootstrap_mean,
    build_report,
    deflated_sharpe_ratio,
    example_genome,
    execute_order,
    load_checkpoint,
    multiple_testing_adjust,
    pareto_front,
    position,
    save_checkpoint,
    signal,
    spa_test,
    validate_expression,
    white_reality_check,
)


def test_more_than_100_primitives_are_callable(sample_feed):
    feed = sample_feed
    ctx = __import__("sizon.core.expression", fromlist=["Context"]).Context(
        {k: feed.column(k) for k in ("close", "high", "low", "volume")}
    )
    assert len(ALL_PRIMITIVES) >= 100
    for name in ALL_PRIMITIVES:
        assert len(Primitive(name, 3).evaluate(ctx)) == len(feed.rows)


def test_typed_research_and_portfolio_safety():
    g = example_genome()
    assert validate_expression(g)
    typed = signal(g)
    assert position(typed).value_type == ValueType.POSITION
    cfg = PortfolioConfig(min_order_size=2, tick_size=0.1)
    order = execute_order(Order("BTC", "buy", 1), 100, 10, cfg)
    assert order.status == "rejected"
    order = execute_order(Order("BTC", "buy", 2), 100, 1, cfg)
    assert order.status == "partial"


def test_statistics_search_and_paper_broker(tmp_path):
    assert bootstrap_mean([1, 2, 3], 20)["simulations"] == 20
    assert deflated_sharpe_ratio(2, 100, 100)["deflated_sharpe"] < 2
    assert len(multiple_testing_adjust([0.01, 0.02])) == 2
    assert white_reality_check([[1, -1], [0.1, 0.1]], 10)["simulations"] == 10
    assert spa_test([[1, -1]], 5)["method"]
    assert len(pareto_front([])) == 0
    path = tmp_path / "checkpoint.json"
    save_checkpoint(path, [], 2, 7)
    assert load_checkpoint(path)["generation"] == 2
    broker = PaperBroker()
    import pytest
    with pytest.raises(PermissionError):
        broker.submit(PaperOrder("BTC", "buy", 1))
    broker.enable()
    assert broker.submit(PaperOrder("BTC", "buy", 1))["status"] == "paper_submitted"


def test_report_and_sqlite_query(tmp_path, sample_feed):
    feed = sample_feed
    Engine(
        feed,
        population=1,
        generations=1,
        seed=1,
        output_dir=tmp_path,
        run_id="report-run",
        robustness_simulations=2,
    ).run()
    run = tmp_path / "report-run"
    assert build_report(run).exists()
    q = ExperimentQuery(tmp_path / "experiments.sqlite")
    record = json.loads(next((run / "strategies").glob("*.json")).read_text())
    q.ingest(record)
    assert q.top(1)
    q.close()


def test_central_clearing_degrades_identical_strategies(sample_feed):
    from sizon.simulation.clearing import CentralClearingBook
    from sizon.simulation.execution import ExecutionModel
    from sizon import example_genome

    exec_model = ExecutionModel(impact_bps_per_turnover=50.0, commission_bps=1.0, spread_bps=1.0, slippage_bps=1.0)
    
    g = example_genome()
    book = CentralClearingBook()
    
    res1 = book.simulate_concurrent(sample_feed, [g], exec_model)
    single_strat_return = res1["strat_0"].total_return
    
    res3 = book.simulate_concurrent(sample_feed, [g, g, g], exec_model)
    multi_strat_return = res3["strat_0"].total_return
    
    assert multi_strat_return < single_strat_return, "Aggregate impact should degrade returns for identical strategies"
