import json
from sizon import *


def test_more_than_100_primitives_are_callable():
    feed = DataFeed.from_csv("examples/sample.csv")
    ctx = __import__("sizon.core.expression", fromlist=["Context"]).Context(
        {k: feed.column(k) for k in ("close", "high", "low", "volume")}
    )
    assert len(ALL_PRIMITIVES) >= 100
    for name in ALL_PRIMITIVES:
        assert len(Primitive(name, 3).evaluate(ctx)) == len(feed.rows)


def test_typed_research_and_portfolio_safety():
    g = example_genome()
    assert validate_expression(g)["lookahead_safe"]
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
    try:
        broker.submit(PaperOrder("BTC", "buy", 1))
        assert False
    except PermissionError:
        pass
    broker.enable()
    assert broker.submit(PaperOrder("BTC", "buy", 1))["status"] == "paper_submitted"


def test_report_and_sqlite_query(tmp_path):
    feed = DataFeed.from_csv("examples/sample.csv")
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
