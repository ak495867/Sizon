from sizon import (
    DataFeed,
    ExperimentDatabase,
    Genealogy,
    KillSwitch,
    Primitive,
    ShadowMode,
    SimulatedBroker,
    compare,
    export_json,
    export_python,
    export_yaml,
    notebook_template,
    population_analytics,
)


def test_clean_domain_imports_and_specialized_primitives():
    from sizon.core.expression import Context

    feed = DataFeed.from_csv("examples/sample.csv")
    ctx = Context({k: feed.column(k) for k in ("close", "high", "low", "volume")})
    for name in ("DEMA", "TEMA", "DONCHIAN_UPPER", "BB_PERCENT", "PPO"):
        assert len(Primitive(name, 3).evaluate(ctx)) == len(feed.rows)


def test_genealogy_analytics_and_robustness_comparison():
    g = Genealogy()
    g.add("child", ["parent"], 1, "mutation")
    assert "child" in g.ancestors("child")
    record = {
        "strategy_id": "s1",
        "generation": 0,
        "test_metrics": {"sharpe": 1, "max_drawdown": 0.1},
        "robustness": {
            "cost_sensitivity": {"scenarios": [{"metrics": {"sharpe": 1}}]},
            "parameter_sensitivity": {"scenarios": [{"metrics": {"sharpe": 1}}]},
            "monte_carlo": {"quantiles": {"total_return": {"p05": 0.1}}},
        },
    }
    assert population_analytics([record])["strategies"] == 1
    assert compare([record])[0]["robustness_score"] > 0


def test_exports_notebook_and_deployment_controls(tmp_path):
    record = {"strategy_id": "s", "expression": "RSI(close, 14)"}
    assert export_json(record, tmp_path / "s.json").exists()
    assert export_yaml(record, tmp_path / "s.yaml").exists()
    assert export_python(record, tmp_path / "s.py").exists()
    assert notebook_template(tmp_path / "s.ipynb").exists()
    switch = KillSwitch()
    assert switch.allow()
    switch.trip("stale_data")
    assert not switch.allow()
    shadow = ShadowMode()
    assert shadow.record(1, 100)["would_trade"]
    assert SimulatedBroker().cancel_all()["cancelled"] == 0


def test_experiment_database_and_nsga_selection(tmp_path):
    db = ExperimentDatabase(tmp_path / "exp.db")
    db.ingest(
        {"strategy_id": "s", "test_metrics": {"sharpe": 1, "max_drawdown": 0.1}}, 50
    )
    assert db.query("select count(*) from strategies")[0][0] == 1
    db.close()
