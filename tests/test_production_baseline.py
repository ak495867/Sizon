import pytest
from sizon import (
    BookLevel,
    DataFeed,
    DeploymentGate,
    Engine,
    ImpactCurve,
    NSGA2,
    OrderBookSnapshot,
    PointInTimeDataset,
    PortfolioLedger,
    SPECS,
    example_genome,
    execute_market,
    optimize_cross_sectional,
    simulate_limit_fill,
    validate_strict,
)


def test_orderbook_partial_fill_and_impact():
    book = OrderBookSnapshot(
        (BookLevel(99, 5),), (BookLevel(101, 2), BookLevel(102, 3))
    )
    fill = execute_market("buy", 4, book, ImpactCurve(), fee_bps=2)
    assert fill.status == "filled"
    assert fill.filled_quantity == 4
    limit = simulate_limit_fill("buy", 5, 101, book)
    assert limit.status == "partial"
    assert 0 < limit.filled_quantity <= 2


def test_point_in_time_and_cross_sectional_accounting():
    pit = PointInTimeDataset(
        [
            {
                "timestamp": "2026-01-01",
                "available_at": "2026-01-02",
                "symbol": "X",
                "close": 10,
            }
        ],
        as_of="2026-01-01",
    )
    assert pit.visible_rows() == []
    assert optimize_cross_sectional({"A": 2, "B": 1}, max_position=0.6)["A"] <= 0.6
    ledger = PortfolioLedger(1000)
    ledger.trade("2026-01-01", "A", 2, 10, 1)
    assert ledger.mark_to_market({"A": 12}) == 1003


def test_strict_types_nsga_and_safety():
    assert validate_strict(example_genome())["all_nodes_registered"]
    assert len(SPECS) >= 100
    assert NSGA2().pareto_front([]) == []
    gate = DeploymentGate()
    with pytest.raises(PermissionError):
        gate.authorize(1)


def test_engine_checkpoint_resume(tmp_path):
    feed = DataFeed.from_csv("examples/sample.csv")
    Engine(
        feed,
        population=1,
        generations=1,
        seed=2,
        output_dir=tmp_path,
        run_id="first",
        robustness_simulations=1,
    ).run()
    checkpoint = tmp_path / "first" / "checkpoint_g0.json"
    assert checkpoint.exists()
    resumed = Engine(
        feed,
        population=1,
        generations=2,
        seed=2,
        output_dir=tmp_path,
        run_id="resumed",
        robustness_simulations=1,
        resume_from=checkpoint,
    ).run()
    assert resumed is not None
