"""Tests for the Sizon CLI (sizon.cli.main).

Uses subprocess to invoke the CLI as a real process, verifying that:
- Commands parse arguments correctly
- Commands produce valid JSON output
- Error paths fail gracefully (corrupt files, missing dirs)
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "examples" / "sample.csv"


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run the Sizon CLI and return the CompletedProcess result."""
    return subprocess.run(
        [sys.executable, "-m", "sizon.cli.main", *args],
        capture_output=True,
        text=True,
        cwd=str(cwd or PROJECT_ROOT),
    )


# ──────────────────────────────────────────────────────────────────────────────
# validate command
# ──────────────────────────────────────────────────────────────────────────────


def test_cli_validate_valid_csv():
    result = _run_cli("validate", str(SAMPLE_CSV))
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    # DataFeed.validate() should return a dict with an 'ok' key
    assert isinstance(data, dict)


def test_cli_validate_missing_file():
    result = _run_cli("validate", "nonexistent_file.csv")
    # Should fail with non-zero exit or print error; must not crash silently
    assert result.returncode != 0 or "error" in result.stdout.lower() or result.stderr


def test_cli_no_command_prints_help():
    result = subprocess.run(
        [sys.executable, "-m", "sizon.cli.main"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    # argparse exits with code 2 when required args are missing
    assert result.returncode in (0, 2)
    assert "sizon" in result.stdout + result.stderr


# ──────────────────────────────────────────────────────────────────────────────
# run command
# ──────────────────────────────────────────────────────────────────────────────


def test_cli_run_produces_run_output(tmp_path):
    result = _run_cli(
        "run",
        str(SAMPLE_CSV),
        "--output-dir", str(tmp_path),
        "--run-id", "cli-test-run",
        "--population", "2",
        "--generations", "1",
        "--seed", "42",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert data["run_id"] == "cli-test-run"
    assert "run_path" in data
    assert "best_train_metrics" in data
    # Verify the run directory was actually created
    run_dir = tmp_path / "cli-test-run"
    assert run_dir.exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.json").exists()


def test_cli_run_creates_strategy_files(tmp_path):
    result = _run_cli(
        "run",
        str(SAMPLE_CSV),
        "--output-dir", str(tmp_path),
        "--run-id", "cli-strat-test",
        "--population", "2",
        "--generations", "1",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    run_dir = tmp_path / "cli-strat-test"
    strategies = list((run_dir / "strategies").glob("*.json"))
    assert len(strategies) == 2  # pop=2, gen=1


# ──────────────────────────────────────────────────────────────────────────────
# ensemble command
# ──────────────────────────────────────────────────────────────────────────────


def test_cli_ensemble_produces_weights(tmp_path):
    # First, create a run with strategies to ensemble
    run_result = _run_cli(
        "run",
        str(SAMPLE_CSV),
        "--output-dir", str(tmp_path),
        "--run-id", "ens-test",
        "--population", "3",
        "--generations", "1",
    )
    assert run_result.returncode == 0, f"run stderr: {run_result.stderr}"

    result = _run_cli(
        "ensemble",
        str(tmp_path / "ens-test"),
        str(SAMPLE_CSV),
        "--method", "equal_weight",
        "--top-n", "2",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert "ensemble_method" in data
    assert data["ensemble_method"] == "equal_weight"
    assert "weights" in data


def test_cli_ensemble_missing_run_dir(tmp_path):
    result = _run_cli(
        "ensemble",
        str(tmp_path / "nonexistent_run"),
        str(SAMPLE_CSV),
    )
    assert result.returncode != 0


# ──────────────────────────────────────────────────────────────────────────────
# gauntlet command
# ──────────────────────────────────────────────────────────────────────────────


def test_cli_gauntlet_runs(tmp_path):
    # Create run first
    run_result = _run_cli(
        "run",
        str(SAMPLE_CSV),
        "--output-dir", str(tmp_path),
        "--run-id", "gauntlet-test",
        "--population", "2",
        "--generations", "1",
    )
    assert run_result.returncode == 0, f"run stderr: {run_result.stderr}"

    result = _run_cli(
        "gauntlet",
        str(tmp_path / "gauntlet-test"),
        str(SAMPLE_CSV),
        "--top-n", "2",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert "strategies_tested" in data
    assert "passed_gauntlet" in data
    assert "results" in data
    assert data["strategies_tested"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# factor command
# ──────────────────────────────────────────────────────────────────────────────


def test_cli_factor_runs(tmp_path):
    # Create run first
    run_result = _run_cli(
        "run",
        str(SAMPLE_CSV),
        "--output-dir", str(tmp_path),
        "--run-id", "factor-test",
        "--population", "2",
        "--generations", "1",
    )
    assert run_result.returncode == 0, f"run stderr: {run_result.stderr}"

    result = _run_cli(
        "factor",
        str(tmp_path / "factor-test"),
        str(SAMPLE_CSV),
        "--quantiles", "3",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert isinstance(data, dict)


# ──────────────────────────────────────────────────────────────────────────────
# _load_strategies helper
# ──────────────────────────────────────────────────────────────────────────────


def test_load_strategies_skips_corrupt_files(tmp_path):
    """_load_strategies should skip corrupt JSON and continue rather than crashing."""
    from sizon.cli.main import _load_strategies

    strat_dir = tmp_path / "strategies"
    strat_dir.mkdir()
    # Write one valid strategy
    valid = {
        "strategy_id": "s1",
        "generation": 0,
        "test_metrics": {"sharpe": 1.5},
        "genome": {"type": "Primitive", "name": "SMA", "period": 10},
    }
    (strat_dir / "s1.json").write_text(json.dumps(valid))
    # Write one corrupt file
    (strat_dir / "s2.json").write_text("not-valid-json{{{")

    records = _load_strategies(tmp_path)
    assert len(records) == 1
    assert records[0]["strategy_id"] == "s1"


def test_load_strategies_sorted_by_sharpe(tmp_path):
    """_load_strategies should return records sorted by test Sharpe descending."""
    from sizon.cli.main import _load_strategies

    strat_dir = tmp_path / "strategies"
    strat_dir.mkdir()
    for sid, sharpe in [("a", 0.5), ("b", 2.0), ("c", -0.3)]:
        rec = {"strategy_id": sid, "generation": 0, "test_metrics": {"sharpe": sharpe}}
        (strat_dir / f"{sid}.json").write_text(json.dumps(rec))

    records = _load_strategies(tmp_path)
    sharpes = [r["test_metrics"]["sharpe"] for r in records]
    assert sharpes == sorted(sharpes, reverse=True)


def test_load_strategies_top_n(tmp_path):
    """top_n parameter should limit the number of returned strategies."""
    from sizon.cli.main import _load_strategies

    strat_dir = tmp_path / "strategies"
    strat_dir.mkdir()
    for i in range(5):
        rec = {"strategy_id": f"s{i}", "generation": 0, "test_metrics": {"sharpe": float(i)}}
        (strat_dir / f"s{i}.json").write_text(json.dumps(rec))

    records = _load_strategies(tmp_path, top_n=2)
    assert len(records) == 2
    assert records[0]["test_metrics"]["sharpe"] == 4.0


def test_load_strategies_missing_dir_exits(tmp_path):
    """_load_strategies should SystemExit when run_dir has no strategies subfolder."""
    from sizon.cli.main import _load_strategies

    with pytest.raises(SystemExit):
        _load_strategies(tmp_path / "nonexistent")
