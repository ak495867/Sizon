"""Shared pytest fixtures for the Sizon test suite."""
from __future__ import annotations
from pathlib import Path
import pytest
import sizon as sz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "examples" / "sample.csv"


@pytest.fixture(scope="session")
def sample_feed():
    """Single-symbol BTC feed loaded from examples/sample.csv."""
    return sz.DataFeed.from_csv(str(SAMPLE_CSV))


@pytest.fixture(scope="session")
def multi_symbol_feed():
    """Two-symbol feed for testing cross-sectional backtests.
    
    Creates ETH rows by scaling BTC prices by 0.06 so we have genuinely
    two different symbols in the same feed.
    """
    btc_feed = sz.DataFeed.from_csv(str(SAMPLE_CSV))
    eth_rows = []
    for row in btc_feed.rows:
        eth_row = dict(row)
        eth_row["symbol"] = "ETH"
        eth_row["open"] = float(row["open"]) * 0.06
        eth_row["high"] = float(row["high"]) * 0.06
        eth_row["low"] = float(row["low"]) * 0.06
        eth_row["close"] = float(row["close"]) * 0.06
        eth_rows.append(eth_row)
    all_rows = btc_feed.rows + eth_rows
    return sz.DataFeed(all_rows, source="multi_symbol_fixture")


@pytest.fixture
def example_genome():
    """A simple example genome for testing."""
    return sz.example_genome()
