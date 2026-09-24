"""Experiment database facade: DuckDB when installed, SQLite fallback otherwise."""

from __future__ import annotations
import sqlite3


class ExperimentDatabase:
    def __init__(self, path="runs/experiments.db"):
        self.path = path
        try:
            import duckdb

            self.db = duckdb.connect(path)
            self.backend = "duckdb"
        except ImportError:
            self.db = sqlite3.connect(path)
            self.backend = "sqlite"
        self.db.execute(
            "create table if not exists strategies (strategy_id varchar, test_sharpe double, test_drawdown double, robustness_score double, payload varchar)"
        )

    def ingest(self, record, robustness_score=0):
        self.db.execute(
            "insert into strategies values (?,?,?,?,?)",
            (
                record["strategy_id"],
                record.get("test_metrics", {}).get("sharpe", 0),
                record.get("test_metrics", {}).get("max_drawdown", 0),
                robustness_score,
                str(record),
            ),
        )
        try:
            self.db.commit()
        except AttributeError:
            pass

    def query(self, sql):
        result = self.db.execute(sql)
        return result.fetchall()

    def close(self):
        self.db.close()
