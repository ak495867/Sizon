"""Optional SQLite index for searching thousands of persisted strategies."""

from __future__ import annotations
import json, sqlite3
from pathlib import Path


class ExperimentQuery:
    def __init__(self, path="runs/sizon.sqlite"):
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.execute(
            "create table if not exists strategies (strategy_id text, generation int, expression text, train_sharpe real, test_sharpe real, max_drawdown real, payload text)"
        )
        self.db.commit()

    def ingest(self, record):
        self.db.execute(
            "insert into strategies values (?,?,?,?,?,?,?)",
            (
                record["strategy_id"],
                record["generation"],
                record["expression"],
                record["train_metrics"].get("sharpe", 0),
                record["test_metrics"].get("sharpe", 0),
                record["test_metrics"].get("max_drawdown", 0),
                json.dumps(record),
            ),
        )
        self.db.commit()

    def top(self, limit=20):
        return self.db.execute(
            "select strategy_id,expression,test_sharpe,max_drawdown from strategies order by test_sharpe desc limit ?",
            (limit,),
        ).fetchall()

    def close(self):
        self.db.close()
