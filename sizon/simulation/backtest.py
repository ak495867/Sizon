"""Leakage-aware bar backtesting with explicit execution assumptions."""

from __future__ import annotations
from dataclasses import dataclass
import math
from sizon.data.data import DataFeed
from sizon.simulation.execution import ExecutionModel, DEFAULT_EXECUTION
from sizon.core.expression import Context, Node


@dataclass
class BacktestResult:
    equity: list[float]
    returns: list[float]
    positions: list[int]
    trades: int = 0
    costs_paid: float = 0.0

    @property
    def sharpe(self):
        r = self.returns[1:]
        if len(r) < 2:
            return 0.0
        mean = sum(r) / len(r)
        var = sum((x - mean) ** 2 for x in r) / (len(r) - 1)
        return math.sqrt(252) * mean / math.sqrt(var) if var else 0.0

    @property
    def sortino(self):
        r = self.returns[1:]
        down = [min(0, x) for x in r]
        denom = math.sqrt(sum(x * x for x in down) / max(1, len(down)))
        return math.sqrt(252) * sum(r) / max(1, len(r)) / denom if denom else 0.0

    @property
    def max_drawdown(self):
        peak = self.equity[0]
        worst = 0.0
        for value in self.equity:
            peak = max(peak, value)
            worst = min(worst, value / peak - 1)
        return abs(worst)

    @property
    def turnover(self):
        return self.trades / max(1, len(self.positions) - 1)

    @property
    def total_return(self):
        return self.equity[-1] - 1

    def metrics(self):
        return {
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "trades": self.trades,
            "costs_paid": self.costs_paid,
        }


def run_backtest(
    feed: DataFeed,
    genome: Node,
    execution: ExecutionModel | None = None,
    threshold: float = 0.0,
) -> BacktestResult:
    execution = execution or DEFAULT_EXECUTION
    close = feed.column("close")
    signal = genome.evaluate(
        Context({k: feed.column(k) for k in ("close", "high", "low", "volume")})
    )
    raw = [0 if math.isnan(v) else (1 if v > threshold else -1) for v in signal]
    delay = max(1, execution.delay_bars)
    positions = [0] * len(raw)
    returns = [0.0]
    trades = 0
    costs = 0.0
    for i in range(1, len(close)):
        positions[i] = raw[i - delay] if i >= delay else 0
        turnover = abs(positions[i] - positions[i - 1])
        changed = turnover > 0
        trades += int(changed)
        gross = positions[i - 1] * (close[i] / close[i - 1] - 1)
        cost = execution.cost_rate(turnover, positions[i])
        costs += cost
        returns.append(gross - cost)
    equity = [1.0]
    for value in returns[1:]:
        equity.append(equity[-1] * (1 + value))
    return BacktestResult(equity, returns, positions, trades, costs)


def fitness(result: BacktestResult, complexity: int):
    return {**result.metrics(), "complexity": float(complexity)}


def run_cross_sectional_backtest(
    feeds: dict[str, DataFeed] | list[DataFeed] | DataFeed,
    genome: Node,
    execution: ExecutionModel | None = None,
    max_gross: float = 1.0,
    max_position: float = 0.2,
) -> BacktestResult:
    execution = execution or DEFAULT_EXECUTION
    if isinstance(feeds, DataFeed):
        symbols = sorted({r["symbol"] for r in feeds.rows})
        if len(symbols) <= 1:
            return run_backtest(feeds, genome, execution)
        symbol_rows: dict[str, list[dict]] = {s: [] for s in symbols}
        for r in feeds.rows:
            symbol_rows[r["symbol"]].append(r)
        feed_dict = {
            s: DataFeed(rows, feeds.source, feeds.timezone)
            for s, rows in symbol_rows.items()
        }
    elif isinstance(feeds, (list, tuple)):
        feed_dict = {f.rows[0]["symbol"]: f for f in feeds}
    else:
        feed_dict = feeds

    symbols = sorted(feed_dict.keys())
    signals = {}
    closes = {}
    for s in symbols:
        f = feed_dict[s]
        closes[s] = f.column("close")
        sig = genome.evaluate(
            Context({k: f.column(k) for k in ("close", "high", "low", "volume")})
        )
        signals[s] = sig

    n_bars = min(len(closes[s]) for s in symbols)
    delay = max(1, execution.delay_bars)
    returns = [0.0]
    total_trades = 0
    total_costs = 0.0
    prev_weights = {s: 0.0 for s in symbols}

    for i in range(1, n_bars):
        if i >= delay:
            raw_scores = {
                s: (0.0 if math.isnan(signals[s][i - delay]) else signals[s][i - delay])
                for s in symbols
            }
            mean_score = sum(raw_scores.values()) / max(1, len(symbols))
            demeaned = {s: raw_scores[s] - mean_score for s in symbols}
            pos_sum = sum(max(0.0, v) for v in demeaned.values()) or 1.0
            neg_sum = sum(abs(min(0.0, v)) for v in demeaned.values()) or 1.0
            target_weights = {}
            for s in symbols:
                val = demeaned[s]
                if val > 0:
                    w = min(max_position, (val / pos_sum) * (0.5 * max_gross))
                elif val < 0:
                    w = max(-max_position, (val / neg_sum) * (0.5 * max_gross))
                else:
                    w = 0.0
                target_weights[s] = w
        else:
            target_weights = {s: 0.0 for s in symbols}

        bar_gross = 0.0
        bar_cost = 0.0
        for s in symbols:
            turnover = abs(target_weights[s] - prev_weights[s])
            if turnover > 1e-6:
                total_trades += 1
            pos_dir = 1 if target_weights[s] > 0 else (-1 if target_weights[s] < 0 else 0)
            cost = execution.cost_rate(turnover, pos_dir)
            bar_cost += cost
            p_prev = closes[s][i - 1]
            p_curr = closes[s][i]
            ret = (p_curr / p_prev - 1) if p_prev > 0 else 0.0
            bar_gross += prev_weights[s] * ret

        total_costs += bar_cost
        returns.append(bar_gross - bar_cost)
        prev_weights = target_weights

    equity = [1.0]
    for value in returns[1:]:
        equity.append(equity[-1] * (1 + value))
    dummy_positions = [1 if r > 0 else 0 for r in returns]
    return BacktestResult(equity, returns, dummy_positions, total_trades, total_costs)
