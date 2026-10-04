import math
from typing import Dict, List
from sizon.data.data import DataFeed
from sizon.core.expression import Node, Context
from sizon.simulation.backtest import BacktestResult
from sizon.simulation.execution import ExecutionModel, DEFAULT_EXECUTION

class CentralClearingBook:
    def simulate_concurrent(
        self,
        feed: DataFeed,
        strategies: List[Node],
        execution_model: ExecutionModel = None,
        threshold: float = 0.0
    ) -> Dict[str, BacktestResult]:
        execution = execution_model or DEFAULT_EXECUTION
        close = feed.column("close")
        
        # Precompute signals for all strategies
        raw_signals = []
        ctx = Context({k: feed.column(k) for k in ("close", "high", "low", "volume")})
        for strat in strategies:
            sig = strat.evaluate(ctx)
            raw = [0 if math.isnan(v) else (1 if v > threshold else (0 if abs(v - threshold) < 1e-12 else -1)) for v in sig]
            raw_signals.append(raw)
            
        n_bars = len(close)
        n_strats = len(strategies)
        delay = max(1, execution.delay_bars)
        
        positions = [[0] * n_bars for _ in range(n_strats)]
        returns = [[0.0] for _ in range(n_strats)]
        trades = [0 for _ in range(n_strats)]
        costs = [0.0 for _ in range(n_strats)]
        
        for i in range(1, n_bars):
            orders = []
            for j in range(n_strats):
                if i >= delay:
                    positions[j][i] = raw_signals[j][i - delay]
                else:
                    positions[j][i] = 0
                
                order_j = positions[j][i] - positions[j][i - 1]
                orders.append(order_j)
            
            net_order = sum(orders)
            if net_order != 0:
                agg_cost = execution.cost_rate(abs(net_order))
                cost_per_unit = agg_cost / net_order
            else:
                cost_per_unit = 0.0
                
            for j in range(n_strats):
                order_j = orders[j]
                if order_j != 0:
                    trades[j] += 1
                
                trade_cost = order_j * cost_per_unit
                carry_cost = execution.carry_rate(positions[j][i])
                total_cost = trade_cost + carry_cost
                costs[j] += total_cost
                
                gross = positions[j][i - 1] * (close[i] / close[i - 1] - 1) if close[i-1] > 0 else 0.0
                returns[j].append(gross - total_cost)
                
        results = {}
        for j in range(n_strats):
            equity = [1.0]
            for r in returns[j][1:]:
                equity.append(equity[-1] * (1 + r))
            results[f"strat_{j}"] = BacktestResult(
                equity=equity,
                returns=returns[j],
                positions=positions[j],
                trades=trades[j],
                costs_paid=costs[j]
            )
            
        return results
