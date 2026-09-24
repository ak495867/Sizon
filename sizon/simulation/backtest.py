"""Leakage-aware bar backtesting with explicit execution assumptions."""
from __future__ import annotations
from dataclasses import dataclass
import math
from sizon.data.data import DataFeed
from sizon.simulation.execution import ExecutionModel, DEFAULT_EXECUTION
from sizon.core.expression import Context, Node

@dataclass
class BacktestResult:
    equity:list[float]; returns:list[float]; positions:list[int]; trades:int=0; costs_paid:float=0.0
    @property
    def sharpe(self):
        r=self.returns[1:]
        if len(r)<2:return 0.0
        mean=sum(r)/len(r); var=sum((x-mean)**2 for x in r)/(len(r)-1)
        return math.sqrt(252)*mean/math.sqrt(var) if var else 0.0
    @property
    def sortino(self):
        r=self.returns[1:]; down=[min(0,x) for x in r]; denom=math.sqrt(sum(x*x for x in down)/max(1,len(down)))
        return math.sqrt(252)*sum(r)/max(1,len(r))/denom if denom else 0.0
    @property
    def max_drawdown(self):
        peak=self.equity[0]; worst=0.0
        for value in self.equity: peak=max(peak,value); worst=min(worst,value/peak-1)
        return abs(worst)
    @property
    def turnover(self): return self.trades/max(1,len(self.positions)-1)
    @property
    def total_return(self): return self.equity[-1]-1
    def metrics(self): return {"sharpe":self.sharpe,"sortino":self.sortino,"total_return":self.total_return,"max_drawdown":self.max_drawdown,"turnover":self.turnover,"trades":self.trades,"costs_paid":self.costs_paid}

def run_backtest(feed:DataFeed, genome:Node, execution:ExecutionModel|None=None, threshold:float=0.0)->BacktestResult:
    execution=execution or DEFAULT_EXECUTION; close=feed.column("close"); signal=genome.evaluate(Context({k:feed.column(k) for k in ("close","high","low","volume")})); raw=[0 if math.isnan(v) else (1 if v>threshold else -1) for v in signal]
    delay=max(1,execution.delay_bars); positions=[0]*len(raw); returns=[0.0]; trades=0; costs=0.0
    for i in range(1,len(close)):
        positions[i]=raw[i-delay] if i>=delay else 0; turnover=abs(positions[i]-positions[i-1]); changed=turnover>0; trades+=int(changed)
        gross=positions[i-1]*(close[i]/close[i-1]-1); cost=execution.cost_rate(turnover,positions[i]); costs+=cost; returns.append(gross-cost)
    equity=[1.0]
    for value in returns[1:]: equity.append(equity[-1]*(1+value))
    return BacktestResult(equity,returns,positions,trades,costs)

def fitness(result:BacktestResult, complexity:int): return {**result.metrics(),"complexity":float(complexity)}
