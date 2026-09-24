"""Repeatable robustness tests for each discovered strategy."""
from __future__ import annotations
import math, random
from dataclasses import replace
from sizon.simulation.backtest import run_backtest
from sizon.data.data import DataFeed
from sizon.simulation.execution import ExecutionModel
from sizon.core.expression import Node, Primitive, Binary

def cost_sensitivity(feed:DataFeed, genome:Node, execution:ExecutionModel|None=None):
    base=execution or ExecutionModel(); results=[]
    for multiplier in (0.5,1.0,1.5,2.0,3.0):
        e=replace(base,commission_bps=base.commission_bps*multiplier,spread_bps=base.spread_bps*multiplier,slippage_bps=base.slippage_bps*multiplier,impact_bps_per_turnover=base.impact_bps_per_turnover*multiplier)
        results.append({"cost_multiplier":multiplier,"execution":e.as_dict(),"metrics":run_backtest(feed,genome,e).metrics()})
    return {"method":"cost_sensitivity","scenarios":results}

def _perturb(node:Node, delta:int)->Node:
    if isinstance(node,Primitive): return Primitive(node.name,max(2,node.period+delta))
    if isinstance(node,Binary): return Binary(node.op,_perturb(node.left,delta),_perturb(node.right,delta))
    return node

def parameter_sensitivity(feed:DataFeed, genome:Node, execution:ExecutionModel|None=None, deltas=(-3,-2,-1,0,1,2,3)):
    return {"method":"parameter_sensitivity","scenarios":[{"delta":d,"metrics":run_backtest(feed,_perturb(genome,d),execution).metrics()} for d in deltas]}

def monte_carlo_returns(feed:DataFeed, genome:Node, execution:ExecutionModel|None=None, simulations:int=100, seed:int=7):
    result=run_backtest(feed,genome,execution); rng=random.Random(seed); source=result.returns[1:]; samples=[]
    if not source:return {"method":"monte_carlo_trade_reshuffle","simulations":0,"quantiles":{}}
    for _ in range(simulations):
        shuffled=list(source); rng.shuffle(shuffled); equity=1.0
        for r in shuffled: equity*=1+r
        mean=sum(shuffled)/len(shuffled); var=sum((x-mean)**2 for x in shuffled)/max(1,len(shuffled)-1); samples.append({"total_return":equity-1,"sharpe":math.sqrt(252)*mean/math.sqrt(var) if var else 0.0})
    def q(key,p):
        values=sorted(x[key] for x in samples); return values[min(len(values)-1,max(0,int((len(values)-1)*p)))]
    return {"method":"monte_carlo_trade_reshuffle","simulations":simulations,"seed":seed,"quantiles":{"total_return":{"p05":q("total_return",.05),"p50":q("total_return",.5),"p95":q("total_return",.95)},"sharpe":{"p05":q("sharpe",.05),"p50":q("sharpe",.5),"p95":q("sharpe",.95)}}}

def robustness_suite(feed:DataFeed, genome:Node, execution:ExecutionModel|None=None, simulations:int=100, seed:int=7):
    return {"cost_sensitivity":cost_sensitivity(feed,genome,execution),"parameter_sensitivity":parameter_sensitivity(feed,genome,execution),"monte_carlo":monte_carlo_returns(feed,genome,execution,simulations,seed)}
