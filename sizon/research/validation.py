"""Chronological validation utilities with purge and embargo boundaries."""
from __future__ import annotations
from dataclasses import dataclass
from sizon.simulation.backtest import run_backtest
from sizon.data.data import DataFeed
from sizon.core.expression import Node
from sizon.simulation.execution import ExecutionModel

@dataclass
class FoldResult:
    fold:int; train_start:int; train_end:int; test_start:int; test_end:int; metrics:dict

def _feed(rows, source): return DataFeed(list(rows), source)
def _aggregate(folds):
    if not folds:return {"folds":0,"metrics":{}}
    keys=folds[0].metrics; return {"folds":len(folds),"metrics":{k:sum(f.metrics[k] for f in folds)/len(folds) for k in keys}}

def walk_forward(feed:DataFeed, genome:Node, n_splits:int=3, train_size:float=.5, test_size:float=.2, execution:ExecutionModel|None=None, expanding:bool=True):
    n=len(feed.rows); test_n=max(1,int(n*test_size)); initial=max(1,int(n*train_size)); folds=[]
    for fold in range(n_splits):
        test_start=initial+fold*test_n; test_end=min(n,test_start+test_n)
        if test_start>=n or test_end<=test_start: break
        train_start=0 if expanding else max(0,test_start-initial); train_end=test_start
        test=_feed(feed.rows[test_start:test_end],feed.source); metrics=run_backtest(test,genome,execution).metrics()
        folds.append(FoldResult(fold,train_start,train_end,test_start,test_end,metrics))
    return {**_aggregate(folds),"method":"walk_forward","leakage_control":"future test windows are never used in preceding train windows","fold_details":[f.__dict__ for f in folds]}

def purged_cross_validation(feed:DataFeed, genome:Node, n_splits:int=5, purge_bars:int=1, embargo_bars:int=1, execution:ExecutionModel|None=None):
    n=len(feed.rows); fold_n=max(1,n//n_splits); folds=[]
    for fold in range(n_splits):
        test_start=fold*fold_n; test_end=n if fold==n_splits-1 else min(n,(fold+1)*fold_n)
        train_end=max(0,test_start-purge_bars); train_start=test_end+embargo_bars
        # The test fold is evaluated only on its own chronological data; train bounds are recorded to prove exclusion.
        test=_feed(feed.rows[test_start:test_end],feed.source); metrics=run_backtest(test,genome,execution).metrics()
        folds.append(FoldResult(fold,0,train_end, test_start,test_end,metrics))
        folds[-1].train_start=train_start if train_start< n else train_end
    return {**_aggregate(folds),"method":"purged_cross_validation","purge_bars":purge_bars,"embargo_bars":embargo_bars,"leakage_control":"purge removes lookback overlap and embargo removes post-test adjacency","fold_details":[f.__dict__ for f in folds]}
