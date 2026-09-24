"""Explicit execution assumptions used by every Sizon backtest."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ExecutionModel:
    commission_bps: float = 2.0
    spread_bps: float = 1.0
    slippage_bps: float = 2.0
    impact_bps_per_turnover: float = 0.0
    delay_bars: int = 1
    funding_bps_per_bar: float = 0.0
    borrow_bps_per_bar: float = 0.0
    def cost_rate(self, turnover: float, position: int = 0) -> float:
        fixed=(self.commission_bps+self.spread_bps/2+self.slippage_bps)/10000
        impact=self.impact_bps_per_turnover*abs(turnover)/10000
        carry=(self.funding_bps_per_bar + (self.borrow_bps_per_bar if position<0 else 0))/10000
        return fixed*abs(turnover)+impact+carry
    def as_dict(self): return self.__dict__.copy()

DEFAULT_EXECUTION=ExecutionModel()
