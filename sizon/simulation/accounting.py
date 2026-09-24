"""Double-entry-ish portfolio ledger and constrained cross-sectional allocation."""
from __future__ import annotations
from dataclasses import dataclass,field
@dataclass
class LedgerEntry: timestamp:str; kind:str; symbol:str; quantity:float; price:float; fees:float=0.0
@dataclass
class PortfolioLedger:
    cash:float; positions:dict[str,float]=field(default_factory=dict); entries:list[LedgerEntry]=field(default_factory=list)
    def trade(self,timestamp,symbol,quantity,price,fees=0.0):
        self.positions[symbol]=self.positions.get(symbol,0)+quantity; self.cash-=quantity*price+fees; self.entries.append(LedgerEntry(timestamp,"trade",symbol,quantity,price,fees))
    def mark_to_market(self,prices):return self.cash+sum(q*prices.get(s,0) for s,q in self.positions.items())
    def snapshot(self,prices):return {"cash":self.cash,"equity":self.mark_to_market(prices),"positions":dict(self.positions),"entries":len(self.entries)}

def optimize_cross_sectional(scores,volatility=None,max_gross=1.0,max_position=.1):
    volatility=volatility or {s:1.0 for s in scores}; positive={s:max(0.0,float(v))/max(volatility.get(s,1.0),1e-12) for s,v in scores.items()}; total=sum(positive.values()) or 1.0; return {s:min(max_position,v/total*max_gross) for s,v in positive.items()}
