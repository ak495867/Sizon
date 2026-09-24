"""Controlled deployment abstractions: paper, shadow, kill switch, and broker interfaces."""
from __future__ import annotations
from dataclasses import dataclass,field
class BrokerAdapter:
    def submit(self,order):raise NotImplementedError
    def cancel_all(self):raise NotImplementedError
@dataclass
class ShadowMode:
    broker:BrokerAdapter|None=None; decisions:list[dict]=field(default_factory=list)
    def record(self,signal,market):self.decisions.append({"signal":signal,"market":market,"would_trade":True}); return self.decisions[-1]
@dataclass
class KillSwitch:
    tripped:bool=False; reason:str|None=None
    def trip(self,reason):self.tripped=True; self.reason=reason
    def reset(self):self.tripped=False; self.reason=None
    def allow(self):return not self.tripped
class SimulatedBroker(BrokerAdapter):
    def __init__(self):self.orders=[]
    def submit(self,order):self.orders.append(order); return {"status":"simulated","order":order}
    def cancel_all(self):count=len(self.orders); self.orders.clear(); return {"cancelled":count}
