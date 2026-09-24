"""Extension points for custom primitives, data connectors, and paper brokers."""
from __future__ import annotations
from dataclasses import dataclass
class PluginRegistry:
    def __init__(self): self.primitives={}; self.connectors={}; self.brokers={}
    def register_primitive(self,name,fn): self.primitives[name]=fn
    def register_connector(self,name,fn): self.connectors[name]=fn
    def register_broker(self,name,broker): self.brokers[name]=broker
@dataclass
class PaperOrder:
    symbol:str; side:str; quantity:float; limit_price:float|None=None
class PaperBroker:
    def __init__(self): self.orders=[]; self.enabled=False
    def enable(self,explicit=True): self.enabled=bool(explicit)
    def submit(self,order:PaperOrder):
        if not self.enabled: raise PermissionError("paper broker is disabled; enable explicitly")
        self.orders.append(order); return {"status":"paper_submitted","order":order.__dict__}
    def cancel_all(self): self.orders.clear(); return {"status":"cancelled_all"}
