"""Controlled deployment abstractions: paper, shadow, kill switch, and broker interfaces."""

from __future__ import annotations
from dataclasses import dataclass, field


class BrokerAdapter:
    def submit(self, order):
        raise NotImplementedError

    def cancel_all(self):
        raise NotImplementedError


@dataclass
class ShadowMode:
    broker: BrokerAdapter | None = None
    max_decisions: int = 10000
    decisions: list[dict] = field(default_factory=list)

    def record(self, signal, market):
        decision = {"signal": signal, "market": market, "would_trade": True}
        self.decisions.append(decision)
        if len(self.decisions) > self.max_decisions:
            self.decisions = self.decisions[-self.max_decisions:]
        return self.decisions[-1]


@dataclass
class KillSwitch:
    tripped: bool = False
    reason: str | None = None

    def trip(self, reason):
        self.tripped = True
        self.reason = reason

    def reset(self):
        self.tripped = False
        self.reason = None

    def allow(self):
        return not self.tripped


class SimulatedBroker(BrokerAdapter):
    def __init__(self, max_orders: int = 10000):
        self.orders: list = []
        self.max_orders = max_orders

    def submit(self, order):
        self.orders.append(order)
        if len(self.orders) > self.max_orders:
            self.orders = self.orders[-self.max_orders:]
        return {"status": "simulated", "order": order}

    def cancel_all(self):
        count = len(self.orders)
        self.orders.clear()
        return {"cancelled": count}
