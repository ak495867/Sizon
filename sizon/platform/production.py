"""Operational safety contracts; real-money execution is disabled by default."""

from __future__ import annotations
from dataclasses import dataclass, field
import os
import time


@dataclass
class SafetyConfig:
    environment: str = "research"
    allow_live_orders: bool = False
    max_daily_loss: float = 0.02
    max_order_notional: float = 1000
    require_manual_approval: bool = True


@dataclass
class HealthMonitor:
    alerts: list[dict] = field(default_factory=list)

    def emit(self, level, message, **details):
        self.alerts.append(
            {
                "timestamp": time.time(),
                "level": level,
                "message": message,
                "details": details,
            }
        )

    def health(self):
        return {
            "ok": not any(a["level"] == "critical" for a in self.alerts),
            "alerts": len(self.alerts),
        }


class DeploymentGate:
    def __init__(self, config=None):
        self.config = config or SafetyConfig()

    def authorize(self, notional, manual_approval=False):
        if not self.config.allow_live_orders:
            raise PermissionError("live orders disabled by safety configuration")
        if self.config.require_manual_approval and not manual_approval:
            raise PermissionError("manual approval required")
        if abs(notional) > self.config.max_order_notional:
            raise PermissionError("order exceeds notional limit")
        return True


def read_secret(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"missing secret: {name}")
    return value
