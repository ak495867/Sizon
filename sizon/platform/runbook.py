"""Operational readiness checks for staged deployment."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ReadinessChecklist:
    items: dict[str, bool] = field(
        default_factory=lambda: {
            "data_lineage_verified": False,
            "paper_soak_passed": False,
            "shadow_soak_passed": False,
            "reconciliation_passed": False,
            "risk_gates_enabled": True,
            "kill_switch_tested": False,
            "secrets_configured": False,
            "incident_route_configured": False,
            "manual_approval_recorded": False,
        }
    )

    def complete(self):
        return all(self.items.values())

    def report(self):
        return {
            "ready": self.complete(),
            "items": dict(self.items),
            "blocking": [k for k, v in self.items.items() if not v],
        }


@dataclass(frozen=True)
class ApprovalRecord:
    approver: str
    strategy_id: str
    environment: str
    approved_at: str
    expiry: str
