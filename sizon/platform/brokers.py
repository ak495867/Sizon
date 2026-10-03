"""Broker integration contracts. Network adapters are opt-in and never bypass safety gates."""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import urllib.request


@dataclass(frozen=True)
class BrokerConfig:
    name: str
    base_url: str
    account_id: str
    environment: str = "paper"
    api_key_env: str = ""
    api_secret_env: str = ""


@dataclass
class BrokerState:
    positions: dict[str, float] = field(default_factory=dict)
    cash: float = 0.0
    open_orders: dict[str, dict] = field(default_factory=dict)


class BrokerAdapter:
    def __init__(self, config):
        self.config = config

    def submit(self, order):
        if self.config.environment != "paper":
            raise PermissionError(
                "live broker submission requires explicit production adapter and deployment approval"
            )
        return {"status": "paper_rejected_without_adapter", "order": order}

    def cancel_all(self):
        return {"status": "paper", "cancelled": 0}

    def snapshot(self):
        return BrokerState()


class HttpBrokerAdapter(BrokerAdapter):
    def request(self, path: str, payload: dict | None = None, headers: dict | None = None) -> bytes:
        req = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/" + path.lstrip("/"),
            data=json.dumps(payload or {}).encode(),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=10).read()
        except Exception as exc:
            raise RuntimeError(f'Broker request to {path} failed: {exc}') from exc


class Reconciler:
    def compare(self, internal: BrokerState, external: BrokerState) -> dict:
        symbols = set(internal.positions) | set(external.positions)
        position_deltas = {
            s: external.positions.get(s, 0.0) - internal.positions.get(s, 0.0)
            for s in symbols
            if not math.isclose(
                external.positions.get(s, 0.0),
                internal.positions.get(s, 0.0),
                rel_tol=1e-6,
                abs_tol=1e-8,
            )
        }
        cash_ok = math.isclose(external.cash, internal.cash, rel_tol=1e-6, abs_tol=1e-8)
        return {
            'ok': not position_deltas and cash_ok,
            'position_deltas': position_deltas,
            'cash_delta': external.cash - internal.cash,
        }
