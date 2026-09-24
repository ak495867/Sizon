"""Event-driven market replay primitives; no live order submission occurs here."""

from __future__ import annotations
from dataclasses import dataclass, field
from .venues import VenueRules, CalibrationRecord


@dataclass(frozen=True)
class MarketEvent:
    timestamp: str
    kind: str
    payload: dict


@dataclass
class ReplayResult:
    fills: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    events: int = 0


class OrderBookReplay:
    def __init__(self, rules: VenueRules, calibration: CalibrationRecord | None = None):
        self.rules = rules
        self.calibration = calibration

    def replay(self, events, orders):
        result = ReplayResult(events=len(events))
        for order in orders:
            notional = float(order.get("quantity", 0)) * float(order.get("price", 0))
            errors = self.rules.validate_order(notional)
            if errors:
                result.rejected.append({"order": order, "reasons": errors})
                continue
            qty = self.rules.round_quantity(order["quantity"])
            price = self.rules.round_price(order["price"])
            fee_bps = (
                self.rules.taker_fee_bps
                if order.get("liquidity", "taker") == "taker"
                else self.rules.maker_fee_bps
            )
            result.fills.append(
                {
                    "order_id": order.get("order_id"),
                    "quantity": qty,
                    "price": price,
                    "fee": abs(qty * price) * fee_bps / 10000,
                    "status": "filled",
                    "venue": self.rules.venue,
                }
            )
        return result
