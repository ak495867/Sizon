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
            price_val = float(order.get("price") or 0)
            qty_val = float(order.get("quantity") or 0)
            notional = qty_val * price_val
            errors = self.rules.validate_order(notional)
            if errors:
                result.rejected.append({"order": order, "reasons": errors})
                continue
            qty = self.rules.round_quantity(qty_val)
            price = self.rules.round_price(price_val)
            
            fill_qty = qty
            if events:
                available_vol = sum(
                    float(e.payload.get("quantity", 0)) 
                    for e in events 
                    if e.kind == "trade" and abs(float(e.payload.get("price", 0)) - price) < 1e-9
                )
                if available_vol < qty:
                    fill_qty = available_vol
                    
            if fill_qty <= 0:
                result.rejected.append({"order": order, "reasons": ["insufficient_liquidity"]})
                continue
                
            fee_bps = (
                self.rules.taker_fee_bps
                if order.get("liquidity", "taker") == "taker"
                else self.rules.maker_fee_bps
            )
            result.fills.append(
                {
                    "order_id": order.get("order_id"),
                    "quantity": fill_qty,
                    "price": price,
                    "fee": abs(fill_qty * price) * fee_bps / 10000,
                    "status": "filled" if fill_qty == qty else "partial",
                    "venue": self.rules.venue,
                    "timestamp": order.get("timestamp", events[-1].timestamp if events else None)
                }
            )
        return result
