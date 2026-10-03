"""Venue rulebooks and calibration records used by event-driven replay."""

from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class VenueRules:
    venue: str
    tick_size: float
    lot_size: float
    maker_fee_bps: float
    taker_fee_bps: float
    initial_margin: float
    maintenance_margin: float
    max_order_notional: float

    def round_price(self, price: float) -> float:
        if not self.tick_size:
            return price
        rounded = round(price / self.tick_size) * self.tick_size
        decimals = max(0, -int(math.floor(math.log10(self.tick_size)))) if self.tick_size < 1 else 0
        return round(rounded, decimals)

    def round_quantity(self, quantity: float) -> float:
        if not self.lot_size:
            return quantity
        rounded = math.floor(quantity / self.lot_size) * self.lot_size
        decimals = max(0, -int(math.floor(math.log10(self.lot_size)))) if self.lot_size < 1 else 0
        return round(rounded, decimals)

    def margin_required(self, notional):
        return abs(notional) * self.initial_margin

    def validate_order(self, notional):
        errors = []
        if abs(notional) > self.max_order_notional:
            errors.append("max_order_notional")
        if abs(notional) == 0:
            errors.append("zero_notional")
        return errors


@dataclass(frozen=True)
class CalibrationRecord:
    venue: str
    symbol: str
    observed_from: str
    observed_to: str
    median_spread_bps: float
    fill_rate: float
    latency_ms: float
    source_hash: str
