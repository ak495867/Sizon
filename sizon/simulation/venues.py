"""Venue rulebooks and calibration records used by event-driven replay."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal,ROUND_DOWN
@dataclass(frozen=True)
class VenueRules:
    venue:str; tick_size:float; lot_size:float; maker_fee_bps:float; taker_fee_bps:float; initial_margin:float; maintenance_margin:float; max_order_notional:float
    def round_price(self,price):return float((Decimal(str(price))/Decimal(str(self.tick_size))).quantize(Decimal('1'),rounding=ROUND_DOWN)*Decimal(str(self.tick_size)))
    def round_quantity(self,quantity):return float((Decimal(str(quantity))/Decimal(str(self.lot_size))).quantize(Decimal('1'),rounding=ROUND_DOWN)*Decimal(str(self.lot_size)))
    def margin_required(self,notional):return abs(notional)*self.initial_margin
    def validate_order(self,notional):
        errors=[]
        if abs(notional)>self.max_order_notional:errors.append('max_order_notional')
        if abs(notional)==0:errors.append('zero_notional')
        return errors
@dataclass(frozen=True)
class CalibrationRecord:
    venue:str; symbol:str; observed_from:str; observed_to:str; median_spread_bps:float; fill_rate:float; latency_ms:float; source_hash:str
