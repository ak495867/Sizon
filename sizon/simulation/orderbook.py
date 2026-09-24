"""Calibrated order-book execution primitives for research simulation.

This module is deterministic and accepts observed book snapshots or a configured
synthetic book. It never pretends to be a venue adapter.
"""
from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class BookLevel: price:float; quantity:float
@dataclass(frozen=True)
class OrderBookSnapshot:
    bids:tuple[BookLevel,...]; asks:tuple[BookLevel,...]; timestamp:str|None=None
    @property
    def mid(self): return (self.bids[0].price+self.asks[0].price)/2
@dataclass(frozen=True)
class ImpactCurve:
    half_spread_bps:float=1.0; temporary_bps:float=2.0; permanent_bps:float=.5; participation_limit:float=.1
    def cost_bps(self,quantity,available):
        participation=abs(quantity)/max(available,1e-12); return self.half_spread_bps+self.temporary_bps*(participation**.5)+self.permanent_bps*participation
@dataclass
class Fill:
    filled_quantity:float; average_price:float; fees:float; queue_ahead_remaining:float; status:str

def simulate_limit_fill(order_side,quantity,limit_price,book:OrderBookSnapshot,queue_ahead=0.0,participation_limit=.1)->Fill:
    levels=book.asks if order_side.lower()=="buy" else book.bids; remaining=float(quantity); notional=fees=filled=0.0; queue=max(0.0,queue_ahead); available=sum(l.quantity for l in levels)*participation_limit
    for level in levels:
        valid=level.price<=limit_price if order_side.lower()=="buy" else level.price>=limit_price
        if not valid: break
        consume=min(remaining,max(0.0,level.quantity-queue),available-filled); filled+=consume; remaining-=consume; notional+=consume*level.price; queue=max(0.0,queue-level.quantity)
        if remaining<=0: break
    status="filled" if remaining<=0 else "partial" if filled>0 else "unfilled"
    return Fill(filled,notional/filled if filled else 0.0,fees,queue,status)

def execute_market(order_side,quantity,book:OrderBookSnapshot,curve:ImpactCurve,fee_bps=0.0)->Fill:
    levels=book.asks if order_side.lower()=="buy" else book.bids; remaining=float(quantity); notional=filled=0.0; available=sum(l.quantity for l in levels)
    for level in levels:
        take=min(remaining,level.quantity); filled+=take; remaining-=take; notional+=take*level.price
        if remaining<=0: break
    if not filled:return Fill(0,0,0,0,"unfilled")
    impact=curve.cost_bps(filled,available)/10000; price=notional/filled*(1+impact if order_side.lower()=="buy" else 1-impact)
    return Fill(filled,price,price*filled*fee_bps/10000,0,"filled" if remaining<=0 else "partial")
