"""Portfolio-level execution constraints for research backtests."""
from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash:float=100000.0; max_gross_exposure:float=1.0; max_net_exposure:float=1.0; max_position_pct:float=1.0; leverage:float=1.0; margin_rate:float=.25; min_order_size:float=0.0; tick_size:float=0.01; daily_loss_limit:float=1.0; liquidation_threshold:float=1.0
@dataclass
class Order:
    symbol:str; side:str; quantity:float; order_type:str="market"; limit_price:float|None=None; stop_price:float|None=None; status:str="created"; filled_quantity:float=0.0; fill_price:float|None=None
@dataclass
class PortfolioState:
    cash:float; equity:float; gross_exposure:float=0.0; net_exposure:float=0.0; margin_used:float=0.0; liquidated:bool=False

def round_tick(price:float,tick_size:float)->float: return round(price/tick_size)*tick_size if tick_size else price
def validate_order(order:Order, price:float, config:PortfolioConfig)->list[str]:
    errors=[]
    if order.quantity<config.min_order_size: errors.append("below_minimum_order_size")
    if order.order_type=="limit" and order.limit_price is None: errors.append("limit_price_required")
    if order.order_type=="stop" and order.stop_price is None: errors.append("stop_price_required")
    if price<=0: errors.append("invalid_market_price")
    return errors
def execute_order(order:Order, market_price:float, liquidity:float, config:PortfolioConfig)->Order:
    errors=validate_order(order,market_price,config)
    if errors: order.status="rejected"; return order
    fill=min(abs(order.quantity),max(0,liquidity)); order.filled_quantity=fill; order.fill_price=round_tick(order.limit_price if order.order_type=="limit" and order.limit_price else market_price,config.tick_size); order.status="filled" if fill==abs(order.quantity) else "partial"; return order
def apply_risk_limits(target_position:float, equity:float, price:float, config:PortfolioConfig)->float:
    if equity<=0:return 0.0
    max_units=equity*config.max_position_pct*config.leverage/max(price,1e-12)
    return max(-max_units,min(max_units,target_position))
def liquidation_check(state:PortfolioState, config:PortfolioConfig)->bool:
    if state.equity<=0 or (state.margin_used>0 and state.equity/state.margin_used<config.liquidation_threshold): state.liquidated=True
    return state.liquidated
