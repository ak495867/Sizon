"""Calibrated order-book execution primitives for research simulation.

This module is deterministic and accepts observed book snapshots or a configured
synthetic book. It never pretends to be a venue adapter.
"""

from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BookLevel:
    price: float
    quantity: float


@dataclass(frozen=True)
class OrderBookSnapshot:
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    timestamp: str | None = None

    @property
    def mid(self):
        return (self.bids[0].price + self.asks[0].price) / 2


@dataclass(frozen=True)
class ImpactCurve:
    half_spread_bps: float = 1.0
    temporary_bps: float = 2.0
    permanent_bps: float = 0.5
    participation_limit: float = 0.1

    def cost_bps(self, quantity, available):
        participation = abs(quantity) / max(available, 1e-12)
        return (
            self.half_spread_bps
            + self.temporary_bps * (participation**0.5)
            + self.permanent_bps * participation
        )


@dataclass
class Fill:
    filled_quantity: float
    average_price: float
    fees: float
    queue_ahead_remaining: float
    status: str


def simulate_limit_fill(
    order_side,
    quantity,
    limit_price,
    book: OrderBookSnapshot,
    queue_ahead=0.0,
    participation_limit=0.1,
) -> Fill:
    levels = book.asks if order_side.lower() == "buy" else book.bids
    remaining = float(quantity)
    notional = fees = filled = 0.0
    queue = max(0.0, queue_ahead)
    available = sum(lvl.quantity for lvl in levels) * participation_limit
    for level in levels:
        valid = (
            level.price <= limit_price
            if order_side.lower() == "buy"
            else level.price >= limit_price
        )
        if not valid:
            break
        consume = min(remaining, max(0.0, level.quantity - queue), available - filled)
        filled += consume
        remaining -= consume
        notional += consume * level.price
        queue = max(0.0, queue - level.quantity)
        if remaining <= 0:
            break
    status = "filled" if remaining <= 0 else "partial" if filled > 0 else "unfilled"
    return Fill(filled, notional / filled if filled else 0.0, fees, queue, status)


def execute_market(
    order_side, quantity, book: OrderBookSnapshot, curve: ImpactCurve, fee_bps=0.0
) -> Fill:
    levels = book.asks if order_side.lower() == "buy" else book.bids
    remaining = float(quantity)
    notional = filled = 0.0
    available = sum(lvl.quantity for lvl in levels)
    for level in levels:
        take = min(remaining, level.quantity)
        filled += take
        remaining -= take
        notional += take * level.price
        if remaining <= 0:
            break
    if not filled:
        return Fill(0, 0, 0, 0, "unfilled")
    impact = curve.cost_bps(filled, available) / 10000
    price = (
        notional / filled * (1 + impact if order_side.lower() == "buy" else 1 - impact)
    )
    return Fill(
        filled,
        price,
        price * filled * fee_bps / 10000,
        0,
        "filled" if remaining <= 0 else "partial",
    )


def stress_test_microstructure(
    feed,
    genome,
    execution=None,
    curve: ImpactCurve | None = None,
    participation_limit: float = 0.1,
) -> dict:
    from sizon.simulation.backtest import run_backtest

    base = run_backtest(feed, genome, execution)
    curve = curve or ImpactCurve(participation_limit=participation_limit)
    close = feed.column("close")
    volume = feed.column("volume")
    spread_bps = execution.spread_bps if execution else 1.0
    fee_bps = execution.commission_bps if execution else 2.0

    micro_returns = [0.0]
    fills = []
    positions = base.positions

    for i in range(1, len(close)):
        turnover = positions[i] - positions[i - 1]
        gross = positions[i - 1] * (close[i] / close[i - 1] - 1)
        if turnover != 0:
            side = "buy" if turnover > 0 else "sell"
            qty = abs(turnover)
            mid = close[i]
            half_spr = (spread_bps / 20000) * mid
            level_vol = max(1.0, volume[i] * 0.25)
            book = OrderBookSnapshot(
                bids=(
                    BookLevel(mid - half_spr, level_vol),
                    BookLevel(mid - 3 * half_spr, level_vol * 2),
                ),
                asks=(
                    BookLevel(mid + half_spr, level_vol),
                    BookLevel(mid + 3 * half_spr, level_vol * 2),
                ),
            )
            fill = execute_market(side, qty, book, curve, fee_bps=fee_bps)
            fills.append(fill)
            slip_pct = abs(fill.average_price - mid) / mid if mid > 0 else 0.0
            fee_pct = (
                (fill.fees / (fill.average_price * qty))
                if (fill.average_price * qty) > 0
                else 0.0
            )
            cost = slip_pct + fee_pct
            micro_returns.append(gross - cost)
        else:
            micro_returns.append(gross)

    eq = [1.0]
    for r in micro_returns[1:]:
        eq.append(eq[-1] * (1 + r))

    r_slice = micro_returns[1:]
    mean_r = sum(r_slice) / max(1, len(r_slice))
    var_r = sum((x - mean_r) ** 2 for x in r_slice) / max(1, len(r_slice) - 1)
    micro_sharpe = (math.sqrt(252) * mean_r / math.sqrt(var_r)) if var_r > 0 else 0.0

    peak = eq[0]
    worst = 0.0
    for val in eq:
        peak = max(peak, val)
        worst = min(worst, val / peak - 1)
    micro_dd = abs(worst)

    return {
        "baseline_sharpe": base.sharpe,
        "microstructure_sharpe": micro_sharpe,
        "baseline_drawdown": base.max_drawdown,
        "microstructure_drawdown": micro_dd,
        "trades_simulated": len(fills),
        "fill_rate": sum(1 for f in fills if f.status == "filled") / max(1, len(fills)),
        "passed": micro_sharpe > 0 and (micro_dd <= max(0.01, base.max_drawdown * 1.5)),
    }


def microstructure_gauntlet(
    feed,
    genomes: list,
    execution=None,
    curve: ImpactCurve | None = None,
) -> list[dict]:
    results = []
    for g in genomes:
        diag = stress_test_microstructure(feed, g, execution, curve)
        results.append(diag)
    return results
