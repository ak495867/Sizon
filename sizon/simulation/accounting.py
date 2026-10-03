"""Double-entry-ish portfolio ledger and constrained cross-sectional allocation."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LedgerEntry:
    timestamp: str
    kind: str
    symbol: str
    quantity: float
    price: float
    fees: float = 0.0


@dataclass
class PortfolioLedger:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    entries: list[LedgerEntry] = field(default_factory=list)
    max_records: int = 10000

    def trade(self, timestamp, symbol, quantity, price, fees=0.0):
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        self.cash -= quantity * price + fees
        self.entries.append(
            LedgerEntry(timestamp, "trade", symbol, quantity, price, fees)
        )
        if len(self.entries) >= self.max_records:
            trim = max(1, int(self.max_records * 0.1))
            self.entries = self.entries[trim:]

    def mark_to_market(self, prices):
        # Skip missing prices
        return self.cash + sum(q * prices.get(s) for s, q in self.positions.items() if prices.get(s) is not None)

    def snapshot(self, prices):
        return {
            "cash": self.cash,
            "equity": self.mark_to_market(prices),
            "positions": dict(self.positions),
            "entries": len(self.entries),
        }


def optimize_cross_sectional(scores, volatility=None, max_gross=1.0, max_position=0.1):
    volatility = volatility or {s: 1.0 for s in scores}
    normalized = {
        s: float(v) / max(volatility.get(s, 1.0), 1e-12)
        for s, v in scores.items()
    }
    pos_sum = sum(v for v in normalized.values() if v > 0) or 1.0
    neg_sum = sum(abs(v) for v in normalized.values() if v < 0) or 1.0
    
    result = {}
    for s, v in normalized.items():
        if v > 0:
            result[s] = min(max_position, (v / pos_sum) * (max_gross / 2.0))
        elif v < 0:
            result[s] = max(-max_position, (v / neg_sum) * (max_gross / 2.0))
        else:
            result[s] = 0.0
    return result
