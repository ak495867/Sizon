"""Point-in-time universe membership; symbols are visible only after inclusion and before exit."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Membership:
    symbol: str
    included_at: str
    excluded_at: str | None = None
    exchange: str | None = None
    sector: str | None = None


class SurvivorshipSafeUniverse:
    def __init__(self, memberships: list[Membership]):
        self.memberships = memberships
        # Pre-sort by start date for efficient range queries
        self._sorted = sorted(memberships, key=lambda m: m.included_at)

    def symbols_at(self, date: str) -> list[str]:
        """Return symbols active on `date`. O(N) but with early termination possible."""
        return sorted(
            m.symbol
            for m in self._sorted
            if m.included_at <= date and (m.excluded_at is None or m.excluded_at > date)
        )

    def metadata_at(self, date: str) -> list[Membership]:
        """Return Membership objects active on `date`."""
        return [
            m for m in self._sorted
            if m.included_at <= date and (m.excluded_at is None or m.excluded_at > date)
        ]

    def validate(self):
        errors = []
        for m in self.memberships:
            if m.excluded_at and m.excluded_at <= m.included_at:
                errors.append(f"invalid membership window: {m.symbol}")
        return {
            "ok": not errors,
            "errors": errors,
            "symbols": len({m.symbol for m in self.memberships}),
        }
