"""Point-in-time universe membership; symbols are visible only after inclusion and before exit."""
from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class Membership:
    symbol:str; included_at:str; excluded_at:str|None=None; exchange:str|None=None; sector:str|None=None
class SurvivorshipSafeUniverse:
    def __init__(self,memberships):self.memberships=list(memberships)
    def symbols_at(self,timestamp):return sorted({m.symbol for m in self.memberships if m.included_at<=timestamp and (m.excluded_at is None or timestamp<m.excluded_at)})
    def metadata_at(self,timestamp):return [m for m in self.memberships if m.included_at<=timestamp and (m.excluded_at is None or timestamp<m.excluded_at)]
    def validate(self):
        errors=[]
        for m in self.memberships:
            if m.excluded_at and m.excluded_at<=m.included_at: errors.append(f'invalid membership window: {m.symbol}')
        return {'ok':not errors,'errors':errors,'symbols':len({m.symbol for m in self.memberships})}
