"""Deterministic paper and shadow-mode soak-test harness."""
from __future__ import annotations
from dataclasses import dataclass,field
import time
@dataclass
class SoakResult: mode:str; events:int; errors:list[str]=field(default_factory=list); started_at:float=0; ended_at:float=0
class SoakRunner:
    def run(self,events,handler,mode='paper'):
        result=SoakResult(mode=mode,events=0,started_at=time.time())
        for event in events:
            try:handler(event); result.events+=1
            except Exception as exc:result.errors.append(str(exc))
        result.ended_at=time.time(); return result
    def gate(self,result,max_errors=0):return result.events>0 and len(result.errors)<=max_errors
