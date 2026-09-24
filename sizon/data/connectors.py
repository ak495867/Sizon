"""Explicit market-data connector contracts with retries, lineage, and safe defaults."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
from hashlib import sha256
import json,time,urllib.request
@dataclass(frozen=True)
class VendorConfig:
    name:str; base_url:str; dataset_version:str; timezone:str='UTC'; point_in_time:bool=True; adjusted:bool=False
@dataclass(frozen=True)
class DatasetLineage:
    vendor:str; dataset_version:str; retrieved_at:str; source_hash:str; point_in_time:bool; survivorship_safe:bool
class MarketDataConnector:
    def __init__(self,config:VendorConfig):self.config=config
    def fetch(self,request_path='',headers=None,retries=3,timeout=10):
        url=self.config.base_url.rstrip('/')+'/'+request_path.lstrip('/')
        last=None
        for attempt in range(retries):
            try:
                req=urllib.request.Request(url,headers=headers or {}); body=urllib.request.urlopen(req,timeout=timeout).read(); return body
            except Exception as exc:
                last=exc; time.sleep(min(2**attempt,.5))
        raise RuntimeError(f'data connector failed after {retries} attempts') from last
    def lineage(self,payload):
        digest=sha256(payload if isinstance(payload,bytes) else json.dumps(payload,sort_keys=True).encode()).hexdigest()
        return DatasetLineage(self.config.name,self.config.dataset_version,datetime.now(timezone.utc).isoformat(),digest,self.config.point_in_time,self.config.point_in_time)
class ConnectorRegistry:
    def __init__(self):self._connectors={}
    def register(self,connector):self._connectors[connector.config.name]=connector
    def get(self,name):return self._connectors[name]
    def names(self):return sorted(self._connectors)
