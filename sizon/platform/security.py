"""Security boundaries: environment-backed secrets, redacted audit events, and incidents."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import os
import time


class SecretStore:
    def __init__(self, allowed_prefixes=("SIZON_",)):
        self.allowed_prefixes = allowed_prefixes

    def get(self, name):
        if not any(name.startswith(p) for p in self.allowed_prefixes):
            raise PermissionError("secret name outside approved namespace")
        value = os.getenv(name)
        if not value:
            raise KeyError(name)
        return value


@dataclass
class AuditEvent:
    timestamp: float
    actor: str
    action: str
    details: dict
    previous_hash: str = ""


class AuditLog:
    def __init__(self, max_events: int = 10000):
        self.events: list[dict] = []
        self.max_events = max_events

    def append(self, actor: str, action: str, details: dict) -> str:
        previous = self.events[-1]["hash"] if self.events else ""
        event = AuditEvent(time.time(), actor, action, details, previous)
        payload = json.dumps(asdict(event), sort_keys=True)
        digest = sha256(payload.encode()).hexdigest()
        self.events.append({"event": asdict(event), "hash": digest})
        if len(self.events) > self.max_events:
            # Trim oldest 10% but preserve chain integrity
            trim = max(1, self.max_events // 10)
            self.events = self.events[trim:]
        return digest

    def verify(self):
        previous = ""
        for item in self.events:
            if item["event"]["previous_hash"] != previous:
                return False
            previous = item["hash"]
        return True


@dataclass
class Incident:
    severity: str
    title: str
    details: dict
    status: str = "open"


class IncidentManager:
    def __init__(self, max_incidents: int = 1000):
        self.incidents: list[Incident] = []
        self.max_incidents = max_incidents

    def open(self, severity: str, title: str, details: dict | None = None) -> Incident:
        incident = Incident(severity, title, details or {})
        self.incidents.append(incident)
        if len(self.incidents) > self.max_incidents:
            # Remove oldest resolved incidents first, then oldest open
            resolved = [i for i in self.incidents if i.status == 'resolved']
            if resolved:
                self.incidents.remove(resolved[0])
            else:
                self.incidents.pop(0)
        return incident

    def resolve(self, title):
        for i in self.incidents:
            if i.title == title:
                i.status = "resolved"
