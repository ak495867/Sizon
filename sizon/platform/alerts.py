"""Alert delivery abstractions; credentials remain outside source code."""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import logging
import urllib.request


@dataclass
class Alert:
    level: str
    title: str
    message: str
    tags: dict = field(default_factory=dict)


class AlertSink:
    def send(self, alert):
        raise NotImplementedError


class LogSink(AlertSink):
    def send(self, alert):
        logging.getLogger("sizon").log(
            logging.ERROR if alert.level == "critical" else logging.WARNING,
            "%s: %s",
            alert.title,
            alert.message,
        )


class WebhookSink(AlertSink):
    def __init__(self, url):
        self.url = url

    def send(self, alert):
        body = json.dumps(
            {
                "level": alert.level,
                "title": alert.title,
                "message": alert.message,
                "tags": alert.tags,
            }
        ).encode()
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(req, timeout=5).status


class AlertRouter:
    def __init__(self, sinks=None):
        self.sinks = sinks or [LogSink()]

    def emit(self, level, title, message, **tags):
        alert = Alert(level, title, message, tags)
        for sink in self.sinks:
            sink.send(alert)
        return alert
