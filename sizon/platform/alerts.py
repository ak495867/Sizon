"""Alert delivery abstractions; credentials remain outside source code."""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import logging
import threading
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
    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self.timeout = timeout

    def send(self, alert: Alert) -> None:
        """Dispatch webhook in background thread to avoid blocking the trading loop."""
        threading.Thread(target=self._post, args=(alert,), daemon=True).start()

    def _post(self, alert: Alert) -> None:
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
        try:
            urllib.request.urlopen(req, timeout=self.timeout)
        except Exception as exc:
            logging.getLogger('sizon.alerts').warning('Webhook delivery failed: %s', exc)


class AlertRouter:
    def __init__(self, sinks=None):
        self.sinks = sinks or [LogSink()]

    def emit(self, level, title, message, **tags):
        alert = Alert(level, title, message, tags)
        for sink in self.sinks:
            sink.send(alert)
        return alert
