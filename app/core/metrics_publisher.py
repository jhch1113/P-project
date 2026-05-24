"""
core/metrics_publisher.py
========================
Posts latest metrics to the orchestrator at a fixed interval.
"""

import json
import logging
import threading
import urllib.request
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class MetricsPublisher:
    def __init__(
        self,
        name: str,
        url: str,
        interval_sec: float,
        payload_fn: Callable[[], Optional[dict]],
        timeout_sec: float = 2.0,
    ) -> None:
        self._name = name
        self._url = url
        self._interval_sec = interval_sec
        self._payload_fn = payload_fn
        self._timeout_sec = timeout_sec
        self._stop_event = threading.Event()

    def start(self) -> threading.Thread:
        t = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"MetricsPublisher-{self._name}",
        )
        t.start()
        return t

    def stop(self) -> None:
        self._stop_event.set()

    def _post_json(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_sec) as resp:
            resp.read()

    def _run(self) -> None:
        logger.info("MetricsPublisher start: name=%s url=%s", self._name, self._url)
        while not self._stop_event.is_set():
            payload = None
            try:
                payload = self._payload_fn()
            except Exception:
                logger.exception("MetricsPublisher payload error: name=%s", self._name)

            if payload:
                try:
                    self._post_json(payload)
                except Exception:
                    logger.warning(
                        "MetricsPublisher post failed: name=%s",
                        self._name,
                        exc_info=True,
                    )

            self._stop_event.wait(timeout=self._interval_sec)
        logger.info("MetricsPublisher stop: name=%s", self._name)
