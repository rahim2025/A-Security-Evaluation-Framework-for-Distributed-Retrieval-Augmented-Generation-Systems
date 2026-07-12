"""
attack/ddos_sim/live_flood.py

A *real* congestion-based DDoS against the actual drag_data_source Docker
containers -- concurrent HTTP request flooding, not a simulation. Distinct
from attack/ddos_sim/ddos_attack.py's DDoSAttack (a seeded, in-process
probability model): this module generates genuine concurrent load against
the real Flask servers so a real client (drag_llm_service, or the
evaluation questions in run_live_evaluation.py) experiences real
congestion -- either the source's Flask-Limiter (60/min default, see
drag_data_source/app/server.py) returning genuine HTTP 429s, or, just as
importantly, real queueing delay: `app.run()` in that server is not
started with `threaded=True`, so concurrent flood requests genuinely
compete for the same single request-handling slot as any other client
hitting the same container, independent of whether the flood's own source
IP shares the Flask-Limiter's per-IP bucket with drag_llm_service's
container-to-container requests. That queueing effect is what makes this
mechanism work even if IP-bucketing differs between host-originated flood
traffic and container-to-container traffic on the Docker bridge network.

Never removes/kills a peer -- purely additional read-only load against
your own local `/query` endpoint, safely bounded by `stop()`/`duration_s`.
"""
from __future__ import annotations

import random
import string
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

_FLOOD_QUESTIONS = [
    "what is the effect of treatment on patient outcomes",
    "how does this condition affect long term survival",
    "what are the risk factors associated with this disease",
    "is there a significant difference between the two groups",
    "what method was used to evaluate the intervention",
]


@dataclass
class FloodStats:
    requests_sent: int = 0
    responses_200: int = 0
    responses_429: int = 0
    errors: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, status: Optional[int]) -> None:
        with self.lock:
            self.requests_sent += 1
            if status == 200:
                self.responses_200 += 1
            elif status == 429:
                self.responses_429 += 1
            elif status is None:
                self.errors += 1

    def as_dict(self) -> Dict[str, int]:
        with self.lock:
            return {
                "requests_sent": self.requests_sent,
                "responses_200": self.responses_200,
                "responses_429": self.responses_429,
                "errors": self.errors,
            }


class TrafficFlood:
    """Starts `workers_per_source` daemon threads per targeted source, each
    firing `/query` requests back-to-back until `stop()` is called. Call
    `start()` before the real workload you want to congest, and `stop()`
    once it's done -- the flood's lifetime is caller-controlled, not a
    fixed duration, so it always covers exactly the real work it's meant
    to interfere with."""

    def __init__(self, source_urls: Dict[str, str], workers_per_source: int = 4,
                 api_key: str = "reliable-derag-secret-2026", request_timeout: float = 5.0):
        self.source_urls = source_urls
        self.workers_per_source = workers_per_source
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.stats: Dict[str, FloodStats] = {name: FloodStats() for name in source_urls}
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

    def _worker(self, name: str, url: str) -> None:
        headers = {"Content-Type": "application/json", "X-API-Key": self.api_key}
        stats = self.stats[name]
        while not self._stop_event.is_set():
            q = random.choice(_FLOOD_QUESTIONS) + " " + "".join(random.choices(string.ascii_lowercase, k=4))
            try:
                r = requests.post(f"{url}/query", headers=headers, json={"query": q, "k": 3},
                                   timeout=self.request_timeout)
                stats.record(r.status_code)
            except Exception:
                stats.record(None)

    def start(self, target_source_names: Optional[List[str]] = None) -> None:
        targets = target_source_names or list(self.source_urls.keys())
        self._stop_event.clear()
        for name in targets:
            url = self.source_urls[name]
            for _ in range(self.workers_per_source):
                t = threading.Thread(target=self._worker, args=(name, url), daemon=True)
                t.start()
                self._threads.append(t)

    def stop(self, join_timeout: float = 5.0) -> Dict[str, Dict[str, int]]:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=join_timeout)
        self._threads.clear()
        return {name: s.as_dict() for name, s in self.stats.items()}
