"""
defense/kb_extraction_defense/query_diversity_throttle.py

Countermeasure for attack/kb_extraction's topic-balanced dataset-question
extraction strategy. No noise-injection defense (as an idealized design doc
for this attack describes -- Laplace noise on a per-peer similarity score)
is applicable to this repo's actual system: drag_data_source's /query
returns raw retrieved document text and drag_llm_service's /query returns
only `{"response": text}` -- there is no numeric confidence-score field
anywhere in the client-visible response to add calibrated noise to. This
is the same architectural fact defense/mia_defense/mia_defense.py's
docstring already establishes for the sibling membership-inference attack
("no numeric score field to add calibrated noise to, unlike the original
DRAG system's confidence-score noise defense") -- it applies identically
here.

What this defends against instead: drag_data_source's existing
Flask-Limiter (60 requests/minute per IP, see drag_data_source/app/server.py)
already rate-limits raw request *volume*, but the extraction attack's
recommended strategy is specifically topic-balanced (deliberately spreading
queries across many different subjects to maximize coverage per query
budget, see attack/kb_extraction/run_attack.py's ground-truth topic_coverage
metric) -- an attacker can stay comfortably under a per-minute request cap
while still touching an anomalous number of distinct topics. This module
adds the missing signal: not "how many requests," but "how many distinct
subjects has this client asked about recently" -- exactly the "reputation-
or rate-based query throttling" direction identified as an open gap for
this attack family.

Real ground truth for "topic" is the SQuAD article title associated with
each loaded document (see attack/kb_extraction/run_attack.py's
context_to_title) -- the actual corpus stored on disk carries no separate
topic/category field, so this is the same real signal the attack script
itself uses for topic_coverage, not an invented one.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Set, Tuple


class QueryDiversityThrottle:
    """
    Per-client sliding-window topic-diversity anomaly detector. Call
    `check_and_record(client_key, query_text, topic)` once per incoming
    query; it returns (allowed, reason) and updates the client's rolling
    window internally -- safe to call from a request-gating hook (see
    attack/kb_extraction/run_attack.py's `probe_source(..., query_gate=...)`)
    or, in a real deployment, a Flask `before_request` hook analogous to
    drag_data_source/app/server.py's existing `check_api_key()`.

    Parameters
    ----------
    window_size              : number of most-recent queries considered per client
    max_topics_per_window     : flag once a client's distinct topics touched in
                                 that window exceeds this
    min_queries_before_check  : don't evaluate diversity until a client has sent
                                 at least this many queries -- avoids flagging a
                                 legitimate client on a handful of naturally
                                 varied early questions
    cooldown_queries          : once flagged, block this many subsequent queries
                                 from that client before re-evaluating (bounded,
                                 not a permanent ban -- a legitimate client whose
                                 early queries happened to be unusually varied
                                 isn't locked out forever)
    """

    def __init__(
        self,
        window_size: int = 30,
        max_topics_per_window: int = 12,
        min_queries_before_check: int = 15,
        cooldown_queries: int = 20,
    ):
        self.window_size = window_size
        self.max_topics_per_window = max_topics_per_window
        self.min_queries_before_check = min_queries_before_check
        self.cooldown_queries = cooldown_queries

        self._windows: Dict[str, Deque[Optional[str]]] = {}
        self._query_counts: Dict[str, int] = {}
        self._cooldown_remaining: Dict[str, int] = {}
        self._flagged_at_query: Dict[str, int] = {}
        self.total_flagged = 0
        self.total_blocked = 0

    def _window_for(self, client_key: str) -> Deque[Optional[str]]:
        if client_key not in self._windows:
            self._windows[client_key] = deque(maxlen=self.window_size)
        return self._windows[client_key]

    def topics_in_window(self, client_key: str) -> Set[str]:
        window = self._windows.get(client_key, deque())
        return {t for t in window if t is not None}

    def check_and_record(self, client_key: str, query_text: str, topic: Optional[str]) -> Tuple[bool, str]:
        self._query_counts[client_key] = self._query_counts.get(client_key, 0) + 1
        cooldown = self._cooldown_remaining.get(client_key, 0)

        if cooldown > 0:
            self._cooldown_remaining[client_key] = cooldown - 1
            self.total_blocked += 1
            return False, "cooldown"

        window = self._window_for(client_key)
        window.append(topic)

        n = self._query_counts[client_key]
        if n >= self.min_queries_before_check:
            distinct_topics = len(self.topics_in_window(client_key))
            if distinct_topics > self.max_topics_per_window:
                self._cooldown_remaining[client_key] = self.cooldown_queries
                self._flagged_at_query[client_key] = n
                self.total_flagged += 1
                self.total_blocked += 1
                return False, "topic_diversity_exceeded"

        return True, "ok"

    def get_stats(self) -> Dict[str, object]:
        return {
            "clients_tracked": len(self._query_counts),
            "total_flagged": self.total_flagged,
            "total_blocked": self.total_blocked,
            "flagged_at_query": dict(self._flagged_at_query),
            "final_topics_touched": {
                client: len(self.topics_in_window(client)) for client in self._windows
            },
        }
