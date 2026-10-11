"""
defense/kb_extraction_defense/evaluation/sim_server.py

In-process stand-in for a data source's /query endpoint, with pluggable defenses,
for fast, deterministic, multi-seed evaluation. Retrieval is the real dense top-k
over the real corpus (see eval_common.Corpus.search); only the HTTP hop is removed.

Modes
-----
none          no defense (what the deployment did before: only the 5-result cap).
ratelimit     per-client sliding-window limit (the deployed Flask-Limiter: 60 / 60 s),
              in LOGICAL time so a sweep does not need wall-clock minutes.
oracle_v1     the previous defense (QueryDiversityThrottle). It needs a topic label for
              every query; a real server has none, so it is given the topic of the
              top-1 retrieved document -- an ORACLE UPPER BOUND on what v1 could do.
guard         extraction_guard.ExtractionGuard (label-free), the new defense.
"""
from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Hashable, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.kb_extraction.adaptive_attack import QueryResult  # noqa: E402
from defense.kb_extraction_defense.evaluation.eval_common import Corpus, Embedder  # noqa: E402
from defense.kb_extraction_defense.extraction_guard import (  # noqa: E402
    ALLOW, DEGRADE, FREEZE, ExtractionGuard, GuardConfig, cluster_corpus,
)
from defense.kb_extraction_defense.legacy_v1.query_diversity_throttle import QueryDiversityThrottle  # noqa: E402

_CLUSTER_CACHE: Dict = {}
SERVER_K_CAP = 5   # drag_data_source/app/server.py: k = min(int(k), 5)


@dataclass
class Response:
    status: str                     # ok | blocked | ratelimited
    results: List[Dict]
    action: str = ALLOW
    reason: str = ""


class SimServer:
    def __init__(self, corpus: Corpus, embedder: Embedder, mode: str = "none",
                 guard_cfg: Optional[GuardConfig] = None, use_dispersion: bool = False,
                 rate_limit: int = 60, rate_window_s: float = 60.0,
                 v1_cfg: Optional[dict] = None):
        assert mode in ("none", "ratelimit", "oracle_v1", "guard")
        self.corpus, self.embedder, self.mode = corpus, embedder, mode
        self.guard = None
        if mode == "guard":
            clusters = None
            if guard_cfg.breadth_thresh is not None:
                key = (id(corpus), guard_cfg.n_clusters)
                if key not in _CLUSTER_CACHE:
                    lab = cluster_corpus(corpus.emb, guard_cfg.n_clusters)
                    _CLUSTER_CACHE[key] = {d: int(l) for d, l in zip(corpus.ids, lab)}
                clusters = _CLUSTER_CACHE[key]
            self.guard = ExtractionGuard(guard_cfg, doc_clusters=clusters)
        self.use_dispersion = use_dispersion
        self.rate_limit, self.rate_window = rate_limit, rate_window_s
        self._rl: Dict[Hashable, Deque[float]] = {}
        self.v1 = QueryDiversityThrottle(**(v1_cfg or {})) if mode == "oracle_v1" else None
        self.queries_served = 0

    def query(self, client: Hashable, text: str, k: int = 5, now: float = 0.0,
              trusted: bool = False) -> Response:
        k = min(k, SERVER_K_CAP)
        q_emb = self.embedder.embed([text])[0]

        if trusted or self.mode == "none":
            self.queries_served += 1
            return Response("ok", self.corpus.search(q_emb, k))

        if self.mode == "ratelimit":
            dq = self._rl.setdefault(client, deque())
            while dq and now - dq[0] >= self.rate_window:
                dq.popleft()
            if len(dq) >= self.rate_limit:
                return Response("ratelimited", [], reason="rate_limit")
            dq.append(now)
            self.queries_served += 1
            return Response("ok", self.corpus.search(q_emb, k))

        if self.mode == "oracle_v1":
            res = self.corpus.search(q_emb, k)
            topic = self.corpus.topics[self.corpus.id_index[res[0]["id"]]] if res else None
            allowed, why = self.v1.check_and_record(client, text, topic)
            if not allowed:
                return Response("blocked", [], action=FREEZE, reason=why)
            self.queries_served += 1
            return Response("ok", res)

        # ---- label-free guard
        dec = self.guard.decide(client, k, now=now)
        eff_k = k if dec.k is None else min(k, dec.k)
        res = self.corpus.search(q_emb, eff_k if dec.allowed_ids is None else k)
        res = ExtractionGuard.apply(dec, res)
        self.guard.observe(client, [r["id"] for r in res],
                           q_emb if self.use_dispersion else None, now=now)
        self.queries_served += 1
        return Response("ok", res, action=dec.action, reason=dec.reason)


class SimTransport:
    """Adapts SimServer to the attacker Transport; advances a logical clock."""

    def __init__(self, server: SimServer, client_fn, qps: float = 1.0, start: float = 0.0,
                 trusted: bool = False):
        self.server, self.client_fn = server, client_fn
        self.dt = 1.0 / qps
        self.now = start
        self.n = 0
        self.trusted = trusted
        self.log: List[tuple] = []          # (status, action) per attempt

    def query(self, text: str, k: int = 5) -> QueryResult:
        cid = self.client_fn(self.n)
        resp = self.server.query(cid, text, k, now=self.now, trusted=self.trusted)
        self.now += self.dt
        self.n += 1
        self.log.append((resp.status, resp.action))
        return QueryResult(resp.status, [{"id": r["id"], "text": r["text"]} for r in resp.results])
