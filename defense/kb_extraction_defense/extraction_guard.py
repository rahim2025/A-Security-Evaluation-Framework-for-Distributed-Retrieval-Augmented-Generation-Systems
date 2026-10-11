"""
defense/kb_extraction_defense/extraction_guard.py

Label-free, server-side guard against knowledge-base extraction.

Threat
------
A caller that can reach a data source's /query endpoint with the shared API key
(stolen key, insider, or any non-orchestrator client) can crawl the corpus: every
query returns the top-k *full documents*, so enough well-chosen queries
reconstruct the knowledge base. A per-IP rate limit only bounds the speed of
that crawl; it never bounds how much is eventually recovered.

Design principle
-----------------
An extraction attacker's objective is to maximise the number of *new* documents
discovered per query. That quantity is directly observable by the server with no
topic labels, no oracle, and no model of the attacker: the server knows which
document ids it has already returned to each caller. A legitimate caller keeps
returning to a small working set (its "topic"), so its discovery rate decays; an
extractor cannot be efficient and look repetitive at the same time. Evading the
detector costs the attacker throughput in direct proportion, and a cumulative
budget bounds what any strategy can recover per epoch.

Server-observable mechanisms (all optional / independently tunable; enabled signals are AND-ed):

  1. Breadth              number of distinct corpus regions the caller's last `window` queries
                          landed in. Regions are unsupervised pseudo-topics: spherical k-means over
                          the server's OWN document embeddings, and a query's region is the region
                          of its top-1 result. A focused reader stays in few regions; a crawler
                          sweeps many. Needs no labels and no oracle.
  2. Discovery velocity   (optional) new documents first-seen by this caller per query. Measured
                          to be weak on its own (a legitimate cold start and a decoy-diluted
                          crawler both fool it), kept as an option and for the ablation.
  3. Query dispersion     (optional) spread of recent query embeddings, 1 - |mean unit embedding|.
  4. Unique-document budget  hard cap on distinct documents served per caller per epoch
                          (client_budget) and across ALL untrusted callers (global_budget, the
                          Sybil-resistant backstop). This is the security guarantee: whatever the
                          attacker does, it recovers at most B documents per identity per epoch.

Graded response, so a false positive is cheap and an attack is bounded:
  ALLOW   -> serve normally.
  DEGRADE -> serve at most `degrade_k` results (caps discovery at degrade_k docs
             per query) for `cooldown_queries` queries.
  FREEZE  -> serve only documents this caller (or, for the global budget, any
             untrusted caller) has already been served: re-reading is still
             possible, discovering anything new is not. Entered when a budget is
             exhausted or after `escalate_after_flags` flags in one epoch.

Trusted callers (the orchestrator, authenticated by a second credential) bypass
the guard entirely; the guard is for everyone else.

Nothing here reads topic labels, ground truth, or document text: only caller id,
returned document ids, and (optionally) query embeddings.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Hashable, Iterable, List, Optional, Sequence, Set

import numpy as np

ALLOW = "allow"
DEGRADE = "degrade"
FREEZE = "freeze"


@dataclass
class GuardConfig:
    window: int = 30
    min_queries: int = 10
    velocity_thresh: Optional[float] = None    # new docs / query; None = signal disabled
    dispersion_thresh: Optional[float] = None  # None = signal disabled
    breadth_thresh: Optional[int] = None       # distinct corpus clusters touched in the window; None = disabled
    n_clusters: int = 16                       # unsupervised pseudo-topics the server derives from its OWN corpus
    degrade_k: int = 1
    cooldown_queries: int = 20
    escalate_after_flags: int = 3              # flags in one epoch before FREEZE; 0 disables
    client_budget: Optional[int] = None        # unique docs per caller per epoch
    global_budget: Optional[int] = None        # unique docs across all untrusted callers per epoch
    epoch_seconds: float = 86400.0
    max_clients: int = 10000

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Decision:
    action: str = ALLOW
    k: Optional[int] = None                    # cap on results to serve (None = caller's request)
    allowed_ids: Optional[Set[Hashable]] = None  # FREEZE: only these ids may be returned
    reason: str = ""
    signals: Dict[str, float] = field(default_factory=dict)


def cluster_corpus(emb: np.ndarray, k: int = 16, seed: int = 0, iters: int = 25) -> np.ndarray:
    """Unsupervised pseudo-topics: spherical k-means over the server's OWN document embeddings.

    No human labels and no oracle -- the server already holds these embeddings. Returns an
    int cluster id per document. Pure numpy (k-means++ init, Lloyd iterations)."""
    x = np.asarray(emb, dtype=np.float32)
    n = len(x)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    cent = [x[rng.integers(n)]]
    for _ in range(1, k):
        d = 1.0 - np.max(x @ np.stack(cent).T, axis=1)
        p = np.clip(d, 0, None) ** 2
        p = p / p.sum() if p.sum() > 0 else np.full(n, 1.0 / n)
        cent.append(x[rng.choice(n, p=p)])
    cent = np.stack(cent)
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        lab = np.argmax(x @ cent.T, axis=1)
        for j in range(k):
            m = x[lab == j]
            if len(m):
                v = m.mean(axis=0)
                cent[j] = v / (np.linalg.norm(v) + 1e-12)
    return lab


class _Client:
    __slots__ = ("seen", "new_counts", "embs", "emb_sum", "n_queries", "degraded_left",
                 "flags", "frozen", "epoch_start", "first_flag_query", "frozen_reason", "clusters")

    def __init__(self, now: float):
        self.seen: Set[Hashable] = set()
        self.new_counts: Deque[int] = deque()
        self.clusters: Deque[frozenset] = deque()
        self.embs: Deque[np.ndarray] = deque()
        self.emb_sum: Optional[np.ndarray] = None
        self.n_queries = 0
        self.degraded_left = 0
        self.flags = 0
        self.frozen = False
        self.frozen_reason = ""
        self.epoch_start = now
        self.first_flag_query: Optional[int] = None


class ExtractionGuard:
    def __init__(self, cfg: Optional[GuardConfig] = None,
                 doc_clusters: Optional[Dict[Hashable, int]] = None):
        self.cfg = cfg or GuardConfig()
        self.doc_clusters = doc_clusters or {}
        self._clients: "OrderedDict[Hashable, _Client]" = OrderedDict()
        self._global_seen: Set[Hashable] = set()
        self._global_epoch_start: Optional[float] = None
        self._lock = threading.Lock()
        self.stats = {"decisions": 0, "allow": 0, "degrade": 0, "freeze": 0, "flags": 0,
                      "client_budget_hits": 0, "global_budget_hits": 0, "escalations": 0}

    # ------------------------------------------------------------------ state
    def _now(self, now: Optional[float]) -> float:
        return time.monotonic() if now is None else now

    def _roll_epoch(self, now: float) -> None:
        if self._global_epoch_start is None:
            self._global_epoch_start = now
        elif now - self._global_epoch_start >= self.cfg.epoch_seconds:
            self._global_seen = set()
            self._global_epoch_start = now

    def _client(self, cid: Hashable, now: float) -> _Client:
        st = self._clients.get(cid)
        if st is None:
            st = _Client(now)
            self._clients[cid] = st
            if len(self._clients) > self.cfg.max_clients:
                self._clients.popitem(last=False)
        else:
            self._clients.move_to_end(cid)
            if now - st.epoch_start >= self.cfg.epoch_seconds:
                st = _Client(now)
                self._clients[cid] = st
        return st

    # ---------------------------------------------------------------- signals
    @staticmethod
    def _dispersion(st: _Client) -> float:
        n = len(st.embs)
        if n == 0 or st.emb_sum is None:
            return 0.0
        return float(max(0.0, 1.0 - np.linalg.norm(st.emb_sum) / n))

    def _signals(self, st: _Client) -> Dict[str, float]:
        n = len(st.new_counts)
        return {
            "window_queries": float(n),
            "velocity": (sum(st.new_counts) / n) if n else 0.0,
            "dispersion": self._dispersion(st),
            "breadth": float(len(frozenset().union(*st.clusters))) if st.clusters else 0.0,
            "unique_docs": float(len(st.seen)),
        }

    def _is_flag(self, sig: Dict[str, float]) -> bool:
        """Flag iff every ENABLED signal exceeds its threshold (signals are AND-ed: velocity
        alone fires on any legitimate caller's cold start, breadth/dispersion alone on a
        legitimate wide reader; together they describe 'scanning the corpus')."""
        c = self.cfg
        checks = []
        if c.velocity_thresh is not None:
            checks.append(sig["velocity"] >= c.velocity_thresh)
        if c.dispersion_thresh is not None:
            checks.append(sig["dispersion"] >= c.dispersion_thresh)
        if c.breadth_thresh is not None:
            checks.append(sig["breadth"] >= c.breadth_thresh)
        if not checks or sig["window_queries"] < c.min_queries:
            return False
        return all(checks)

    # -------------------------------------------------------------------- API
    def decide(self, client_id: Hashable, requested_k: int = 5,
               now: Optional[float] = None) -> Decision:
        """Called BEFORE retrieval for an untrusted caller."""
        with self._lock:
            now = self._now(now)
            self._roll_epoch(now)
            st = self._client(client_id, now)
            c = self.cfg
            sig = self._signals(st)
            self.stats["decisions"] += 1

            if c.global_budget is not None and len(self._global_seen) >= c.global_budget:
                self.stats["global_budget_hits"] += 1
                self.stats["freeze"] += 1
                return Decision(FREEZE, allowed_ids=set(self._global_seen),
                                reason="global_budget", signals=sig)
            if c.client_budget is not None and len(st.seen) >= c.client_budget:
                self.stats["client_budget_hits"] += 1
                self.stats["freeze"] += 1
                return Decision(FREEZE, allowed_ids=set(st.seen), reason="client_budget", signals=sig)
            if st.frozen:
                self.stats["freeze"] += 1
                return Decision(FREEZE, allowed_ids=set(st.seen), reason=st.frozen_reason, signals=sig)

            if st.degraded_left > 0:
                st.degraded_left -= 1
                self.stats["degrade"] += 1
                return Decision(DEGRADE, k=min(requested_k, c.degrade_k), reason="cooldown", signals=sig)

            if self._is_flag(sig):
                st.flags += 1
                self.stats["flags"] += 1
                if st.first_flag_query is None:
                    st.first_flag_query = st.n_queries
                if c.escalate_after_flags and st.flags >= c.escalate_after_flags:
                    st.frozen = True
                    st.frozen_reason = "escalated"
                    self.stats["escalations"] += 1
                    self.stats["freeze"] += 1
                    return Decision(FREEZE, allowed_ids=set(st.seen), reason="escalated", signals=sig)
                st.degraded_left = max(0, c.cooldown_queries - 1)
                self.stats["degrade"] += 1
                return Decision(DEGRADE, k=min(requested_k, c.degrade_k), reason="velocity", signals=sig)

            self.stats["allow"] += 1
            return Decision(ALLOW, signals=sig)

    def observe(self, client_id: Hashable, doc_ids: Iterable[Hashable],
                query_emb: Optional[np.ndarray] = None, now: Optional[float] = None) -> int:
        """Called AFTER retrieval with the ids actually returned. Returns #new docs."""
        with self._lock:
            now = self._now(now)
            self._roll_epoch(now)
            st = self._client(client_id, now)
            ids = list(doc_ids)
            new = [d for d in ids if d not in st.seen]
            st.seen.update(ids)
            if self.cfg.global_budget is not None:
                self._global_seen.update(ids)
            st.n_queries += 1
            st.new_counts.append(len(new))
            # a query's 'intent cluster' = the cluster of its top-1 result (the label-free analogue of
            # a topic label); the union over all k results is too noisy -- neighbours spill across clusters
            st.clusters.append(frozenset(
                [self.doc_clusters[ids[0]]] if ids and ids[0] in self.doc_clusters else []))
            if len(st.new_counts) > self.cfg.window:
                st.new_counts.popleft()
                st.clusters.popleft()
            if query_emb is not None:
                u = np.asarray(query_emb, dtype=np.float32)
                u = u / (np.linalg.norm(u) + 1e-12)
                st.embs.append(u)
                st.emb_sum = u.copy() if st.emb_sum is None else st.emb_sum + u
                if len(st.embs) > self.cfg.window:
                    st.emb_sum = st.emb_sum - st.embs.popleft()
            return len(new)

    @staticmethod
    def apply(decision: Decision, results: Sequence[Dict[str, Any]],
              id_key: str = "id") -> List[Dict[str, Any]]:
        """Restrict a retrieval result list according to the decision."""
        out = list(results)
        if decision.allowed_ids is not None:
            out = [r for r in out if r.get(id_key) in decision.allowed_ids]
        if decision.k is not None:
            out = out[: decision.k]
        return out

    # ------------------------------------------------------------- inspection
    def signals(self, client_id: Hashable) -> Dict[str, float]:
        with self._lock:
            st = self._clients.get(client_id)
            return self._signals(st) if st else {}

    def client_summary(self, client_id: Hashable) -> Dict[str, Any]:
        with self._lock:
            st = self._clients.get(client_id)
            if not st:
                return {}
            return {"queries": st.n_queries, "unique_docs": len(st.seen), "flags": st.flags,
                    "frozen": st.frozen, "first_flag_query": st.first_flag_query}

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {**self.stats, "clients_tracked": len(self._clients),
                    "global_unique_docs": len(self._global_seen), "config": self.cfg.to_dict()}

    def reset(self) -> None:
        with self._lock:
            self._clients.clear()
            self._global_seen = set()
            self._global_epoch_start = None
            for k in self.stats:
                self.stats[k] = 0
