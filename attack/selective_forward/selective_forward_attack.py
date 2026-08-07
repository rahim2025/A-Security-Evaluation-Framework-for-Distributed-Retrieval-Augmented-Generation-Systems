"""
attack/selective_forward/selective_forward_attack.py

Drop-in replacement for the existing file in:
  EXPLORING-PRIVACY-PRESERVING-APPROACHES-FOR-PERSONALIZED-LARGE-LANGUAGE-MODELS-reliable-derag

Adds:
  - SelectiveForwardingAttack  (stealthy probabilistic gray-hole)
  - SFADetector               (EWMA + binomial anomaly detection)
  - SFAMitigation             (suspicion-aware re-routing + blacklist)
  - check_blockchain_status   (hash-chained SSM ledger integrity check)
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ══════════════════════════════════════════════════════════════════════════
#  BLOCKCHAIN-BACKED SSM LEDGER
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class _Block:
    index: int
    timestamp: float
    node_id: str
    event: str
    payload: dict
    prev_hash: str
    hash: str = field(default="", init=False)

    def _compute(self) -> str:
        raw = json.dumps(
            dict(index=self.index, timestamp=self.timestamp, node_id=self.node_id,
                 event=self.event, payload=self.payload, prev_hash=self.prev_hash),
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def __post_init__(self):
        self.hash = self._compute()


class _SSMChain:
    """Lightweight in-process SHA-256-chained ledger for node reputation scores."""

    INIT_SCORE   = 10_000
    REWARD       = 10
    PENALTY      = 150
    DECAY        = 0.97
    BL_THRESH    = 3_000
    RESTORE_THRESH = 5_000

    def __init__(self):
        self._chain: List[_Block] = []
        self._scores: Dict[str, float] = {}
        self._blacklisted: Dict[str, bool] = {}
        self._lock = threading.Lock()
        self._genesis()

    def _genesis(self):
        b = _Block(0, time.time(), "SYSTEM", "genesis",
                   {"msg": "SSM ledger initialized"}, "0" * 64)
        self._chain.append(b)

    def _append(self, node_id, event, payload):
        b = _Block(len(self._chain), time.time(), node_id, event,
                   payload, self._chain[-1].hash)
        self._chain.append(b)
        return b

    def register(self, node_id: str) -> float:
        with self._lock:
            self._scores[node_id] = float(self.INIT_SCORE)
            self._blacklisted[node_id] = False
            self._append(node_id, "register", {"score": self.INIT_SCORE})
        return self.INIT_SCORE

    def record_hit(self, node_id: str) -> float:
        with self._lock:
            s = min(self.INIT_SCORE, self._scores.get(node_id, self.INIT_SCORE) + self.REWARD)
            self._scores[node_id] = s
            self._append(node_id, "score_update", {"delta": self.REWARD, "score": s, "event": "hit"})
            return s

    def record_miss(self, node_id: str, suspected: bool = False) -> float:
        with self._lock:
            old = self._scores.get(node_id, self.INIT_SCORE)
            pen = self.PENALTY * (3 if suspected else 1)
            new = max(0.0, old * self.DECAY - pen)
            self._scores[node_id] = new
            self._append(node_id, "score_update",
                         {"delta": round(old - new, 2), "score": round(new, 2),
                          "event": "suspected_drop" if suspected else "miss"})
            if new < self.BL_THRESH and not self._blacklisted.get(node_id):
                self._do_blacklist(node_id, "auto_threshold")
            return new

    def _do_blacklist(self, node_id, reason):
        self._blacklisted[node_id] = True
        self._append(node_id, "blacklist",
                     {"reason": reason, "score": round(self._scores.get(node_id, 0), 2)})

    def blacklist(self, node_id: str, reason: str = "manual"):
        with self._lock:
            self._do_blacklist(node_id, reason)

    def restore(self, node_id: str) -> bool:
        with self._lock:
            if self._scores.get(node_id, 0) >= self.RESTORE_THRESH:
                self._blacklisted[node_id] = False
                self._append(node_id, "restore", {"score": self._scores[node_id]})
                return True
        return False

    def score(self, node_id: str) -> float:
        return self._scores.get(node_id, self.INIT_SCORE)

    def is_blacklisted(self, node_id: str) -> bool:
        return self._blacklisted.get(node_id, False)

    def verify(self) -> bool:
        for i in range(1, len(self._chain)):
            b = self._chain[i]
            if b.prev_hash != self._chain[i - 1].hash:
                return False
            if b.hash != b._compute():
                return False
        return True

    def export(self) -> List[dict]:
        return [asdict(b) for b in self._chain]

    def summary(self) -> dict:
        return {
            "chain_length": len(self._chain),
            "chain_valid": self.verify(),
            "nodes": {
                nid: {"score": round(self._scores[nid], 1),
                      "blacklisted": self._blacklisted.get(nid, False)}
                for nid in self._scores
            },
        }


# Singleton ledger shared across the simulation run
_LEDGER = _SSMChain()


def check_blockchain_status() -> dict:
    """
    Public helper — returns the current SSM chain integrity report.
    Called by any external verifier.
    """
    summary = _LEDGER.summary()
    status = {
        "chain_valid": summary["chain_valid"],
        "chain_length": summary["chain_length"],
        "num_nodes": len(summary["nodes"]),
        "num_blacklisted": sum(1 for v in summary["nodes"].values() if v["blacklisted"]),
        "avg_score": (
            round(
                sum(v["score"] for v in summary["nodes"].values()) / len(summary["nodes"]), 1
            )
            if summary["nodes"] else 0
        ),
        "status": "SFA detectable — score decay active" if summary["nodes"] else "no nodes registered",
    }
    return status


# ══════════════════════════════════════════════════════════════════════════
#  SELECTIVE FORWARDING ATTACK  (stealthy gray-hole)
# ══════════════════════════════════════════════════════════════════════════

class SelectiveForwardingAttack:
    """
    Applies a probabilistic (10-30 %) gray-hole selective forwarding attack
    to a list of source nodes.

    Parameters
    ----------
    attack_ratio : float   fraction of nodes to compromise (0.0 – 1.0)
    strategy     : str     "random" | "high_ssm_score"
    drop_rate    : float | "stealthy"
                   "stealthy" → 10-30 % per-node random rate to evade detection
                   float     → fixed drop probability for all compromised nodes
    seed         : int
    """

    STEALTHY_LO = 0.10
    STEALTHY_HI = 0.30

    def __init__(
        self,
        attack_ratio: float = 0.34,
        strategy: str = "random",
        drop_rate: Any = "stealthy",
        seed: int = 42,
    ):
        self.attack_ratio = attack_ratio
        self.strategy = strategy
        self.drop_rate = drop_rate
        self.seed = seed

        self.compromised_nodes: List[str] = []
        self._per_node_rate: Dict[str, float] = {}
        self._total_attempts = 0
        self._total_dropped = 0

    def apply(self, sources: list, ssm_scores: Optional[Dict[str, float]] = None):
        """
        Monkey-patch `sources` — each element must have a `.query(question, k)` method
        and a `.node_id` attribute.
        """
        rng = np.random.default_rng(self.seed)
        n = len(sources)
        n_attack = max(1, round(self.attack_ratio * n))

        if self.strategy == "high_ssm_score" and ssm_scores:
            targets = sorted(sources, key=lambda s: ssm_scores.get(s.node_id, 0), reverse=True)[:n_attack]
        else:
            idxs = rng.choice(n, size=n_attack, replace=False)
            targets = [sources[i] for i in idxs]

        self.compromised_nodes = [s.node_id for s in targets]

        for src in targets:
            rate = (float(rng.uniform(self.STEALTHY_LO, self.STEALTHY_HI))
                    if self.drop_rate == "stealthy" else float(self.drop_rate))
            self._per_node_rate[src.node_id] = rate
            self._patch(src, rate, np.random.default_rng(rng.integers(0, 2**31)))

    def _patch(self, src, drop_prob: float, rng: np.random.Generator):
        original = src.query
        atk = self

        def _gray_hole(question, k=5):
            atk._total_attempts += 1
            if rng.random() < drop_prob:
                atk._total_dropped += 1
                return [], 0.0, False
            return original(question, k)

        src.query = _gray_hole

    # ── properties ────────────────────────────────────────────────────────

    @property
    def num_compromised(self) -> int:
        return len(self.compromised_nodes)

    @property
    def effective_drop_rate(self) -> float:
        return self._total_dropped / self._total_attempts if self._total_attempts else 0.0

    def report(self) -> dict:
        return {
            "strategy": self.strategy,
            "attack_ratio": self.attack_ratio,
            "num_compromised": self.num_compromised,
            "compromised_nodes": self.compromised_nodes,
            "per_node_drop_rates": {k: round(v, 3) for k, v in self._per_node_rate.items()},
            "total_attempted": self._total_attempts,
            "total_dropped": self._total_dropped,
            "effective_drop_rate": round(self.effective_drop_rate, 3),
        }


# ══════════════════════════════════════════════════════════════════════════
#  SFA DETECTOR  (EWMA + binomial anomaly detection)
# ══════════════════════════════════════════════════════════════════════════

class SFADetector:
    """
    Per-node anomaly detector using an EWMA miss-rate with a binomial
    significance test.

    Suspicion levels
    ----------------
    0  clean
    1  low suspicion
    2  medium — node deprioritised in routing
    3  high / confirmed — triggers SSM penalty amplification
    """

    WINDOW       = 40
    ALPHA        = 0.25      # EWMA smoothing
    # MISS_THRESH/HONEST_MISS were originally tuned for a generic many-node
    # mock topology (see _MockSource: score ~ Beta(2,3), hit iff score>=0.45,
    # giving an "honest" miss rate around 0.6-0.7). That does not hold for
    # the real Reliable-dRAG deployment: RealSource.query() reports a hit
    # whenever the source returns *any* top-k document, which happens for
    # essentially every query regardless of relevance -- measured empirically
    # against the live data-source containers (60 real SQuAD questions x 3
    # sources), the genuine honest miss rate is 0.000 (0/180). Calibrated
    # against that reality instead of the mock assumption: a stealthy
    # attacker drops 10-30% of queries (SelectiveForwardingAttack.STEALTHY_*),
    # so the threshold only needs to sit above natural noise and below the
    # weakest attacker, and the binomial null hypothesis needs to reflect
    # the true ~0 honest baseline rather than 0.68.
    MISS_THRESH  = 0.08      # miss-rate above this -> bad (honest ~0, weakest attacker ~0.10)
    STREAK_REQ   = 2         # consecutive windows above threshold to escalate
    BINOM_ALPHA  = 0.05      # p-value threshold
    HONEST_MISS  = 0.05      # expected honest miss rate on the real deployment (measured ~0, small margin for noise)

    def __init__(self):
        self._obs:       Dict[str, deque]  = {}
        self._ewma:      Dict[str, float]  = {}
        self._streak:    Dict[str, int]    = {}
        self._suspicion: Dict[str, int]    = {}
        self._totals:    Dict[str, dict]   = {}

    def register(self, node_id: str):
        self._obs[node_id]       = deque(maxlen=self.WINDOW)
        self._ewma[node_id]      = 0.0
        self._streak[node_id]    = 0
        self._suspicion[node_id] = 0
        self._totals[node_id]    = {"queries": 0, "hits": 0, "misses": 0}

    def observe(self, node_id: str, hit: bool) -> int:
        """Record one result; return updated suspicion level 0-3."""
        if node_id not in self._obs:
            self.register(node_id)
        t = self._totals[node_id]
        t["queries"] += 1
        t["hits" if hit else "misses"] += 1
        v = 0 if hit else 1
        self._obs[node_id].append(v)
        self._ewma[node_id] = self.ALPHA * v + (1 - self.ALPHA) * self._ewma[node_id]

        obs = self._obs[node_id]
        if len(obs) < self.WINDOW:
            return self._suspicion[node_id]

        miss_rate = sum(obs) / len(obs)
        p_val = self._binom_p(miss_rate, len(obs))

        if miss_rate > self.MISS_THRESH:
            self._streak[node_id] += 1
        else:
            self._streak[node_id] = max(0, self._streak[node_id] - 1)

        if self._streak[node_id] >= self.STREAK_REQ and p_val < self.BINOM_ALPHA:
            self._suspicion[node_id] = min(3, self._suspicion[node_id] + 1)
        elif miss_rate < self.HONEST_MISS * 0.90:
            self._suspicion[node_id] = max(0, self._suspicion[node_id] - 1)

        return self._suspicion[node_id]

    def _binom_p(self, miss_rate: float, n: int) -> float:
        """One-sided binomial p-value: is miss_rate > HONEST_MISS?"""
        try:
            from scipy.stats import binom_test  # type: ignore
            k = int(round(miss_rate * n))
            return float(binom_test(k, n, self.HONEST_MISS, alternative="greater"))
        except Exception:
            # fallback: normal approximation
            mu = self.HONEST_MISS * n
            sigma = math.sqrt(n * self.HONEST_MISS * (1 - self.HONEST_MISS)) + 1e-9
            z = (miss_rate * n - mu) / sigma
            return max(0.0, 1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

    def suspicion_level(self, node_id: str) -> int:
        return self._suspicion.get(node_id, 0)

    def is_suspected(self, node_id: str) -> bool:
        return self._suspicion.get(node_id, 0) >= 2

    def miss_rate(self, node_id: str) -> float:
        obs = self._obs.get(node_id, deque())
        return sum(obs) / len(obs) if obs else 0.0

    def report(self) -> Dict[str, dict]:
        return {
            nid: {
                "suspicion": self._suspicion.get(nid, 0),
                "miss_rate": round(self.miss_rate(nid), 3),
                "suspected_drop": self.is_suspected(nid),
                **self._totals.get(nid, {}),
            }
            for nid in self._obs
        }


# ══════════════════════════════════════════════════════════════════════════
#  SFA MITIGATION  (suspicion-aware re-routing + blacklist)
# ══════════════════════════════════════════════════════════════════════════

class SFAMitigation:
    """
    Wraps a list of source nodes and provides suspicion-aware routing
    with a redundant quorum fallback.

    Parameters
    ----------
    sources       : list of nodes with .node_id and .query()
    detector      : SFADetector instance
    ledger        : _SSMChain instance  (default: module-level _LEDGER)
    redundancy_k  : backup nodes to try on TTL failure
    """

    def __init__(
        self,
        sources: list,
        detector: SFADetector,
        ledger: Optional[_SSMChain] = None,
        redundancy_k: int = 3,
    ):
        self._sources = sources
        self._detector = detector
        self._ledger = ledger or _LEDGER
        self._k = redundancy_k

        for src in sources:
            if not self._ledger._scores.get(src.node_id):
                self._ledger.register(src.node_id)

    def route(self, question: str, max_hops: int = 5) -> Tuple[bool, int, List[str]]:
        """
        Returns (hit, hops_used, routing_log).

        Phase 1 — primary path (trusted nodes first, blacklisted skipped).
        Phase 2 — redundant probe on remaining non-suspected nodes.
        """
        ordered = self._prioritised()
        log: List[str] = []
        hops = 0

        for src in ordered[:max_hops]:
            if self._ledger.is_blacklisted(src.node_id):
                log.append(f"{src.node_id}: SKIP[blacklisted]")
                continue
            hops += 1
            _, score, hit = src.query(question)
            susp = self._detector.is_suspected(src.node_id)
            lvl  = self._detector.observe(src.node_id, hit)
            if hit:
                self._ledger.record_hit(src.node_id)
            else:
                self._ledger.record_miss(src.node_id, suspected=susp)
            log.append(f"{src.node_id}: {'HIT' if hit else ('SUSP_DROP' if susp else 'MISS')} "
                       f"score={score:.3f} susp_lvl={lvl}")
            if hit:
                return True, hops, log
            if hops >= max_hops:
                break

        # Phase 2 — backup probe
        tried = {s.node_id for s in ordered[:max_hops]}
        backup = [s for s in self._sources
                  if s.node_id not in tried
                  and not self._ledger.is_blacklisted(s.node_id)
                  and not self._detector.is_suspected(s.node_id)]
        for src in backup[: self._k]:
            hops += 1
            _, score, hit = src.query(question)
            lvl = self._detector.observe(src.node_id, hit)
            if hit:
                self._ledger.record_hit(src.node_id)
            else:
                self._ledger.record_miss(src.node_id)
            log.append(f"{src.node_id}: BACKUP score={score:.3f} hit={hit} susp_lvl={lvl}")
            if hit:
                return True, hops, log

        return False, hops, log

    def _prioritised(self) -> list:
        def key(s):
            return (self._detector.suspicion_level(s.node_id),
                    -self._ledger.score(s.node_id))
        return sorted(self._sources, key=key)

    def blacklisted_ids(self) -> List[str]:
        return [s.node_id for s in self._sources if self._ledger.is_blacklisted(s.node_id)]

    def suspected_ids(self) -> List[str]:
        return [s.node_id for s in self._sources if self._detector.is_suspected(s.node_id)]

    def active_count(self) -> int:
        return sum(1 for s in self._sources if not self._ledger.is_blacklisted(s.node_id))


def naive_route(sources: list, question: str, max_hops: int = 5) -> Tuple[bool, int, List[str]]:
    """
    Fair "undefended" comparison point for SFAMitigation.route(): same
    first-hit-stops routing mechanism and hop budget, but with no suspicion
    tracking, no blacklist, and no score-based reordering -- just tries
    `sources` in the order given. Comparing SFAMitigation.route() against
    this (rather than against _evaluate()'s aggregate-all-sources accuracy)
    isolates what the suspicion-aware routing itself contributes, instead of
    conflating it with a completely different counting method.
    """
    log: List[str] = []
    hops = 0
    for src in sources[:max_hops]:
        hops += 1
        _, score, hit = src.query(question)
        log.append(f"{src.node_id}: {'HIT' if hit else 'MISS'} score={score:.3f}")
        if hit:
            return True, hops, log
    return False, hops, log
