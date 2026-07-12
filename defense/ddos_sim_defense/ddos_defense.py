"""
defense/ddos_sim_defense/ddos_defense.py

Countermeasure for attack.ddos_sim.DDoSAttack, implementing the three
directions reports/SFA_Security_Analysis_Report.md's DDoS section
identifies as missing from this repo:

1. Load-aware / reputation-based deprioritization -- a defender has no
   legitimate access to the attacker's internal `load_penalty` state (that
   would be cheating the simulation), so this observes the same signal a
   real client actually has: per-peer response outcomes over time,
   EMA-blended into a reputation score exactly like
   defense/sfa_sim_defense's layer 1, then deprioritized once the response
   rate crosses `deprioritize_threshold`.
2. Routing bypass + bounded redundant backup probe -- reuses the *same*
   `network._sfa_defense` hook slot that network_sim.MockRAGNetwork /
   live_network.LiveRAGNetwork's `topic_aware_query()` already consults on
   every hop (see those modules): a deprioritized peer is skipped and its
   neighbours are expanded at the same hop depth instead. No changes to
   either network module are needed, because that hook is duck-typed --
   it only needs `is_peer_blacklisted()` / `record_bypass()` /
   `backup_candidates()` / `record_redundant_probe()`, not any
   SFA-specific behaviour. Deliberately reusing the existing slot rather
   than adding a second, parallel hook the routing loop would need to
   learn about.
3. Recovery-aware retry backoff -- the one piece the SFA defense doesn't
   need, because congestion (unlike a compromised peer) clears itself
   after a fixed duration in this model. A peer deprioritized here is
   automatically given a fresh evaluation after `backoff_waves` waves
   have passed since it was flagged, rather than staying deprioritized
   forever on stale evidence -- matching real-world DDoS mitigation
   (scrubbing, rate-limiting) restoring a peer within minutes.

Quorum-preserving cap (`max_blacklist_fraction`) is carried over from
defense/sfa_sim_defense/selective_forwarding_defense.py verbatim: a single
fixed response-rate threshold can't tell "this peer is being flooded" from
"the whole population's honest response rate is just low," so
auto-deprioritization never excludes more than this fraction of the known
population (and never all of it).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


class DDoSDefense:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.deprioritize_threshold = float(cfg.get("deprioritize_threshold", 0.5))
        self.min_queries_before_action = int(cfg.get("min_queries_before_action", 5))
        self.reputation_decay = float(cfg.get("reputation_decay", 0.6))
        self.reputation_ema_alpha = 1.0 - self.reputation_decay
        self.redundancy_k = int(cfg.get("redundancy_k", 2))
        self.max_blacklist_fraction = float(cfg.get("max_blacklist_fraction", 0.5))
        # Recovery-aware backoff: a peer auto-clears `backoff_waves` waves after
        # being deprioritized, getting a clean re-evaluation instead of staying
        # flagged forever on stale evidence (see module docstring point 3).
        self.backoff_waves = int(cfg.get("backoff_waves", 2))
        self._num_peers: Optional[int] = None
        self._current_wave = 0

        self._query_count: Dict[int, int] = {}
        self._response_count: Dict[int, int] = {}
        self._reputation: Dict[int, float] = {}
        self.blacklisted_peers: Set[int] = set()
        self._blacklisted_at_wave: Dict[int, int] = {}

        self._original_query_fns: Dict[int, Any] = {}
        self._total_bypasses = 0
        self._total_blacklistings = 0
        self._total_recoveries = 0
        self._redundant_probes = 0
        self._redundant_hits = 0

    # ── layer 1: reputation ─────────────────────────────────────────────
    def record_peer_interaction(self, peer_id: int, responded: bool) -> None:
        self._query_count[peer_id] = self._query_count.get(peer_id, 0) + 1
        self._response_count.setdefault(peer_id, 0)
        if responded:
            self._response_count[peer_id] += 1

        n = self._query_count[peer_id]
        raw_rate = self._response_count[peer_id] / n
        current = self._reputation.get(peer_id, 1.0)
        alpha = self.reputation_ema_alpha
        self._reputation[peer_id] = (1.0 - alpha) * current + alpha * raw_rate

        if peer_id in self.blacklisted_peers or n < self.min_queries_before_action:
            return
        if raw_rate < self.deprioritize_threshold and len(self.blacklisted_peers) < self._blacklist_cap():
            self.blacklisted_peers.add(peer_id)
            self._blacklisted_at_wave[peer_id] = self._current_wave
            self._total_blacklistings += 1

    def _blacklist_cap(self) -> int:
        if self._num_peers is None:
            return 10**9
        return max(0, min(int(self._num_peers * self.max_blacklist_fraction), self._num_peers - 1))

    def reputation(self, peer_id: int) -> float:
        return self._reputation.get(peer_id, 1.0)

    # ── layer 3: recovery-aware backoff ─────────────────────────────────
    def advance_wave(self, wave_num: int) -> List[int]:
        """Call once per attack wave (before that wave's queries run) so
        deprioritized peers get a fresh evaluation after `backoff_waves`
        waves, instead of staying flagged on evidence from a since-recovered
        congestion event."""
        self._current_wave = wave_num
        recovered = [
            pid for pid, at_wave in self._blacklisted_at_wave.items()
            if wave_num - at_wave >= self.backoff_waves
        ]
        for pid in recovered:
            self.blacklisted_peers.discard(pid)
            del self._blacklisted_at_wave[pid]
            self._reputation[pid] = 1.0
            self._query_count[pid] = 0
            self._response_count[pid] = 0
            self._total_recoveries += 1
        return recovered

    # ── phase 2: bounded redundant backup probe ─────────────────────────
    def backup_candidates(self, tried_ids, all_peer_ids: List[int]) -> List[int]:
        candidates = [pid for pid in all_peer_ids if pid not in tried_ids and not self.is_peer_blacklisted(pid)]
        candidates.sort(key=lambda pid: -self.reputation(pid))
        return candidates[: self.redundancy_k]

    def record_redundant_probe(self, hit: bool) -> None:
        self._redundant_probes += 1
        if hit:
            self._redundant_hits += 1

    # ── layer 2: blacklist / routing-bypass hook ────────────────────────
    def is_peer_blacklisted(self, peer_id: int) -> bool:
        return peer_id in self.blacklisted_peers

    def record_bypass(self) -> None:
        self._total_bypasses += 1

    # ── apply/remove ─────────────────────────────────────────────────────
    def apply(self, network) -> None:
        """Wrap every peer's `.query()` to observe responses, and install
        this defense on `network._sfa_defense` -- the routing-bypass hook
        slot network_sim.MockRAGNetwork / live_network.LiveRAGNetwork's
        `topic_aware_query()` already consults every hop (see module
        docstring point 2 for why this is intentional reuse, not a
        misnomer)."""
        self._num_peers = sum(1 for p in network.peers if p is not None)
        for pid, peer in enumerate(network.peers):
            if peer is None:
                continue
            original_fn = peer.query
            self._original_query_fns[pid] = original_fn
            peer.query = self._make_wrapper(pid, original_fn)
        network._sfa_defense = self

    def _make_wrapper(self, peer_id: int, orig):
        defense = self

        def _wrapped_query(question, query_confidence_threshold=0.5, *args, **kwargs):
            if defense.is_peer_blacklisted(peer_id):
                defense.record_bypass()
                return None, None, 0.0, False
            result = orig(question, query_confidence_threshold, *args, **kwargs)
            responded = result[0] is not None
            defense.record_peer_interaction(peer_id, responded)
            return result

        return _wrapped_query

    def remove(self, network) -> None:
        for pid, original_fn in self._original_query_fns.items():
            peer = network.peers[pid]
            if peer is not None:
                peer.query = original_fn
        self._original_query_fns.clear()
        if getattr(network, "_sfa_defense", None) is self:
            network._sfa_defense = None

    # ── stats ────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict[str, Any]:
        return {
            "blacklisted_peers": sorted(self.blacklisted_peers),
            "blacklisted_count": len(self.blacklisted_peers),
            "total_bypasses": self._total_bypasses,
            "total_blacklistings": self._total_blacklistings,
            "total_recoveries": self._total_recoveries,
            "redundant_probes": self._redundant_probes,
            "redundant_probe_hits": self._redundant_hits,
            "peer_reputations": {pid: round(rep, 4) for pid, rep in self._reputation.items()},
        }
