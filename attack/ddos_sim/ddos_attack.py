"""
attack/ddos_sim/ddos_attack.py

Congestion-based (application-layer) DDoS simulation against a DRAG-style
P2P overlay, built on top of the same network abstraction already used by
attack/selective_forward_sim: network_sim.MockRAGNetwork (in-process
Barabasi-Albert graph, no Docker/blockchain required) and
live_network.LiveRAGNetwork (the real drag_data_source Docker containers +
on-chain DragScores reliability reads). See reports/SFA_Security_Analysis_Report.md
for the full design writeup this module implements.

Design invariant: nodes are never removed from `network.peers` or
`network.network`. An attacked peer stays reachable -- it degrades
gracefully (drops some fraction of queries, adds simulated hop/message
overhead to the rest) and recovers automatically once its wave's duration
elapses. This mirrors how a real DDoS victim stays "up" but becomes unable
to serve requests, and is the same node-removal-vs-degradation distinction
selective_forwarding_attack.py draws for silent dropping.

Not packet-level. No real network traffic is generated or measured; this
is an app-layer abstraction for studying RAG-level availability metrics
(hit rate, hops, availability %), not a network-security packet simulator.

Known limitation -- shared RNG stream across mock peers: MockRAGNetwork
(attack/selective_forward_sim/network_sim.py) passes a single random.Random
instance to every MockPeer. Because this attack's wrapper drops some
queries before they ever reach a peer's real .query() (see "Query-time
interception" below), the number of RNG draws consumed from that shared
stream differs between an attacked run and its baseline, shifting
downstream peers' random outcomes in a path-dependent way. This adds noise,
not bias -- it's visible in this module's own hit_rate not decaying
monotonically with availability_percentage across some configurations --
and is a deliberate, documented caveat rather than a fix, since giving each
peer an independent RNG would change the shared simulation's reproducible
output for selective_forward_sim as well (see problems/ddos_attack_gaps.md #5).

Wave lifecycle (execute one wave via `run_wave()`)
---------------------------------------------------
1. Recover -- any peer whose recovery timer (tracked in wave units, see
   `wave_interval_s` below) has expired is cleared back to healthy before
   this wave's targeting runs.
2. Select targets -- `num_to_attack = max(1, floor(num_peers * attack_ratio))`
   peers chosen by `random` | `targeted` | `sequential` (or an explicit
   `target_peers` list, which overrides strategy selection every wave).
3. Assign congestion -- sample intensity in [intensity_min, intensity_max],
   derive `drop_probability = min(0.95, intensity * 0.8)` and
   `load_penalty = min(0.90, intensity * 0.9)`. Re-attacking an
   already-overloaded peer keeps the higher of the old/new intensity
   (worst-case accumulation).
4. Cascade -- spill `intensity * cascade_factor` (capped at
   `max_cascade_intensity`) onto each target's graph neighbours, so damage
   isn't confined to directly-hit peers.

Recovery timing is expressed in simulated wave units, not wall-clock
sleeps: `wave_interval_s` is the assumed real-world seconds each wave/query
-batch represents, so `ddos_duration` (seconds) converts to
`waves_to_recover = max(1, round(ddos_duration / wave_interval_s))`. This
reproduces the report's finding -- a long recovery window relative to wave
cadence means the network never gets a chance to recover between waves --
without requiring an actual multi-minute sleep per run.

Query-time interception
------------------------
`attach(network)` installs a dynamic-lookup wrapper on every peer's
`.query()` (same monkey-patch mechanism as SelectiveForwardingAttack /
SelectiveForwardingDefense): on each call it looks up the peer's current
`OverloadState` in `self.overload_table` and rolls a seeded Bernoulli trial
against `drop_probability`. A drop returns `(None, None, 0.0, False)` --
scored as a fully failed retrieval by the same BFS routing loop that
selective_forward_sim already drives, so the attack is measurable in the
evaluation metrics without any routing-loop changes. A non-dropped call to
a congested peer still accrues `extra_hops`/`extra_msgs` overhead counters
(queuing/retry cost of a congested-but-still-responding node) even though
it doesn't inflate the BFS's actual hop count -- an explicit, documented
simplification (see module docstring "Not packet-level" above).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class OverloadState:
    intensity: float
    drop_probability: float
    load_penalty: float
    extra_hops: int
    extra_msgs: int
    recovery_wave: int


class DDoSAttack:
    """
    Parameters
    ----------
    attack_ratio         : fraction of peers targeted per wave (0.0-1.0)
    iterations            : number of waves `run_wave()` will be called for
                             (informational -- callers drive the actual loop)
    strategy               : "random" | "targeted" | "sequential"
    target_peers           : explicit peer index list; overrides `strategy` every
                              wave when non-empty
    ddos_duration           : seconds until an attacked/cascaded peer auto-recovers
    wave_interval_s         : assumed real-world seconds one wave represents,
                               used to convert `ddos_duration` into whole waves
    intensity_min/max       : uniform sampling bounds for per-peer attack intensity
    cascade_factor          : fraction of a direct target's intensity spilled onto
                               its graph neighbours
    max_cascade_intensity   : cap on cascaded (indirect) intensity, so cascade
                               victims are always less damaged than direct targets
    seed                    : RNG seed for reproducible targeting/intensity/drops
    """

    #: A peer is classified "effectively down" once drop_probability reaches
    #: this SLA-style threshold (it would fail the majority of requests).
    DOWN_THRESHOLD = 0.5

    def __init__(
        self,
        attack_ratio: float = 0.3,
        iterations: int = 5,
        strategy: str = "random",
        target_peers: Optional[List[int]] = None,
        ddos_duration: float = 60.0,
        wave_interval_s: float = 30.0,
        intensity_min: float = 0.5,
        intensity_max: float = 1.0,
        cascade_factor: float = 0.25,
        max_cascade_intensity: float = 0.6,
        seed: int = 42,
    ):
        self.attack_ratio = attack_ratio
        self.iterations = iterations
        self.strategy = strategy
        self.target_peers = list(target_peers) if target_peers else []
        self.ddos_duration = ddos_duration
        self.wave_interval_s = wave_interval_s
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.cascade_factor = cascade_factor
        self.max_cascade_intensity = max_cascade_intensity
        self.seed = seed
        self._rng = random.Random(seed)

        self.waves_to_recover = max(1, round(ddos_duration / wave_interval_s))
        self.overload_table: Dict[int, OverloadState] = {}
        self._original_query_fns: Dict[int, Any] = {}
        self._attached_network: Any = None

        self.dropped_queries_total: int = 0
        self.congested_queries_total: int = 0
        self.extra_hops_total: int = 0
        self.extra_msgs_total: int = 0
        self.wave_log: List[Dict[str, Any]] = []

    # ── query-time interception ─────────────────────────────────────────
    def attach(self, network) -> None:
        """Install a dynamic overload-table lookup on every peer's `.query()`.
        Idempotent: call once per network, then call `run_wave()` repeatedly
        -- the wrapper always consults the live `overload_table`, so it
        doesn't need to be reinstalled between waves."""
        if self._attached_network is not None:
            return
        for pid, peer in enumerate(network.peers):
            if peer is None:
                continue
            original_query = peer.query
            self._original_query_fns[pid] = original_query
            peer.query = self._make_wrapper(pid, original_query)
        self._attached_network = network

    def detach(self, network) -> None:
        for pid, original_query in self._original_query_fns.items():
            peer = network.peers[pid]
            if peer is not None:
                peer.query = original_query
        self._original_query_fns.clear()
        self._attached_network = None

    def _make_wrapper(self, peer_id: int, orig):
        attack_ref = self

        def _intercepted(question, query_confidence_threshold=0.5, *args, **kwargs):
            state = attack_ref.overload_table.get(peer_id)
            if state is None:
                return orig(question, query_confidence_threshold, *args, **kwargs)
            if attack_ref._rng.random() < state.drop_probability:
                attack_ref.dropped_queries_total += 1
                return None, None, 0.0, False
            attack_ref.congested_queries_total += 1
            attack_ref.extra_hops_total += state.extra_hops
            attack_ref.extra_msgs_total += state.extra_msgs
            return orig(question, query_confidence_threshold, *args, **kwargs)

        return _intercepted

    # ── target selection ────────────────────────────────────────────────
    def _select_targets(self, num_peers: int, wave_num: int) -> List[int]:
        if self.target_peers:
            return [p for p in self.target_peers if 0 <= p < num_peers]

        num_to_attack = max(1, math.floor(num_peers * self.attack_ratio))

        if self.strategy == "sequential":
            start = (wave_num * num_to_attack) % num_peers
            return [(start + i) % num_peers for i in range(num_to_attack)]

        if self.strategy == "targeted":
            fresh = [p for p in range(num_peers) if p not in self.overload_table]
            self._rng.shuffle(fresh)
            if len(fresh) >= num_to_attack:
                return fresh[:num_to_attack]
            already_hit = sorted(
                (p for p in range(num_peers) if p in self.overload_table),
                key=lambda p: self.overload_table[p].intensity,
            )
            return fresh + already_hit[: num_to_attack - len(fresh)]

        # random (default): uniform sample, undirected attacker
        candidates = list(range(num_peers))
        self._rng.shuffle(candidates)
        return candidates[: min(num_to_attack, num_peers)]

    # ── wave lifecycle ──────────────────────────────────────────────────
    def _recover_expired(self, wave_num: int) -> List[int]:
        expired = [pid for pid, s in self.overload_table.items() if s.recovery_wave <= wave_num]
        for pid in expired:
            del self.overload_table[pid]
        return expired

    def _neighbours_of(self, network, pid: int, num_peers: int) -> List[int]:
        graph = getattr(network, "network", None)
        if graph is not None:
            try:
                return list(graph.neighbors(pid))
            except Exception:
                pass
        # Documented fallback for a network exposing no real adjacency:
        # an index +/-1 ring.
        return [n % num_peers for n in (pid - 1, pid + 1) if n != pid]

    def _touch_peer(self, pid: int, intensity: float, wave_num: int) -> None:
        intensity = max(0.0, min(1.0, intensity))
        existing = self.overload_table.get(pid)
        if existing is not None and existing.intensity >= intensity:
            # worst-case accumulation: keep the stronger state, just refresh the timer
            existing.recovery_wave = wave_num + self.waves_to_recover
            return
        self.overload_table[pid] = OverloadState(
            intensity=intensity,
            drop_probability=min(0.95, intensity * 0.8),
            load_penalty=min(0.90, intensity * 0.9),
            extra_hops=math.ceil(intensity * 3),
            extra_msgs=math.ceil(intensity * 5),
            recovery_wave=wave_num + self.waves_to_recover,
        )

    def run_wave(self, network, wave_num: int) -> Dict[str, Any]:
        """Execute one attack wave: recover expired peers, select this
        wave's targets, assign congestion, and cascade a capped fraction of
        it onto graph neighbours. Returns a summary dict; also appended to
        `self.wave_log`."""
        num_peers = network.num_peers
        recovered = self._recover_expired(wave_num)

        targets = self._select_targets(num_peers, wave_num)
        for pid in targets:
            intensity = self._rng.uniform(self.intensity_min, self.intensity_max)
            self._touch_peer(pid, intensity, wave_num)

        cascaded: Set[int] = set()
        for pid in targets:
            primary_intensity = self.overload_table[pid].intensity
            cascade_intensity = min(self.max_cascade_intensity, primary_intensity * self.cascade_factor)
            for nid in self._neighbours_of(network, pid, num_peers):
                if nid in targets:
                    continue
                self._touch_peer(nid, cascade_intensity, wave_num)
                cascaded.add(nid)

        snapshot = self.availability_snapshot(num_peers)
        info = {
            "wave": wave_num,
            "strategy": self.strategy if not self.target_peers else "explicit",
            "targeted": sorted(targets),
            "cascaded": sorted(cascaded),
            "recovered": sorted(recovered),
            **snapshot,
        }
        self.wave_log.append(info)
        return info

    # ── availability accounting ─────────────────────────────────────────
    def availability_snapshot(self, num_peers: int) -> Dict[str, Any]:
        down = [pid for pid, s in self.overload_table.items() if s.drop_probability >= self.DOWN_THRESHOLD]
        active = num_peers - len(down)
        avg_intensity = (
            float(np.mean([s.intensity for s in self.overload_table.values()]))
            if self.overload_table else 0.0
        )
        return {
            "availability_percentage": 100.0 * active / num_peers if num_peers else 0.0,
            "active_nodes": active,
            "total_peers": num_peers,
            "overloaded_count": len(self.overload_table),
            "down_count": len(down),
            "avg_load_intensity": avg_intensity,
        }

    # ── metrics over a batch of queries (same shape as SelectiveForwardingAttack) ──
    def collect_metrics(self, rag_answers: list, max_ttl: int) -> Dict[str, Any]:
        total = len(rag_answers)
        if total == 0:
            return {
                "hit_rate": 0.0,
                "avg_hops_per_query": 0.0,
                "ttl_exhaustion_rate": 0.0,
                "total_queries": 0,
                "answered_queries": 0,
                "exhausted_queries": 0,
            }
        answered = sum(1 for r in rag_answers if r.answer and r.is_query_hit)
        exhausted = sum(1 for r in rag_answers if r.num_hops >= max_ttl and not r.is_query_hit)
        avg_hops = float(np.mean([r.num_hops for r in rag_answers]))
        return {
            "hit_rate": answered / total,
            "avg_hops_per_query": avg_hops,
            "ttl_exhaustion_rate": exhausted / total,
            "total_queries": total,
            "answered_queries": answered,
            "exhausted_queries": exhausted,
        }


def apply_ddos_wave(
    network, wave_num: int = 0, attack_ratio: float = 0.3, strategy: str = "random", seed: int = 42,
) -> Tuple[DDoSAttack, Dict[str, Any]]:
    """One-shot convenience helper: build, attach, and run a single wave."""
    attack = DDoSAttack(attack_ratio=attack_ratio, strategy=strategy, seed=seed)
    attack.attach(network)
    info = attack.run_wave(network, wave_num)
    return attack, info
