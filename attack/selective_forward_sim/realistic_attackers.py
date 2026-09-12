"""
attack/selective_forward_sim/realistic_attackers.py

Realistic-attacker extensions to SelectiveForwardingAttack, responding to
the NAACL improvement proposal's Selective Forwarding Attack section:
"Test realistic attackers: partial drops, delayed responses, changing drop
rates, colluding peers, and attackers that adapt to the reputation/
blacklist threshold." (See reports/updated_reports_safin/ for the broader
writeup this belongs to.)

Why a subclass in a new file, not edits to selective_forwarding_attack.py
---------------------------------------------------------------------------
`SelectiveForwardingAttack` is the subject of an 866-line, multi-revision
security analysis report (reports/SFA_Security_Analysis_Report.md) with
live-validated numbers behind every claim in it. Editing that class in
place would put every one of those already-validated figures at risk of
silently changing. `AdvancedSelectiveForwardingAttack` below subclasses it
and overrides only `apply()` (plus adds new, purely additive parameters);
`SelectiveForwardingAttack` itself, and every number reports/
SFA_Security_Analysis_Report.md already reports, is completely untouched.
This mirrors how attack/Mia_attack/mia_attack.py's diagnostic variants
(`_decision_match_adaptive`, `_decision_match_semantic`,
`_consistency_score`) were added alongside, never in place of, the
production `_decision_match`.

What "partial drops" already covers, and what's new here
--------------------------------------------------------
The base class already supports a *fixed* per-peer drop rate (a float, or
"stealthy" -- a rate drawn once from Uniform(0.10, 0.30) and then constant
for the whole run). That is a partial (non-black-hole) drop, but it is
STATIC: one number, chosen once, unaffected by anything that happens
during the run. What the proposal asks for beyond that -- and what this
module adds -- are FOUR independent, composable capabilities a static
per-peer rate cannot express:

  1. delay_range      -- a compromised peer that still eventually answers
                          (or still eventually drops), but only after an
                          injected delay -- a distinct failure mode from
                          silence, and one a naive "did it respond at all"
                          check doesn't catch as damage (see
                          `run_realistic_attack_eval.py`'s latency metric).
  2. drift             -- a peer's drop rate is not fixed for the run; it
                          oscillates over the query sequence (e.g. an
                          attacker probing for when detection windows
                          reset, or simply non-stationary behavior a
                          single-constant-rate model can't represent).
  3. collusion          -- compromised peers coordinate via a SHARED
                          round-robin counter so that, for any given
                          query, only ONE compromised peer is "on duty" to
                          drop it. This spreads the same aggregate damage
                          across the whole compromised set instead of
                          concentrating it, so EACH individual peer's own
                          raw response rate stays much closer to 1.0 than
                          an independently-acting peer's would at the same
                          effective aggregate drop rate -- directly
                          targeting a PER-PEER threshold defense's blind
                          spot (see `defense/sfa_sim_defense/
                          selective_forwarding_defense.py`'s own docstring
                          on why a fixed per-peer threshold is structurally
                          blind to certain attacker shapes).
  4. adaptive_to_threshold -- each compromised peer tracks its own
                          cumulative (responses / queries_seen) -- the
                          EXACT quantity `record_peer_interaction()`
                          computes as `raw_rate` and compares against
                          `blacklist_threshold` (defense module, "threshold"
                          detection mode) -- and self-throttles: drops
                          aggressively while comfortably above
                          threshold+margin, backs off as it approaches the
                          line. This assumes the attacker knows the
                          defense's configured threshold (not its live
                          internal state) -- the same "full knowledge of
                          system parameters, no knowledge of defender's
                          private observations" worst-case framing already
                          used by this module's `high_connectivity`
                          targeting strategy (see selective_forwarding_
                          attack.py's threat-model table).

All four are independently toggleable and composable (e.g., collusion +
adaptive_to_threshold together is a realistic strong-attacker
configuration). None is active unless explicitly requested -- constructing
`AdvancedSelectiveForwardingAttack` with no new arguments reproduces
`SelectiveForwardingAttack` exactly (verified in
test_realistic_attackers.py).
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .selective_forwarding_attack import SelectiveForwardingAttack

logger = logging.getLogger(__name__)


class AdvancedSelectiveForwardingAttack(SelectiveForwardingAttack):
    """
    Parameters (all new ones default to inactive/off -- see class docstring)
    ----------------------------------------------------------------------
    delay_range : (lo, hi) seconds -- if set, every compromised-peer
        response (drop or pass-through) sleeps `Uniform(lo, hi)` seconds
        before returning. `None` (default): no injected delay, identical
        timing to the base class.
    drift : {"period_queries": int, "amplitude": float} -- if set, a
        peer's effective drop probability at its k-th query is
        `base_rate + amplitude * sin(2*pi*k/period_queries)`, clamped to
        [0, 1]. `base_rate` is whatever `_resolve_drop_rate()` would have
        returned (so drift composes with `drop_rate="stealthy"` -- a
        drifting stealthy attacker). `None` (default): fixed rate, same
        as the base class.
    collusion : bool -- if True, compromised peers share one query
        counter; only `counter % num_compromised == this_peer's_rotation_
        slot` is "on duty" to apply the drop roll for a given query,
        everyone else passes it straight to the real peer. Distributes
        aggregate damage across the compromised set instead of each peer
        independently dropping at the full rate. `False` (default):
        every compromised peer decides independently, same as the base
        class.
    adaptive_to_threshold : {"blacklist_threshold": float,
        "safety_margin": float} -- if set, overrides drop_rate/drift for
        that peer with an online controller: maintain cumulative raw
        response rate at `blacklist_threshold + safety_margin` by
        dropping when currently above that line, responding when at or
        below it. `None` (default): rate is NOT self-throttled, same as
        the base class. Composes with `collusion` (the on-duty peer for a
        given query still runs its own local adaptive controller).
    """

    def __init__(
        self,
        attack_ratio: float = 0.3,
        drop_rate: Union[float, str] = 1.0,
        seed: int = 42,
        delay_range: Optional[Tuple[float, float]] = None,
        drift: Optional[Dict[str, float]] = None,
        collusion: bool = False,
        adaptive_to_threshold: Optional[Dict[str, float]] = None,
    ):
        super().__init__(attack_ratio=attack_ratio, drop_rate=drop_rate, seed=seed)
        self.delay_range = delay_range
        self.drift = drift
        self.collusion = collusion
        self.adaptive_to_threshold = adaptive_to_threshold

        # Diagnostics, populated during apply()/queries -- not present on
        # the base class, additive only.
        self.injected_delays: List[float] = []
        self._collusion_counter: List[int] = [0]  # shared mutable cell across all patched peers
        self._peer_query_index: Dict[int, int] = {}     # for drift: k-th query seen by this peer
        self._peer_local_stats: Dict[int, Dict[str, int]] = {}  # for adaptive: {queries, responses}

    def apply(self, rag_network, strategy: str = "random") -> Dict[str, Any]:
        if self._patched_peers:
            return {}  # already applied; call revert() first

        targets = self.select_targets(rag_network, strategy)
        self.compromised_ids = set(targets)
        self._dropped_queries = 0
        self._collusion_counter[0] = 0
        degree_map = dict(rag_network.network.degree())
        num_compromised = len(targets)

        for rotation_slot, peer_id in enumerate(targets):
            peer = rag_network.peers[peer_id]
            if peer is None:
                continue
            original_query = peer.query
            self._patched_peers[peer_id] = original_query
            base_rate = self._resolve_drop_rate()
            self._per_peer_drop_rate[peer_id] = base_rate
            self._peer_query_index[peer_id] = 0
            self._peer_local_stats[peer_id] = {"queries": 0, "responses": 0}
            attack_ref = self

            def _advanced_drop(
                question, query_confidence_threshold=0.5, *args,
                _pid=peer_id, _base_rate=base_rate, _orig=original_query,
                _atk=attack_ref, _rotation_slot=rotation_slot,
                _num_compromised=num_compromised, **kwargs,
            ):
                # 1. Determine whether this call is even "on duty" to drop
                #    (collusion) -- an off-duty compromised peer behaves
                #    exactly like an honest peer for this one query.
                on_duty = True
                if _atk.collusion and _num_compromised > 1:
                    idx = _atk._collusion_counter[0]
                    _atk._collusion_counter[0] += 1
                    on_duty = (idx % _num_compromised) == _rotation_slot

                # 2. Resolve this call's effective drop probability.
                if on_duty:
                    if _atk.adaptive_to_threshold is not None:
                        stats = _atk._peer_local_stats[_pid]
                        n = stats["queries"]
                        threshold = float(_atk.adaptive_to_threshold.get("blacklist_threshold", 0.05))
                        margin = float(_atk.adaptive_to_threshold.get("safety_margin", 0.05))
                        target_rate = min(1.0, threshold + margin)
                        current_rate = (stats["responses"] / n) if n > 0 else 1.0
                        # Drop iff staying comfortably above the target line
                        # even after this one more interaction counts against
                        # it either way -- a simple, deterministic bang-bang
                        # controller, not a probabilistic rate (an adaptive
                        # attacker doesn't need to gamble once it can compute
                        # the exact consequence of each choice).
                        would_be_rate_if_respond = (stats["responses"] + 1) / (n + 1)
                        would_be_rate_if_drop = stats["responses"] / (n + 1)
                        effective_drop_prob = 1.0 if (
                            current_rate > target_rate and would_be_rate_if_drop >= threshold
                        ) else 0.0
                    elif _atk.drift is not None:
                        k = _atk._peer_query_index[_pid]
                        _atk._peer_query_index[_pid] += 1
                        period = max(1, int(_atk.drift.get("period_queries", 50)))
                        amp = float(_atk.drift.get("amplitude", 0.2))
                        effective_drop_prob = _base_rate + amp * math.sin(2 * math.pi * k / period)
                        effective_drop_prob = max(0.0, min(1.0, effective_drop_prob))
                    else:
                        effective_drop_prob = _base_rate
                else:
                    effective_drop_prob = 0.0  # off-duty this round: never drops

                will_drop = _atk._rng.random() < effective_drop_prob

                # 3. Injected delay, applied regardless of the eventual
                #    drop/respond decision -- a delayed-but-eventually-
                #    correct response is itself a distinct, measurable cost.
                if _atk.delay_range is not None:
                    lo, hi = _atk.delay_range
                    delay = _atk._rng.uniform(lo, hi)
                    _atk.injected_delays.append(delay)
                    time.sleep(delay)

                # 4. Update this peer's local stats -- mirrors exactly what
                #    defense/sfa_sim_defense's record_peer_interaction()
                #    tracks (queries, responses), computed independently
                #    here so the adaptive controller above needs no access
                #    to any live defense object.
                stats = _atk._peer_local_stats[_pid]
                stats["queries"] += 1
                if not will_drop:
                    stats["responses"] += 1

                if will_drop:
                    _atk._dropped_queries += 1
                    logger.debug("[SFA-advanced] peer %s drops (on_duty=%s, p=%.3f): %r",
                                 _pid, on_duty, effective_drop_prob, str(question)[:60])
                    return None, None, 0.0, False
                return _orig(question, query_confidence_threshold, *args, **kwargs)

            peer.query = _advanced_drop

        return {
            "attack": "selective_forwarding_advanced",
            "strategy": strategy,
            "attack_ratio": self.attack_ratio,
            "drop_rate": self.drop_rate,
            "delay_range": self.delay_range,
            "drift": self.drift,
            "collusion": self.collusion,
            "adaptive_to_threshold": self.adaptive_to_threshold,
            "num_compromised": len(self._patched_peers),
            "compromised_ids": sorted(self._patched_peers.keys()),
            "compromised_node_ids": [rag_network.peers[p].node_id for p in self._patched_peers],
            "compromised_degrees": {p: degree_map.get(p, 0) for p in self._patched_peers},
            "per_peer_drop_rates": {p: round(r, 3) for p, r in self._per_peer_drop_rate.items()},
        }

    def per_peer_final_response_rate(self) -> Dict[int, float]:
        """Diagnostic: each compromised peer's actual cumulative response
        rate at the end of the run -- e.g. to verify an adaptive attacker
        really did converge near its target line (see
        test_realistic_attackers.py) or that collusion really did spread
        load thinner per peer than independent dropping would have."""
        return {
            pid: (s["responses"] / s["queries"] if s["queries"] else 1.0)
            for pid, s in self._peer_local_stats.items()
        }
