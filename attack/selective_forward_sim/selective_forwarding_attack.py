"""
attack/selective_forward_sim/selective_forwarding_attack.py

Selective Forwarding Attack (SFA): compromises a subset of peers in a
network_sim.MockRAGNetwork or live_network.LiveRAGNetwork by
monkey-patching their `.query()` method to always report a miss.

How this differs from node removal: a removed peer is dropped from
`network.peers` (set to None) and disappears from the overlay graph, so
routing and health checks both see it's gone. A selective-forwarding
peer stays intact and reachable -- its object is untouched, it's still
listed as a neighbour in `network.network`, and it would pass any
liveness/health check -- only its query *response* is suppressed. That
silence is indistinguishable, at the routing layer, from an honest peer
that simply has no knowledge relevant to this particular query: no
exception, no timeout, just `(None, None, 0.0, False)`. Each visit
still consumes one TTL hop, so with enough compromised peers -- especially
high-degree ones on a Barabasi-Albert graph, which see disproportionate
BFS traffic -- queries exhaust their hop budget before ever reaching a
peer with real relevant knowledge. See defense/sfa_sim_defense for the
countermeasure (reputation tracking -> blacklist -> routing bypass).

Drop rate: `drop_rate=1.0` (default) is a black-hole -- a compromised
peer always drops, which a fixed-threshold defense catches trivially
(response rate near 0%, nowhere close to any reasonable
`blacklist_threshold`). `drop_rate="stealthy"` is the harder,
adversarially-realistic case: each compromised peer independently drops
with a rate drawn once from Uniform(STEALTHY_LO, STEALTHY_HI) -- a
peer that still answers 70-90% of the time, tuned to sit below a naive
absolute response-rate threshold while still degrading the network over
many queries. This mirrors attack/selective_forward's
SelectiveForwardingAttack.STEALTHY_LO/HI exactly, so results are
comparable across both modules.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class SelectiveForwardingAttack:
    """
    Parameters
    ----------
    attack_ratio : fraction of peers to compromise (0.0-1.0)
    drop_rate    : float in [0.0, 1.0], or the string "stealthy"
                   float     -> every compromised peer drops at exactly this rate
                   "stealthy"-> each compromised peer independently drops at a rate
                                drawn from Uniform(STEALTHY_LO, STEALTHY_HI)
    seed         : RNG seed for reproducible target selection and drop-rate sampling
    """

    STEALTHY_LO = 0.10
    STEALTHY_HI = 0.30

    def __init__(self, attack_ratio: float = 0.3, drop_rate: Union[float, str] = 1.0, seed: int = 42):
        self.attack_ratio = attack_ratio
        self.drop_rate = drop_rate
        self.seed = seed
        self._rng = random.Random(seed)

        # peer_id -> original bound query method (restored by revert())
        self._patched_peers: Dict[int, Any] = {}
        self._per_peer_drop_rate: Dict[int, float] = {}
        self.compromised_ids: Set[int] = set()
        self._dropped_queries: int = 0

    def _resolve_drop_rate(self) -> float:
        if self.drop_rate == "stealthy":
            return float(self._rng.uniform(self.STEALTHY_LO, self.STEALTHY_HI))
        return float(self.drop_rate)

    def select_targets(self, rag_network, strategy: str = "random") -> List[int]:
        num_peers = rag_network.num_peers
        num_compromise = max(1, int(num_peers * self.attack_ratio))

        if strategy == "high_connectivity":
            # Sort peers by overlay-graph degree (descending). On a BA
            # graph a handful of hub peers carry disproportionate BFS
            # traffic, so compromising them wastes the most hop budget
            # per compromised peer -- the adversarially optimal choice.
            # On the live 3-node fully-connected network every peer has
            # equal degree, so this falls back to on-chain reliability
            # score (see live_network.LiveRAGNetwork.onchain_reliability)
            # when available: the most-trusted real source does the most
            # damage if compromised.
            onchain = getattr(rag_network, "onchain_reliability", None)
            if onchain:
                node_ids = {i: rag_network.peers[i].node_id for i in range(num_peers)}
                sorted_peers = sorted(
                    range(num_peers), key=lambda p: onchain.get(node_ids[p], 0), reverse=True
                )
            else:
                degree_map = dict(rag_network.network.degree())
                sorted_peers = sorted(degree_map.keys(), key=lambda p: degree_map[p], reverse=True)
            return sorted_peers[:num_compromise]

        candidates = list(range(num_peers))
        self._rng.shuffle(candidates)
        return candidates[: min(num_compromise, num_peers)]

    def apply(self, rag_network, strategy: str = "random") -> Dict[str, Any]:
        """Monkey-patch the selected peers' `.query()` to a silent drop."""
        if self._patched_peers:
            return {}  # already applied; call revert() first

        targets = self.select_targets(rag_network, strategy)
        self.compromised_ids = set(targets)
        self._dropped_queries = 0
        degree_map = dict(rag_network.network.degree())

        for peer_id in targets:
            peer = rag_network.peers[peer_id]
            if peer is None:
                continue  # skip a genuinely removed node
            original_query = peer.query
            self._patched_peers[peer_id] = original_query
            rate = self._resolve_drop_rate()
            self._per_peer_drop_rate[peer_id] = rate
            attack_ref = self

            def _partial_drop(question, query_confidence_threshold=0.5, *args,
                               _pid=peer_id, _rate=rate, _orig=original_query,
                               _atk=attack_ref, **kwargs):
                if _atk._rng.random() < _rate:
                    _atk._dropped_queries += 1
                    logger.debug("[SFA] peer %s silently drops query: %r", _pid, str(question)[:60])
                    return None, None, 0.0, False
                return _orig(question, query_confidence_threshold, *args, **kwargs)

            peer.query = _partial_drop

        return {
            "attack": "selective_forwarding",
            "strategy": strategy,
            "attack_ratio": self.attack_ratio,
            "drop_rate": self.drop_rate,
            "num_compromised": len(self._patched_peers),
            "compromised_ids": sorted(self._patched_peers.keys()),
            "compromised_node_ids": [rag_network.peers[p].node_id for p in self._patched_peers],
            "compromised_degrees": {p: degree_map.get(p, 0) for p in self._patched_peers},
            "per_peer_drop_rates": {p: round(r, 3) for p, r in self._per_peer_drop_rate.items()},
        }

    def revert(self, rag_network) -> None:
        """Restore original `.query()` on every compromised peer."""
        for peer_id, original_query in self._patched_peers.items():
            peer = rag_network.peers[peer_id]
            if peer is not None:
                peer.query = original_query
        self._patched_peers.clear()
        self._per_peer_drop_rate.clear()
        self.compromised_ids.clear()

    def collect_metrics(self, rag_answers: list, max_ttl: int) -> Dict[str, Any]:
        total = len(rag_answers)
        if total == 0:
            return {
                "hit_rate": 0.0,
                "avg_hops_per_query": 0.0,
                "ttl_exhaustion_rate": 0.0,
                "dropped_queries": self._dropped_queries,
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
            "dropped_queries": self._dropped_queries,
            "total_queries": total,
            "answered_queries": answered,
            "exhausted_queries": exhausted,
        }


def apply_selective_forwarding(
    rag_network, attack_ratio: float = 0.3, strategy: str = "random",
    drop_rate: Union[float, str] = 1.0, seed: int = 42,
) -> Tuple[SelectiveForwardingAttack, Dict[str, Any]]:
    """One-shot convenience helper: build + apply in a single call."""
    attack = SelectiveForwardingAttack(attack_ratio=attack_ratio, drop_rate=drop_rate, seed=seed)
    info = attack.apply(rag_network, strategy=strategy)
    return attack, info
