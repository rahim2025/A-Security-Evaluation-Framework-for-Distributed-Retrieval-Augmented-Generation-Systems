"""
selective_forwarding_defense.py
================================
Defense against the Selective Forwarding Attack (SFA) in DRAG networks.

Attack recap
------------
SelectiveForwardingAttack monkey-patches a fraction of peers so their
query() always returns (None, None, 0.0, False).  The node stays in the
overlay graph and passes health checks — it just silently drops every
query routed through it.  High-connectivity variants target hub nodes to
maximally disrupt BFS routing, causing TTL exhaustion and hit-rate collapse.

Defense strategy
----------------
Three layered countermeasures:

1. Reputation tracking  
   Every peer.query() call is wrapped.  A running exponential moving average
   of each peer's "response rate" (fraction of queries where it returned a
   non-None answer) is maintained as its reputation score in [0, 1].

2. Blacklisting  
   A peer whose reputation falls below `blacklist_threshold` after at least
   `min_queries_before_blacklist` interactions is added to the blacklist.
   Blacklisted peers are skipped during routing (their query() short-circuits
   to (None, None, 0.0, False) before touching the network).

3. Routing bypass / neighbor expansion  
   When the BFS is about to enqueue a blacklisted peer it skips it and
   instead enqueues that peer's neighbors directly — effectively routing
   *around* the dead node.  This recovers reachability without enlarging
   TTL.

Usage
-----
With MockRAGNetwork (run_selective_forwarding.py):

    from modules.defenses.selective_forwarding_defense import SelectiveForwardingDefense

    defense = SelectiveForwardingDefense({
        'blacklist_threshold': 0.2,
        'min_queries_before_blacklist': 5,
        'suspicion_threshold': 0.35,
        'reputation_decay': 0.85,
    })
    defense.apply(network)          # patches peer.query() and installs bypass hook

    # ... run queries as normal ...

    defense.remove(network)         # restores originals
    print(defense.get_stats())

With full RAGNetwork (modules/rag_network.py):
    Same apply() / remove() interface — the defense wraps peer.query()
    at the peer level, independent of which network class is used.
"""

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from modules.defenses.base_defense import BaseDefense


class SelectiveForwardingDefense(BaseDefense):
    """
    Reputation-based defense against Selective Forwarding Attacks.

    Monitors per-peer response behaviour, builds reputation scores, blacklists
    chronically silent nodes, and redirects BFS routing around them.
    """

    # ------------------------------------------------------------------ init

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        super().__init__("SelectiveForwardingDefense", config)

        # --- reputation / blacklist parameters ----------------------------
        # Peer is blacklisted when its response_rate < this value
        self.blacklist_threshold: float = config.get("blacklist_threshold", 0.05)
        # Minimum interactions before a peer can be blacklisted
        self.min_queries_before_blacklist: int = config.get(
            "min_queries_before_blacklist", 15
        )
        # Peer is "suspicious" (soft-flag) below this reputation
        self.suspicion_threshold: float = config.get("suspicion_threshold", 0.10)
        # EMA weight: higher = faster adaptation to recent behaviour
        self.reputation_ema_alpha: float = 1.0 - config.get("reputation_decay", 0.70)
        # min_peers used by validate()
        self.min_peers: int = config.get("min_peers_for_validation", 2)

        # --- per-peer tracking -------------------------------------------
        self._query_count: Dict[int, int] = defaultdict(int)
        self._response_count: Dict[int, int] = defaultdict(int)
        self._reputation: Dict[int, float] = defaultdict(lambda: 1.0)
        self.blacklisted_peers: Set[int] = set()

        # --- patch bookkeeping -------------------------------------------
        self._original_query_fns: Dict[int, Any] = {}
        self._patched_network: Optional[object] = None

        # --- counters for get_stats() ------------------------------------
        self._total_blacklistings: int = 0
        self._total_bypasses: int = 0     # times BFS skipped a blacklisted peer
        self._total_peer_calls: int = 0

        logger.info(
            f"SelectiveForwardingDefense initialised | "
            f"blacklist_threshold={self.blacklist_threshold} "
            f"min_queries={self.min_queries_before_blacklist} "
            f"suspicion_threshold={self.suspicion_threshold}"
        )

    # ------------------------------------------------------- reputation API

    def record_peer_interaction(self, peer_id: int, responded: bool) -> None:
        """
        Record a query interaction and update the peer's reputation score.

        Args:
            peer_id:   Peer identifier.
            responded: True if the peer returned a non-None answer.
        """
        self._query_count[peer_id] += 1
        self._total_peer_calls += 1

        if responded:
            self._response_count[peer_id] += 1

        n = self._query_count[peer_id]
        raw_rate = self._response_count[peer_id] / n

        # Exponential moving average: blend current reputation with raw_rate
        current = self._reputation[peer_id]
        alpha = self.reputation_ema_alpha
        new_rep = (1.0 - alpha) * current + alpha * raw_rate
        self._reputation[peer_id] = new_rep

        logger.debug(
            f"[SFD] peer={peer_id} responded={responded} "
            f"raw_rate={raw_rate:.2f} rep={new_rep:.3f}"
        )

        # Auto-blacklist if conditions met
        if (
            peer_id not in self.blacklisted_peers
            and n >= self.min_queries_before_blacklist
            and raw_rate < self.blacklist_threshold
        ):
            self.blacklisted_peers.add(peer_id)
            self._total_blacklistings += 1
            logger.warning(
                f"[SFD] Peer {peer_id} BLACKLISTED after {n} queries "
                f"(response_rate={raw_rate:.2f} < {self.blacklist_threshold})"
            )

    def get_peer_reputation(self, peer_id: int) -> float:
        """Return current reputation score for *peer_id* (0.0 – 1.0)."""
        return self._reputation[peer_id]

    def is_peer_blacklisted(self, peer_id: int) -> bool:
        """True if the peer has been blacklisted."""
        return peer_id in self.blacklisted_peers

    def is_peer_suspicious(self, peer_id: int) -> bool:
        """True if reputation is below the suspicion threshold."""
        return (
            peer_id in self.blacklisted_peers
            or self._reputation[peer_id] < self.suspicion_threshold
        )

    def filter_trusted_peers(self, peer_ids: List[int]) -> List[int]:
        """
        Return only non-suspicious peers from *peer_ids*.
        Falls back to non-blacklisted peers if all are suspicious,
        and to the full list as a last resort to avoid disconnection.
        """
        trusted = [p for p in peer_ids if not self.is_peer_suspicious(p)]
        if trusted:
            return trusted

        non_blacklisted = [p for p in peer_ids if p not in self.blacklisted_peers]
        if non_blacklisted:
            logger.debug("[SFD] All peers suspicious — using non-blacklisted fallback")
            return non_blacklisted

        logger.debug("[SFD] All peers blacklisted — returning all as last resort")
        return list(peer_ids)

    # --------------------------------------------------- BaseDefense.validate

    def validate(
        self,
        question: str,
        candidate_answer: str,
        peer_responses: List[Dict[str, Any]],
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Validate an answer by checking peer trust and consensus.

        Args:
            question:         The input question (unused here, present for API).
            candidate_answer: The answer to validate.
            peer_responses:   List of dicts with keys 'peer_id' and 'answer'.

        Returns:
            (is_valid, confidence, details)
        """
        if not self.enabled:
            return True, 1.0, {"reason": "Defense disabled"}

        # Record interactions from the collected peer responses
        for resp in peer_responses:
            pid = resp.get("peer_id", -1)
            if pid >= 0:
                self.record_peer_interaction(pid, responded=bool(resp.get("answer")))

        # Reject answers sourced from blacklisted peers
        candidate_source = next(
            (
                resp.get("peer_id")
                for resp in peer_responses
                if resp.get("answer") == candidate_answer
            ),
            None,
        )
        if candidate_source is not None and self.is_peer_blacklisted(candidate_source):
            self.update_stats(False, 0.05)
            return False, 0.05, {
                "reason": f"Source peer {candidate_source} is blacklisted",
                "blacklisted_peers": sorted(self.blacklisted_peers),
                "suspicious": True,
            }

        # Gather responses from trusted peers
        trusted_responses = [
            r
            for r in peer_responses
            if r.get("answer") and not self.is_peer_suspicious(r.get("peer_id", -1))
        ]

        if len(trusted_responses) < self.min_peers:
            self.update_stats(True, 0.5)
            return True, 0.5, {
                "reason": (
                    f"Insufficient trusted peers "
                    f"({len(trusted_responses)} < {self.min_peers})"
                ),
                "trusted_peer_count": len(trusted_responses),
                "validation_skipped": True,
            }

        # Simple majority agreement among trusted peers
        trusted_answers = [r["answer"] for r in trusted_responses]
        agreement = trusted_answers.count(candidate_answer) / len(trusted_answers)
        confidence = min(1.0, agreement + 0.1 * min(len(trusted_responses), 5) / 5)
        is_valid = agreement >= 0.5

        details = {
            "reason": "Validated by trusted peer consensus",
            "agreement": agreement,
            "trusted_peer_count": len(trusted_responses),
            "blacklisted_peers": sorted(self.blacklisted_peers),
        }

        self.update_stats(is_valid, confidence)
        return is_valid, confidence, details

    # ---------------------------------------- network-level apply / remove

    def apply(self, network) -> None:
        """
        Attach this defense to a RAGNetwork (or MockRAGNetwork).

        Wraps every peer's query() so interactions are tracked, and
        blacklisted peers short-circuit immediately without touching the LLM.
        Also installs a bypass hook on the network itself so the BFS can
        route around blacklisted peers.

        Args:
            network: Any network object with a ``peers`` list attribute
                     where each entry has a callable ``query`` method.
        """
        if not hasattr(network, "peers"):
            logger.warning("[SFD] Network has no 'peers' attribute; apply() skipped")
            return

        self._original_query_fns.clear()
        defense = self  # capture for closures

        for pid, peer in enumerate(network.peers):
            if peer is None:
                continue

            original_fn = peer.query
            self._original_query_fns[pid] = original_fn

            def _make_wrapper(peer_id, orig):
                def _wrapped_query(question, query_confidence_threshold):
                    # Short-circuit for blacklisted peers — no wasted LLM call
                    if defense.is_peer_blacklisted(peer_id):
                        defense._total_bypasses += 1
                        logger.debug(
                            f"[SFD] Skipping blacklisted peer {peer_id}"
                        )
                        return None, None, 0.0, False

                    result = orig(question, query_confidence_threshold)
                    responded = result[0] is not None
                    defense.record_peer_interaction(peer_id, responded)
                    return result

                return _wrapped_query

            peer.query = _make_wrapper(pid, original_fn)

        # Install neighbor-bypass hook so BFS can call it
        network._sfa_defense = self
        self._patched_network = network

        logger.info(
            f"[SFD] Defense applied to {len(self._original_query_fns)} peers"
        )

    def remove(self, network=None) -> None:
        """
        Remove the defense and restore original peer.query() functions.

        Args:
            network: Network to restore.  If None, uses the last patched network.
        """
        target = network or self._patched_network
        if target is None:
            logger.warning("[SFD] remove() called but no network to restore")
            return

        for pid, orig_fn in self._original_query_fns.items():
            if pid < len(target.peers) and target.peers[pid] is not None:
                target.peers[pid].query = orig_fn

        if hasattr(target, "_sfa_defense"):
            del target._sfa_defense

        self._original_query_fns.clear()
        self._patched_network = None
        logger.info("[SFD] Defense removed from network; original queries restored")

    def reset(self) -> None:
        """Reset all reputation and blacklist state (keeps parameters intact)."""
        self._query_count.clear()
        self._response_count.clear()
        self._reputation.clear()
        self.blacklisted_peers.clear()
        self._total_blacklistings = 0
        self._total_bypasses = 0
        self._total_peer_calls = 0
        self.reset_stats()
        logger.info("[SFD] State reset")

    # ----------------------------------------------------------- statistics

    def get_stats(self) -> Dict[str, Any]:
        """Return full defense statistics."""
        base = super().get_stats()
        suspicious_count = sum(
            1
            for pid in self._reputation
            if self._reputation[pid] < self.suspicion_threshold
            and pid not in self.blacklisted_peers
        )
        return {
            **base,
            "blacklisted_peers": sorted(self.blacklisted_peers),
            "blacklisted_count": len(self.blacklisted_peers),
            "suspicious_count": suspicious_count,
            "total_peer_calls_tracked": self._total_peer_calls,
            "total_bypasses": self._total_bypasses,
            "total_blacklistings": self._total_blacklistings,
            "peer_reputations": {
                pid: round(rep, 4)
                for pid, rep in sorted(self._reputation.items())
            },
        }
