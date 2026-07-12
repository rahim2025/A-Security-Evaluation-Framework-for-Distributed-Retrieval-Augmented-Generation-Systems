"""
attack/selective_forward_sim/network_sim.py

In-process simulation of a DRAG-style P2P overlay: a Barabasi-Albert
scale-free graph of MockPeer nodes queried via a TTL-bounded,
topic-aware-random-walk-style BFS. This lets SelectiveForwardingAttack
be swept across network sizes/ratios/strategies in seconds, with no
Docker or blockchain required.

For the real deployment (3 live drag_data_source containers + the
on-chain DragScores ledger), see live_network.py -- it exposes the same
`.peers` / `.network` (networkx graph) / `.topic_aware_query()` /
`._sfa_defense` surface, so SelectiveForwardingAttack and
SelectiveForwardingDefense drive both without any mode-specific code.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import networkx as nx


@dataclass
class QueryResult:
    question: str
    answer: Optional[str]
    is_query_hit: bool
    num_hops: int
    routing_log: List[str] = field(default_factory=list)


class MockPeer:
    """A peer with a synthetic knowledge base: answers with probability
    `hit_prob` on any given query (independent Bernoulli hit model)."""

    def __init__(self, peer_id: int, hit_prob: float, rng: random.Random):
        self.peer_id = peer_id
        self.node_id = f"peer_{peer_id}"
        self.hit_prob = hit_prob
        self._rng = rng

    def query(
        self, question: str, query_confidence_threshold: float = 0.5
    ) -> Tuple[Optional[str], Optional[str], float, bool]:
        if self._rng.random() < self.hit_prob:
            score = self._rng.uniform(query_confidence_threshold, 1.0)
            return f"answer[{self.peer_id}]", f"knowledge[{self.peer_id}]", score, True
        return None, None, 0.0, False


class MockRAGNetwork:
    """
    Barabasi-Albert overlay of MockPeer nodes with TTL-bounded BFS query
    routing. `topic_aware_query()` starts from a (random, by default)
    peer and expands breadth-first through unvisited neighbours, up to
    `num_query_neighbor` per hop, until either a peer reports a hit
    above `query_confidence_threshold` or `query_ttl` hops are spent.

    If a SelectiveForwardingDefense has been installed via
    `defense.apply(network)`, blacklisted peers are bypassed: their
    neighbours are expanded directly, at the same hop depth, instead of
    spending a TTL hop calling a peer already known to be compromised.
    """

    def __init__(
        self,
        num_peers: int = 20,
        num_attachments: int = 4,
        num_query_neighbor: int = 4,
        query_ttl: int = 6,
        peer_hit_prob: float = 0.4,
        seed: int = 0,
    ):
        self.num_peers = num_peers
        self.num_query_neighbor = num_query_neighbor
        self.query_ttl = query_ttl
        self._rng = random.Random(seed)

        self.network = nx.barabasi_albert_graph(num_peers, num_attachments, seed=seed)
        self.peers: List[Optional[MockPeer]] = [
            MockPeer(pid, peer_hit_prob, self._rng) for pid in range(num_peers)
        ]
        # Installed by SelectiveForwardingDefense.apply(); left None (defense
        # inactive) otherwise. See defense/sfa_sim_defense/selective_forwarding_defense.py.
        self._sfa_defense = None

    def topic_aware_query(
        self,
        question: str,
        query_confidence_threshold: float = 0.5,
        start: Optional[int] = None,
    ) -> QueryResult:
        start_id = start if start is not None else self._rng.randrange(self.num_peers)
        visited = {start_id}   # discovered/enqueued -- controls BFS traversal, NOT "actually queried"
        queried: set = set()   # peers whose .query() was actually invoked in phase 1
        queue = deque([start_id])
        log: List[str] = []
        defense = self._sfa_defense
        hops = 0

        while queue and hops < self.query_ttl:
            current_id = queue.popleft()
            peer = self.peers[current_id]
            if peer is None:
                continue

            if defense is not None and defense.is_peer_blacklisted(current_id):
                defense.record_bypass()
                log.append(f"peer_{current_id}: BYPASS[blacklisted]")
                neighbours = [n for n in self.network.neighbors(current_id) if n not in visited]
                for nid in neighbours[: self.num_query_neighbor]:
                    visited.add(nid)
                    queue.append(nid)
                continue  # no TTL cost for a bypassed peer

            hops += 1
            queried.add(current_id)
            answer, knowledge, score, is_hit = peer.query(question, query_confidence_threshold)
            log.append(f"peer_{current_id}: {'HIT' if is_hit else 'MISS'} hop={hops}")
            if is_hit and score >= query_confidence_threshold:
                return QueryResult(question, answer, True, hops, log)

            neighbours = [n for n in self.network.neighbors(current_id) if n not in visited]
            for nid in neighbours[: self.num_query_neighbor]:
                visited.add(nid)
                queue.append(nid)

        # Phase 2 -- bounded redundant backup probe (defense only). The
        # primary TTL-bounded BFS above ran out of hop budget without a hit.
        # If a defense is installed, it gets up to `defense.redundancy_k`
        # extra attempts against peers it trusts and hasn't already tried --
        # this is what lets correct blacklisting actually translate into a
        # recovered answer once the hop budget alone is too tight to route
        # around a compromised peer. See
        # defense/sfa_sim_defense/selective_forwarding_defense.py.
        #
        # Deliberately excludes by `queried`, not `visited`: on a small,
        # densely-connected graph (e.g. the live 3-node deployment), a
        # single failed hop discovers -- and therefore marks `visited` --
        # every other peer as a BFS neighbour, even though `query_ttl` may
        # be too small to ever actually pop and query them. Excluding by
        # `visited` made Phase 2 see zero candidates in exactly that case,
        # silently no-opping the whole mechanism where it's needed most.
        if defense is not None and getattr(defense, "redundancy_k", 0) > 0:
            for bid in defense.backup_candidates(queried, list(range(self.num_peers))):
                peer = self.peers[bid]
                if peer is None:
                    continue
                hops += 1
                answer, knowledge, score, is_hit = peer.query(question, query_confidence_threshold)
                defense.record_redundant_probe(is_hit)
                log.append(f"peer_{bid}: REDUNDANT {'HIT' if is_hit else 'MISS'} hop={hops}")
                if is_hit and score >= query_confidence_threshold:
                    return QueryResult(question, answer, True, hops, log)

        return QueryResult(question, None, False, hops if hops else self.query_ttl, log)
