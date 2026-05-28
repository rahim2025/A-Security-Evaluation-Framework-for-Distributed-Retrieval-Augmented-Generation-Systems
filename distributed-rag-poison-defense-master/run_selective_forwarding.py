"""
run_selective_forwarding.py
===========================
Standalone simulation of the Selective Forwarding Attack on the DRAG network.

No LLM / Ollama required — peers use a lightweight mock that simulates
knowledge-base hits probabilistically.  This lets you measure the three
primary attack metrics immediately on any machine:

  hit_rate            — fraction of queries that produced an answer
  avg_hops_per_query  — mean hops consumed per query
  ttl_exhaustion_rate — fraction of queries that burned through max_ttl

The script sweeps over:
  - compromise ratios : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
  - targeting strategy: ['random', 'high_connectivity']

Results are printed to stdout and saved to
  logs/selective_forwarding/results.csv

Usage
-----
  python run_selective_forwarding.py
  python run_selective_forwarding.py --num_peers 30 --max_ttl 8 --num_queries 200
"""

import argparse
import csv
import os
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Minimal data types (mirrors modules/data_types.py)
# ---------------------------------------------------------------------------

@dataclass
class RAGAnswer:
    answer: str
    relevant_knowledge: str
    relevant_score: float
    num_hops: int
    num_messages: int
    is_query_hit: bool


# ---------------------------------------------------------------------------
# Mock Peer
# ---------------------------------------------------------------------------

class MockPeer:
    """
    A peer that simulates a knowledge-base hit with probability `hit_prob`.
    No LLM or embeddings are needed — just a Bernoulli draw.
    """

    def __init__(self, peer_id: int, hit_prob: float = 0.4):
        self.peer_id = peer_id
        self.hit_prob = hit_prob

    def query(
        self, question: str, query_confidence_threshold: float
    ) -> Tuple[Optional[str], Optional[str], float, bool]:
        """
        Simulate a knowledge-base query.
        With probability hit_prob, return a synthetic answer.
        """
        score = float(np.random.beta(2, 3))  # realistic-ish score distribution
        if score >= query_confidence_threshold:
            return f"answer_from_peer_{self.peer_id}", f"knowledge_{self.peer_id}", score, True
        return None, f"knowledge_{self.peer_id}", score, False


# ---------------------------------------------------------------------------
# Mock RAGNetwork (TARW-style BFS, same logic as rag_network.py)
# ---------------------------------------------------------------------------

class MockRAGNetwork:
    """
    Minimal RAG network over a Barabási–Albert overlay graph.
    Implements the TARW query loop (topic-aware BFS with TTL).
    """

    def __init__(
        self,
        num_peers: int = 20,
        num_attachments: int = 4,
        num_query_neighbor: int = 4,
        peer_hit_prob: float = 0.4,
        seed: int = 0,
    ):
        self.num_peers = num_peers
        self.num_query_neighbor = num_query_neighbor
        self.seed = seed

        rng = random.Random(seed)
        np.random.seed(seed)

        # Build Barabási–Albert overlay graph (same as DRAG default)
        self.network = nx.barabasi_albert_graph(num_peers, num_attachments, seed=seed)

        # Create peers
        self.peers: List[Optional[MockPeer]] = [
            MockPeer(i, hit_prob=peer_hit_prob) for i in range(num_peers)
        ]

    def topic_aware_query(
        self,
        question: str,
        query_peer_id: Optional[int] = None,
        query_confidence_threshold: float = 0.5,
        max_ttl: int = 6,
    ) -> RAGAnswer:
        """
        BFS query (TARW-style). At each hop, up to num_query_neighbor
        unvisited neighbours are enqueued. Returns on first hit, or
        after TTL is exhausted.
        """
        if query_peer_id is None:
            query_peer_id = random.randrange(self.num_peers)

        visited: Set[int] = {query_peer_id}
        queue: deque = deque([(query_peer_id, 0)])
        num_messages = 0
        first_hop = max_ttl  # default: exhausted

        while queue:
            current_id, hop = queue.popleft()

            if hop >= max_ttl:
                continue

            peer = self.peers[current_id]
            if peer is None:
                # Genuinely removed node — skip (different from selective fwd)
                continue

            answer, knowledge, score, is_hit = peer.query(
                question, query_confidence_threshold
            )
            num_messages += 1

            if is_hit:
                return RAGAnswer(
                    answer=answer,
                    relevant_knowledge=knowledge,
                    relevant_score=score,
                    num_hops=hop,
                    num_messages=num_messages,
                    is_query_hit=True,
                )

            # Enqueue neighbours (capped at num_query_neighbor)
            neighbours = list(self.network.neighbors(current_id))
            unvisited = [n for n in neighbours if n not in visited]
            picked = unvisited[:self.num_query_neighbor]

            for nid in picked:
                visited.add(nid)
                queue.append((nid, hop + 1))

        # TTL exhausted — no answer found
        return RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=max_ttl,
            num_messages=num_messages,
            is_query_hit=False,
        )


# ---------------------------------------------------------------------------
# Selective Forwarding Attack (self-contained copy for standalone use)
# ---------------------------------------------------------------------------

class SelectiveForwardingAttack:
    """
    Compromises a fraction of peers so they silently drop queries.
    The peer stays in the overlay graph (passes health checks) but
    always returns (None, None, 0.0, False) from query().
    """

    def __init__(self, attack_ratio: float = 0.3, seed: int = 42):
        self.attack_ratio = attack_ratio
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self._patched: Dict[int, object] = {}
        self.compromised_ids: Set[int] = set()
        self._dropped: int = 0

    # ---- targeting -------------------------------------------------------

    def select_targets(self, network: MockRAGNetwork, strategy: str) -> List[int]:
        num_compromise = max(1, int(network.num_peers * self.attack_ratio))

        if strategy == "high_connectivity":
            degree_map = dict(network.network.degree())
            sorted_peers = sorted(degree_map, key=degree_map.get, reverse=True)
            targets = sorted_peers[:num_compromise]
        else:
            targets = random.sample(range(network.num_peers),
                                    min(num_compromise, network.num_peers))
        return targets

    # ---- apply / revert ---------------------------------------------------

    def apply(self, network: MockRAGNetwork, strategy: str = "random") -> Dict:
        if self._patched:
            return {}

        targets = self.select_targets(network, strategy)
        self.compromised_ids = set(targets)
        self._dropped = 0

        for pid in targets:
            peer = network.peers[pid]
            if peer is None:
                continue
            self._patched[pid] = peer.query
            atk = self

            def _drop(q, t, _pid=pid, _atk=atk):
                _atk._dropped += 1
                return None, None, 0.0, False

            peer.query = _drop

        degree_map = dict(network.network.degree())
        return {
            'strategy': strategy,
            'attack_ratio': self.attack_ratio,
            'num_compromised': len(self._patched),
            'compromised_ids': sorted(self._patched),
            'avg_degree_compromised': (
                np.mean([degree_map[p] for p in self._patched])
                if self._patched else 0.0
            ),
            'avg_degree_all': np.mean(list(degree_map.values())),
        }

    def revert(self, network: MockRAGNetwork) -> None:
        for pid, orig in self._patched.items():
            peer = network.peers[pid]
            if peer is not None:
                peer.query = orig
        self._patched.clear()
        self.compromised_ids.clear()

    # ---- metrics ----------------------------------------------------------

    @property
    def dropped_queries(self) -> int:
        return self._dropped

    def collect_metrics(self, answers: List[RAGAnswer], max_ttl: int) -> Dict:
        if not answers:
            return {}
        total = len(answers)
        hit = sum(1 for r in answers if r.is_query_hit)
        exhausted = sum(1 for r in answers if r.num_hops >= max_ttl and not r.is_query_hit)
        return {
            'hit_rate': hit / total,
            'avg_hops_per_query': float(np.mean([r.num_hops for r in answers])),
            'ttl_exhaustion_rate': exhausted / total,
            'dropped_queries': self._dropped,
            'total_queries': total,
        }


# ---------------------------------------------------------------------------
# Simulation sweep
# ---------------------------------------------------------------------------

def run_sweep(
    num_peers: int = 20,
    num_attachments: int = 4,
    num_query_neighbor: int = 4,
    max_ttl: int = 6,
    num_queries: int = 100,
    peer_hit_prob: float = 0.4,
    query_confidence_threshold: float = 0.5,
    ratios: List[float] = None,
    strategies: List[str] = None,
    seed: int = 0,
) -> List[Dict]:
    if ratios is None:
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    if strategies is None:
        strategies = ["random", "high_connectivity"]

    # Build network once — reuse across all runs (attack.revert() restores it)
    network = MockRAGNetwork(
        num_peers=num_peers,
        num_attachments=num_attachments,
        num_query_neighbor=num_query_neighbor,
        peer_hit_prob=peer_hit_prob,
        seed=seed,
    )

    # Fixed set of query starting peers for reproducibility
    rng = random.Random(seed + 1)
    query_peers = [rng.randrange(num_peers) for _ in range(num_queries)]

    results = []

    for strategy in strategies:
        for ratio in ratios:
            label = f"ratio={ratio:.1f}  strategy={strategy}"

            if ratio == 0.0:
                # Baseline — no attack
                answers = [
                    network.topic_aware_query(
                        question=f"q{i}",
                        query_peer_id=query_peers[i],
                        query_confidence_threshold=query_confidence_threshold,
                        max_ttl=max_ttl,
                    )
                    for i in range(num_queries)
                ]
                total = len(answers)
                hit = sum(1 for r in answers if r.is_query_hit)
                exhausted = sum(1 for r in answers if r.num_hops >= max_ttl and not r.is_query_hit)
                row = {
                    'strategy': 'baseline',
                    'attack_ratio': 0.0,
                    'num_compromised': 0,
                    'avg_degree_compromised': 0.0,
                    'avg_degree_all': float(np.mean([d for _, d in network.network.degree()])),
                    'hit_rate': hit / total,
                    'avg_hops_per_query': float(np.mean([r.num_hops for r in answers])),
                    'ttl_exhaustion_rate': exhausted / total,
                    'dropped_queries': 0,
                    'total_queries': total,
                }
                # Only add baseline once (same regardless of strategy)
                if not any(r['strategy'] == 'baseline' for r in results):
                    results.append(row)
                    _print_row(row)
                continue

            # Apply attack
            sfa = SelectiveForwardingAttack(attack_ratio=ratio, seed=seed)
            apply_info = sfa.apply(network, strategy=strategy)

            # Run queries under attack
            answers = [
                network.topic_aware_query(
                    question=f"q{i}",
                    query_peer_id=query_peers[i],
                    query_confidence_threshold=query_confidence_threshold,
                    max_ttl=max_ttl,
                )
                for i in range(num_queries)
            ]

            metrics = sfa.collect_metrics(answers, max_ttl)
            sfa.revert(network)  # restore for next run

            row = {**apply_info, **metrics}
            results.append(row)
            _print_row(row)

    return results


def _print_row(row: Dict) -> None:
    strat = row['strategy']
    ratio = row['attack_ratio']
    nc = row['num_compromised']
    hr = row['hit_rate']
    ah = row['avg_hops_per_query']
    ter = row['ttl_exhaustion_rate']
    dq = row['dropped_queries']
    print(
        f"  {strat:20s}  ratio={ratio:.1f}  compromised={nc:3d}  "
        f"hit_rate={hr:.3f}  avg_hops={ah:.2f}  "
        f"ttl_exhaust={ter:.3f}  dropped={dq}"
    )


def save_csv(results: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not results:
        return
    # Flatten list/set fields to pipe-joined strings for CSV compatibility
    clean = []
    for row in results:
        clean.append({
            k: ("|".join(str(x) for x in v) if isinstance(v, (list, set)) else v)
            for k, v in row.items()
        })
    # Union of all keys across all rows (baseline row differs from attack rows)
    all_keys: List[str] = []
    seen: set = set()
    for row in clean:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in clean:
            padded = {k: row.get(k, "") for k in all_keys}
            writer.writerow(padded)
    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Selective Forwarding Attack simulation")
    p.add_argument("--num_peers", type=int, default=20,
                   help="Number of peers in the network (default: 20)")
    p.add_argument("--num_attachments", type=int, default=4,
                   help="BA graph attachment parameter (default: 4)")
    p.add_argument("--num_query_neighbor", type=int, default=4,
                   help="Max neighbours queried per hop (default: 4)")
    p.add_argument("--max_ttl", type=int, default=6,
                   help="Maximum query TTL / hops (default: 6)")
    p.add_argument("--num_queries", type=int, default=100,
                   help="Queries to run per scenario (default: 100)")
    p.add_argument("--peer_hit_prob", type=float, default=0.4,
                   help="Per-peer answer probability (default: 0.4)")
    p.add_argument("--confidence_threshold", type=float, default=0.5,
                   help="Score threshold for accepting an answer (default: 0.5)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (default: 0)")
    p.add_argument("--output", type=str,
                   default="logs/selective_forwarding/results.csv",
                   help="CSV output path")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  Selective Forwarding Attack — DRAG Simulation")
    print("=" * 70)
    print(f"  peers={args.num_peers}  ttl={args.max_ttl}  "
          f"queries={args.num_queries}  hit_prob={args.peer_hit_prob}")
    print()

    print("Strategy / Ratio          | hit_rate | avg_hops | ttl_exhaust | dropped")
    print("-" * 70)

    results = run_sweep(
        num_peers=args.num_peers,
        num_attachments=args.num_attachments,
        num_query_neighbor=args.num_query_neighbor,
        max_ttl=args.max_ttl,
        num_queries=args.num_queries,
        peer_hit_prob=args.peer_hit_prob,
        query_confidence_threshold=args.confidence_threshold,
        seed=args.seed,
    )

    print()
    print("=" * 70)
    print("  Summary: hit_rate degradation vs baseline")
    print("=" * 70)
    baseline_hr = next(r['hit_rate'] for r in results if r['strategy'] == 'baseline')
    print(f"  Baseline hit_rate: {baseline_hr:.3f}")
    print()
    for r in results:
        if r['strategy'] == 'baseline':
            continue
        delta = r['hit_rate'] - baseline_hr
        print(
            f"  {r['strategy']:20s}  ratio={r['attack_ratio']:.1f}  "
            f"hit_rate={r['hit_rate']:.3f}  "
            f"delta={delta:+.3f}  "
            f"ttl_exhaust={r['ttl_exhaustion_rate']:.3f}"
        )

    save_csv(results, args.output)


if __name__ == "__main__":
    main()
