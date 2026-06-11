"""
run_selective_forwarding_with_defense.py
=========================================
Same simulation as run_selective_forwarding.py but runs each attack scenario
THREE times side-by-side:

  1. Baseline      — no attack, no defense
  2. Attack only   — SelectiveForwardingAttack, no defense
  3. Attack + SFD  — SelectiveForwardingAttack + SelectiveForwardingDefense

This lets you measure the recovery the defense achieves at each attack ratio
and targeting strategy.

Usage
-----
  python run_selective_forwarding_with_defense.py
  python run_selective_forwarding_with_defense.py --num_peers 30 --max_ttl 8 --num_queries 200
  python run_selective_forwarding_with_defense.py --blacklist_threshold 0.15 --min_queries 4

Results are printed to stdout and saved to:
  logs/selective_forwarding/results_with_defense.csv
"""

import argparse
import csv
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from loguru import logger as _log
_log.remove()
_log.add(sys.stderr, level="WARNING")
del _log

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
    """Probabilistic knowledge-base hit — no LLM / embeddings needed."""

    def __init__(self, peer_id: int, hit_prob: float = 0.4):
        self.peer_id = peer_id
        self.hit_prob = hit_prob

    def query(
        self, question: str, query_confidence_threshold: float
    ) -> Tuple[Optional[str], Optional[str], float, bool]:
        score = float(np.random.beta(2, 3))
        if score >= query_confidence_threshold:
            return f"answer_from_peer_{self.peer_id}", f"knowledge_{self.peer_id}", score, True
        return None, f"knowledge_{self.peer_id}", score, False


# ---------------------------------------------------------------------------
# Mock RAGNetwork
# ---------------------------------------------------------------------------

class MockRAGNetwork:
    """Barabási–Albert overlay graph with TARW-style BFS query routing."""

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

        np.random.seed(seed)
        self.network = nx.barabasi_albert_graph(num_peers, num_attachments, seed=seed)
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
        BFS query (TARW-style).

        If a SelectiveForwardingDefense has been applied to the network via
        defense.apply(network), blacklisted peers are skipped and the BFS
        expands through their neighbours instead, recovering reachability.
        """
        if query_peer_id is None:
            query_peer_id = random.randrange(self.num_peers)

        defense = getattr(self, "_sfa_defense", None)

        visited: Set[int] = {query_peer_id}
        queue: deque = deque([(query_peer_id, 0)])
        num_messages = 0

        while queue:
            current_id, hop = queue.popleft()

            if hop >= max_ttl:
                continue

            peer = self.peers[current_id]
            if peer is None:
                continue

            # Defense bypass: skip blacklisted peer and expand its neighbours
            if defense is not None and defense.is_peer_blacklisted(current_id):
                defense._total_bypasses += 1
                neighbours = list(self.network.neighbors(current_id))
                unvisited = [n for n in neighbours if n not in visited]
                for nid in unvisited[: self.num_query_neighbor]:
                    visited.add(nid)
                    queue.append((nid, hop + 1))
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

            # Enqueue neighbours
            neighbours = list(self.network.neighbors(current_id))
            unvisited = [n for n in neighbours if n not in visited]
            for nid in unvisited[: self.num_query_neighbor]:
                visited.add(nid)
                queue.append((nid, hop + 1))

        return RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=max_ttl,
            num_messages=num_messages,
            is_query_hit=False,
        )


# ---------------------------------------------------------------------------
# Selective Forwarding Attack (self-contained copy)
# ---------------------------------------------------------------------------

class SelectiveForwardingAttack:
    """
    Compromises a fraction of peers so they silently drop queries.
    The peer stays in the overlay graph but always returns
    (None, None, 0.0, False) from query().
    """

    def __init__(self, attack_ratio: float = 0.3, seed: int = 42):
        self.attack_ratio = attack_ratio
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self._patched: Dict[int, object] = {}
        self.compromised_ids: Set[int] = set()
        self._dropped: int = 0

    def select_targets(self, network: MockRAGNetwork, strategy: str) -> List[int]:
        num_compromise = max(1, int(network.num_peers * self.attack_ratio))
        if strategy == "high_connectivity":
            degree_map = dict(network.network.degree())
            sorted_peers = sorted(degree_map, key=degree_map.get, reverse=True)
            return sorted_peers[:num_compromise]
        return random.sample(
            range(network.num_peers), min(num_compromise, network.num_peers)
        )

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
            "strategy": strategy,
            "attack_ratio": self.attack_ratio,
            "num_compromised": len(self._patched),
            "compromised_ids": sorted(self._patched),
            "avg_degree_compromised": (
                float(np.mean([degree_map[p] for p in self._patched]))
                if self._patched else 0.0
            ),
            "avg_degree_all": float(np.mean(list(degree_map.values()))),
        }

    def revert(self, network: MockRAGNetwork) -> None:
        for pid, orig in self._patched.items():
            peer = network.peers[pid]
            if peer is not None:
                peer.query = orig
        self._patched.clear()
        self.compromised_ids.clear()

    @property
    def dropped_queries(self) -> int:
        return self._dropped

    def collect_metrics(self, answers: List[RAGAnswer], max_ttl: int) -> Dict:
        if not answers:
            return {}
        total = len(answers)
        hit = sum(1 for r in answers if r.is_query_hit)
        exhausted = sum(
            1 for r in answers if r.num_hops >= max_ttl and not r.is_query_hit
        )
        return {
            "hit_rate": hit/total,
            "early_hit_rate": sum(1 for r in answers[:len(answers)//2] if r.is_query_hit)/max(1,len(answers)//2),
            "late_hit_rate": sum(1 for r in answers[len(answers)//2:] if r.is_query_hit)/max(1,len(answers)-len(answers)//2),
            "avg_hops_per_query": float(np.mean([r.num_hops for r in answers])),
            "ttl_exhaustion_rate": exhausted / total,
            "dropped_queries": self._dropped,
            "total_queries": total,
        }


# ---------------------------------------------------------------------------
# Defense import
# ---------------------------------------------------------------------------

try:
    from modules.defenses.selective_forwarding_defense import SelectiveForwardingDefense
    _HAS_DEFENSE = True
except ImportError:
    _HAS_DEFENSE = False
    print("[WARNING] Could not import SelectiveForwardingDefense; defense column will be empty.")


# ---------------------------------------------------------------------------
# Simulation sweep: baseline / attack-only / attack+defense
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
    defense_config: Dict = None,
) -> List[Dict]:
    if ratios is None:
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    if strategies is None:
        strategies = ["random", "high_connectivity"]
    if defense_config is None:
        defense_config = {}

    network = MockRAGNetwork(
        num_peers=num_peers,
        num_attachments=num_attachments,
        num_query_neighbor=num_query_neighbor,
        peer_hit_prob=peer_hit_prob,
        seed=seed,
    )

    rng = random.Random(seed + 1)
    query_peers = [rng.randrange(num_peers) for _ in range(num_queries)]

    results = []
    baseline_done = False

    for strategy in strategies:
        for ratio in ratios:

            # ---- Baseline (no attack, no defense) ----
            if ratio == 0.0 and not baseline_done:
                answers = [
                    network.topic_aware_query(
                        question=f"q{i}",
                        query_peer_id=query_peers[i],
                        query_confidence_threshold=query_confidence_threshold,
                        max_ttl=max_ttl,
                    )
                    for i in range(num_queries)
                ]
                row = _metrics_row(
                    answers, max_ttl,
                    label="baseline", strategy="none", ratio=0.0, num_compromised=0,
                    network=network,
                )
                results.append(row)
                _print_row(row)
                baseline_done = True
                continue
            elif ratio == 0.0:
                continue

            # ---- Attack only ----
            sfa = SelectiveForwardingAttack(attack_ratio=ratio, seed=seed)
            apply_info = sfa.apply(network, strategy=strategy)

            answers_attack = [
                network.topic_aware_query(
                    question=f"q{i}",
                    query_peer_id=query_peers[i],
                    query_confidence_threshold=query_confidence_threshold,
                    max_ttl=max_ttl,
                )
                for i in range(num_queries)
            ]
            metrics_attack = sfa.collect_metrics(answers_attack, max_ttl)
            sfa.revert(network)

            row_attack = {
                "label": "attack_only",
                "strategy": strategy,
                "attack_ratio": ratio,
                "num_compromised": apply_info.get("num_compromised", 0),
                "avg_degree_compromised": apply_info.get("avg_degree_compromised", 0.0),
                "avg_degree_all": apply_info.get("avg_degree_all", 0.0),
                **metrics_attack,
                "defense_blacklisted": 0,
                "defense_bypasses": 0,
            }
            results.append(row_attack)
            _print_row(row_attack)

            # ---- Attack + Defense ----
            if _HAS_DEFENSE:
                sfa2 = SelectiveForwardingAttack(attack_ratio=ratio, seed=seed)
                apply_info2 = sfa2.apply(network, strategy=strategy)

                defense = SelectiveForwardingDefense(defense_config)
                defense.apply(network)

                answers_defended = [
                    network.topic_aware_query(
                        question=f"q{i}",
                        query_peer_id=query_peers[i],
                        query_confidence_threshold=query_confidence_threshold,
                        max_ttl=max_ttl,
                    )
                    for i in range(num_queries)
                ]

                metrics_defended = sfa2.collect_metrics(answers_defended, max_ttl)
                def_stats = defense.get_stats()

                defense.remove(network)
                sfa2.revert(network)

                row_defended = {
                    "label": "attack_plus_defense",
                    "strategy": strategy,
                    "attack_ratio": ratio,
                    "num_compromised": apply_info2.get("num_compromised", 0),
                    "avg_degree_compromised": apply_info2.get("avg_degree_compromised", 0.0),
                    "avg_degree_all": apply_info2.get("avg_degree_all", 0.0),
                    **metrics_defended,
                    "defense_blacklisted": def_stats.get("blacklisted_count", 0),
                    "defense_bypasses": def_stats.get("total_bypasses", 0),
                }
                results.append(row_defended)
                _print_row(row_defended)

    return results


def _metrics_row(
    answers, max_ttl, label, strategy, ratio, num_compromised, network
) -> Dict:
    total = len(answers)
    hit = sum(1 for r in answers if r.is_query_hit)
    exhausted = sum(
        1 for r in answers if r.num_hops >= max_ttl and not r.is_query_hit
    )
    avg_deg = float(np.mean([d for _, d in network.network.degree()]))
    return {
        "label": label,
        "strategy": strategy,
        "attack_ratio": ratio,
        "num_compromised": num_compromised,
        "avg_degree_compromised": 0.0,
        "avg_degree_all": avg_deg,
        "hit_rate": hit / total,
        "avg_hops_per_query": float(np.mean([r.num_hops for r in answers])),
        "ttl_exhaustion_rate": exhausted / total,
        "dropped_queries": 0,
        "total_queries": total,
        "defense_blacklisted": 0,
        "defense_bypasses": 0,
    }


def _print_row(row: Dict) -> None:
    label = row["label"]
    strat = row["strategy"]
    ratio = row["attack_ratio"]
    nc = row["num_compromised"]
    hr = row["hit_rate"]
    ter = row["ttl_exhaustion_rate"]
    bl = row.get("defense_blacklisted", 0)
    byp = row.get("defense_bypasses", 0)
    print(
        f"  {label:22s}  {strat:20s}  ratio={ratio:.1f}  comp={nc:3d}"
        f"  hit={hr:.3f}  ttl_ex={ter:.3f}  bl={bl:3d}  byp={byp:4d}"
    )


def save_csv(results: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not results:
        return
    all_keys: List[str] = []
    seen: set = set()
    for row in results:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in all_keys})
    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Selective Forwarding Attack + Defense simulation"
    )
    p.add_argument("--num_peers", type=int, default=20)
    p.add_argument("--num_attachments", type=int, default=4)
    p.add_argument("--num_query_neighbor", type=int, default=4)
    p.add_argument("--max_ttl", type=int, default=6)
    p.add_argument("--num_queries", type=int, default=100)
    p.add_argument("--peer_hit_prob", type=float, default=0.4)
    p.add_argument("--confidence_threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        type=str,
        default="logs/selective_forwarding/results_with_defense.csv",
    )
    # Defense tuning
    p.add_argument(
        "--blacklist_threshold", type=float, default=0.05,
        help="Response-rate below which a peer is blacklisted (default: 0.20)"
    )
    p.add_argument(
        "--min_queries", type=int, default=15,
        help="Minimum queries before blacklisting can trigger (default: 5)"
    )
    p.add_argument(
        "--suspicion_threshold", type=float, default=0.10,
        help="Reputation below this marks peer as suspicious (default: 0.35)"
    )
    p.add_argument(
        "--reputation_decay", type=float, default=0.70,
        help="EMA decay for reputation (higher = slower adaptation, default: 0.85)"
    )
    p.add_argument("--trials",type=int,default=1,help="Trials to average")
    return p.parse_args()


def main():
    args = parse_args()

    defense_config = {
        "blacklist_threshold": args.blacklist_threshold,
        "min_queries_before_blacklist": args.min_queries,
        "suspicion_threshold": args.suspicion_threshold,
        "reputation_decay": args.reputation_decay,
    }

    print("=" * 80)
    print("  Selective Forwarding Attack + Defense — DRAG Simulation")
    print("=" * 80)
    print(
        f"  peers={args.num_peers}  ttl={args.max_ttl}  "
        f"queries={args.num_queries}  hit_prob={args.peer_hit_prob}"
    )
    print(
        f"  defense: blacklist_threshold={args.blacklist_threshold}  "
        f"min_queries={args.min_queries}  suspicion={args.suspicion_threshold}"
    )
    print()

    header = (
        f"  {'label':22s}  {'strategy':20s}  ratio  comp"
        f"  hit    ttl_ex  bl   byp"
    )
    print(header)
    print("-" * 80)

    results = run_sweep(
        num_peers=args.num_peers,
        num_attachments=args.num_attachments,
        num_query_neighbor=args.num_query_neighbor,
        max_ttl=args.max_ttl,
        num_queries=args.num_queries,
        peer_hit_prob=args.peer_hit_prob,
        query_confidence_threshold=args.confidence_threshold,
        seed=args.seed,
        defense_config=defense_config,
    )

    # Summary: hit_rate recovery per scenario
    print()
    print("=" * 80)
    print("  Summary: hit_rate recovery (attack_only vs attack_plus_defense)")
    print("=" * 80)
    baseline_hr = next(
        (r["hit_rate"] for r in results if r["label"] == "baseline"), None
    )
    if baseline_hr is not None:
        print(f"  Baseline hit_rate: {baseline_hr:.3f}")
        print()
        seen_pairs = set()
        for r in results:
            if r["label"] != "attack_only":
                continue
            key = (r["strategy"], r["attack_ratio"])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            defended = next(
                (
                    x for x in results
                    if x["label"] == "attack_plus_defense"
                    and x["strategy"] == r["strategy"]
                    and x["attack_ratio"] == r["attack_ratio"]
                ),
                None,
            )

            hr_a = r["hit_rate"]
            hr_d = defended["hit_rate"] if defended else float("nan")
            recovery = hr_d - hr_a if defended else float("nan")
            late_a=r.get('late_hit_rate',hr_a); late_d=defended.get('late_hit_rate',hr_d) if defended else float('nan')
            late_rec=late_d-late_a if defended else float('nan')
            print(f"  {r['strategy']:20s}  ratio={r['attack_ratio']:.1f}  attack={hr_a:.3f}  defended={hr_d:.3f}  recovery={recovery:+.3f}  late_recover={late_rec:+.3f}")

    if args.trials>1:
        import copy
        all_r=[]; base_seed=args.seed
        for t in range(args.trials):
            args.seed=base_seed+t*97
            rr=run_sweep(num_peers=args.num_peers,num_attachments=args.num_attachments,num_query_neighbor=args.num_query_neighbor,max_ttl=args.max_ttl,num_queries=args.num_queries,peer_hit_prob=args.peer_hit_prob,query_confidence_threshold=args.confidence_threshold,seed=args.seed,defense_config=defense_config)
            all_r.append(rr)
        args.seed=base_seed
        import numpy as _np
        for i,row in enumerate(results):
            for field in ['hit_rate','early_hit_rate','late_hit_rate','ttl_exhaustion_rate','defense_blacklisted','defense_bypasses']:
                vals=[t[i].get(field,0) for t in all_r if i<len(t)]
                if vals: row[field]=float(_np.mean(vals)); row[field+'_std']=float(_np.std(vals))
    save_csv(results, args.output)


if __name__ == "__main__":
    main()
