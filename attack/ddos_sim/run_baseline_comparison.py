"""
attack/ddos_sim/run_baseline_comparison.py

DDoS item 2 of the NAACL improvement proposal: "Compare against strong
baselines: centralized RAG, Reliable-dRAG without reliability-aware
routing, replicated retrieval, ordinary rate limiting, and load-aware
routing. Show whether the distributed design is more or less resilient
than alternatives."

Reuses attack.ddos_sim.ddos_attack.DDoSAttack and
attack.ddos_sim.network_sim.MockRAGNetwork unmodified (no new attack
mechanism) -- this script only varies TOPOLOGY (how many/which nodes hold
the corpus and how a query reaches one) and DEFENSE (none / load-aware /
flat rate-limit) around the same, already-validated attack.

What is and isn't faithfully reproduced here
-----------------------------------------------
Three of the five named baselines map cleanly onto this mock simulation
layer and ARE implemented:

  - "centralized RAG"       -> topology="centralized": num_peers=1. If
                                DDoS targets the only node (which it always
                                does, attack_ratio>=1/1), availability
                                collapses to 0% the instant that one node
                                is targeted -- the point of comparison
                                isn't nuance, it's the structural fact that
                                a single point of failure has no fallback
                                at all, unlike a distributed overlay where
                                un-targeted peers keep answering.
  - "replicated retrieval"  -> topology="replicated": a NEW routing
                                function, `replicated_query()` below, that
                                queries EVERY peer for a question (not
                                BFS-until-first-hit) and succeeds if ANY
                                one of them answers -- the maximally
                                redundant alternative to this project's
                                actual TTL-bounded BFS. Costs `num_peers`
                                "hops" per query regardless of outcome
                                (every peer is always queried), which is
                                the fairness-relevant tradeoff to report
                                alongside its higher hit-rate: replication
                                is not free.
  - "ordinary rate limiting" vs "load-aware routing" -> defense="flat_rate_limit"
                                vs defense="ddos_defense": a flat
                                admission-control stub (admits a fixed
                                fraction of queries to ANY peer,
                                independent of whether that specific peer
                                is actually the one currently overloaded)
                                compared against the existing, reused
                                `DDoSDefense` (load-aware: tracks EACH
                                peer's own response-rate reputation and
                                specifically bypasses the ones that are
                                actually struggling).

Two are explicitly NOT reproduced here, flagged rather than faked:

  - "Reliable-dRAG without reliability-aware routing" -- reliability-aware
    routing (`rerank_with_reliability` / `reliability_weight`,
    drag_llm_service/configs/config.yaml) is a property of the REAL
    drag_llm_service reranker (drag_llm_service/src/retriever/reranker.py),
    which this mock simulation's topic_aware_query() does not model at
    all (it's plain TTL-bounded BFS with no reliability weighting at this
    layer, live or mock). Testing this baseline correctly means running
    the live deployment twice with that config flag flipped, under real
    live_flood.TrafficFlood congestion -- infra/live work, not something
    this mock harness can honestly approximate.
  - A true "centralized RAG" system (one server, one document index, no
    distributed retrieval architecture at all) is a DIFFERENT SYSTEM, not
    a configuration of this one. topology="centralized" above is the
    closest honest analogue expressible within this project's own
    network model (one node holding everything), not a claim of having
    built or benchmarked an actual alternative RAG system.

Usage
-----
  python attack/ddos_sim/run_baseline_comparison.py
  python attack/ddos_sim/run_baseline_comparison.py --topologies distributed centralized replicated \\
      --defenses none ddos_defense flat_rate_limit --seeds 0 42 123
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.selective_forward_sim.network_sim import MockRAGNetwork  # noqa: E402
from attack.ddos_sim.ddos_attack import DDoSAttack  # noqa: E402
from defense.ddos_sim_defense.ddos_defense import DDoSDefense  # noqa: E402

LOG_DIR = os.path.join(_ROOT, "attack_logs", "ddos_sim")
os.makedirs(LOG_DIR, exist_ok=True)

TOPOLOGIES = ["distributed", "centralized", "replicated"]
DEFENSES = ["none", "ddos_defense", "flat_rate_limit"]
DEFAULT_SEEDS = [0, 42, 123]  # this project's documented multi-seed convention


def replicated_query(network, question: str, threshold: float):
    """Query EVERY peer (not BFS-until-first-hit) -- succeeds if any one
    answers above threshold. Reuses MockPeer.query() directly (same
    interface DDoSAttack's monkey-patch wraps), so an attached DDoSAttack's
    congestion-dropping still applies per-peer exactly as it does for the
    BFS router. Returns a network_sim.QueryResult for drop-in compatibility
    with DDoSAttack.collect_metrics()."""
    from attack.selective_forward_sim.network_sim import QueryResult

    hops = 0
    log: List[str] = []
    for pid, peer in enumerate(network.peers):
        if peer is None:
            continue
        hops += 1
        answer, knowledge, score, is_hit = peer.query(question, threshold)
        log.append(f"peer_{pid}: {'HIT' if is_hit else 'MISS'}")
        if is_hit and score >= threshold:
            return QueryResult(question, answer, True, hops, log)
    return QueryResult(question, None, False, hops, log)


class _FlatRateLimitDefense:
    """"Ordinary rate limiting" baseline -- admits a fixed fraction of
    queries to ANY peer, uninformed by which peer is actually the one
    currently overloaded. Deliberately NOT load-aware, as a foil for
    DDoSDefense's reputation-informed approach. Duck-typed to the same
    _sfa_defense hook interface (is_peer_blacklisted / record_bypass /
    backup_candidates / record_redundant_probe), so it installs and
    compares identically."""

    def __init__(self, admit_prob: float, seed: int):
        import random
        self.admit_prob = admit_prob
        self._rng = random.Random(seed)
        self.redundancy_k = 1
        self._total_bypasses = 0

    def apply(self, network) -> None:
        network._sfa_defense = self

    def remove(self, network) -> None:
        if getattr(network, "_sfa_defense", None) is self:
            network._sfa_defense = None

    def is_peer_blacklisted(self, peer_id: int) -> bool:
        return self._rng.random() > self.admit_prob  # uninformed: same odds for every peer

    def record_bypass(self) -> None:
        self._total_bypasses += 1

    def backup_candidates(self, tried_ids, all_peer_ids: List[int]) -> List[int]:
        untried = [p for p in all_peer_ids if p not in set(tried_ids)]
        return untried[: self.redundancy_k]

    def record_redundant_probe(self, hit: bool) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {"defense": "flat_rate_limit", "admit_prob": self.admit_prob,
                "total_bypasses": self._total_bypasses}


def build_network(topology: str, args: argparse.Namespace, seed: int) -> MockRAGNetwork:
    if topology == "centralized":
        # networkx.barabasi_albert_graph(n, m) requires m < n, so
        # MockRAGNetwork's normal constructor cannot build a genuine
        # single-node network (n=1 admits no valid m>=1). Bypass __init__
        # via __new__ and set the exact attributes topic_aware_query()
        # actually reads, so the SAME bound method runs unmodified against
        # a true one-node "centralized" network -- not a 2-node
        # approximation, which would understate a real single-point-of-
        # failure's fragility.
        import random as _random
        import networkx as _nx
        from attack.selective_forward_sim.network_sim import MockPeer

        network = MockRAGNetwork.__new__(MockRAGNetwork)
        network.num_peers = 1
        network.num_query_neighbor = args.num_query_neighbor
        network.query_ttl = args.max_ttl
        network._rng = _random.Random(seed)
        network.network = _nx.Graph()
        network.network.add_node(0)
        network.peers = [MockPeer(0, args.peer_hit_prob, _random.Random(seed * 1_000_003))]
        network._sfa_defense = None
        return network

    num_peers = args.num_peers
    return MockRAGNetwork(
        num_peers=num_peers, num_attachments=min(args.num_attachments, max(1, num_peers - 1)),
        num_query_neighbor=args.num_query_neighbor, query_ttl=args.max_ttl,
        peer_hit_prob=args.peer_hit_prob, seed=seed,
    )


def build_defense(defense_name: str, args: argparse.Namespace, seed: int):
    if defense_name == "none":
        return None
    if defense_name == "ddos_defense":
        return DDoSDefense({
            "deprioritize_threshold": args.deprioritize_threshold,
            "min_queries_before_action": args.min_queries_before_action,
            "redundancy_k": args.redundancy_k,
        })
    if defense_name == "flat_rate_limit":
        return _FlatRateLimitDefense(admit_prob=args.flat_admit_prob, seed=seed)
    raise ValueError(defense_name)


def run_one(args: argparse.Namespace, topology: str, defense_name: str, seed: int) -> Dict[str, Any]:
    network = build_network(topology, args, seed)
    defense = build_defense(defense_name, args, seed)
    if defense is not None:
        defense.apply(network)

    attack = DDoSAttack(
        attack_ratio=args.attack_ratio, iterations=args.iterations, strategy=args.strategy,
        ddos_duration=args.ddos_duration, wave_interval_s=args.wave_interval_s, seed=seed,
    )
    attack.attach(network)

    questions = [f"query_{i}" for i in range(args.num_questions)]
    batch = max(1, len(questions) // args.iterations)
    q_idx = 0
    for wave_num in range(args.iterations):
        attack.run_wave(network, wave_num)
        wave_questions = questions[q_idx: q_idx + batch]
        q_idx += batch
        if topology == "replicated":
            for q in wave_questions:
                replicated_query(network, q, args.threshold)
        else:
            for q in wave_questions:
                network.topic_aware_query(q, args.threshold)

    # Final measurement pass, post-attack state, no further waves.
    if topology == "replicated":
        final_answers = [replicated_query(network, q, args.threshold) for q in questions]
    else:
        final_answers = [network.topic_aware_query(q, args.threshold) for q in questions]
    metrics = attack.collect_metrics(final_answers, max_ttl=args.max_ttl)

    defense_stats = defense.get_stats() if defense is not None else {}
    attack.detach(network)
    if defense is not None:
        defense.remove(network)

    return {
        "topology": topology, "defense": defense_name, "seed": seed,
        "num_peers": network.num_peers,
        **metrics,
        "defense_stats": defense_stats,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--topologies", nargs="+", default=TOPOLOGIES, choices=TOPOLOGIES)
    p.add_argument("--defenses", nargs="+", default=DEFENSES, choices=DEFENSES)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--num_peers", type=int, default=20)
    p.add_argument("--num_attachments", type=int, default=4)
    p.add_argument("--num_query_neighbor", type=int, default=4)
    p.add_argument("--max_ttl", type=int, default=6)
    p.add_argument("--peer_hit_prob", type=float, default=0.4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--num_questions", type=int, default=100)
    p.add_argument("--attack_ratio", type=float, default=0.5)
    p.add_argument("--strategy", choices=["random", "sequential", "targeted"], default="targeted")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--ddos_duration", type=float, default=60.0)
    p.add_argument("--wave_interval_s", type=float, default=30.0)
    p.add_argument("--deprioritize_threshold", type=float, default=0.5)
    p.add_argument("--min_queries_before_action", type=int, default=5)
    p.add_argument("--redundancy_k", type=int, default=2)
    p.add_argument("--flat_admit_prob", type=float, default=0.5,
                    help="fixed admission probability for the flat_rate_limit baseline")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: List[Dict[str, Any]] = []
    for topology in args.topologies:
        for defense_name in args.defenses:
            per_seed = [run_one(args, topology, defense_name, seed) for seed in args.seeds]
            n = len(per_seed)
            mean_hit_rate = sum(r["hit_rate"] for r in per_seed) / n
            std_hit_rate = (
                (sum((r["hit_rate"] - mean_hit_rate) ** 2 for r in per_seed) / (n - 1)) ** 0.5
                if n > 1 else 0.0
            )
            print(f"  topology={topology:<12s} defense={defense_name:<16s}  "
                  f"hit_rate={mean_hit_rate:.3f}±{std_hit_rate:.3f}  (n={n} seeds)")
            all_rows.append({
                "topology": topology, "defense": defense_name,
                "n_seeds": n, "mean_hit_rate": mean_hit_rate, "std_hit_rate": std_hit_rate,
                "per_seed": per_seed,
            })

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOG_DIR, f"baseline_comparison_{ts}.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump({"timestamp": ts, "config": vars(args), "results": all_rows}, fh, indent=2, default=str)
    print(f"\nLog saved -> {log_path}")


if __name__ == "__main__":
    main()
