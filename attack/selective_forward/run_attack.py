"""
attack/selective_forward/run_attack.py

Usage:
    python3 attack/selective_forward/run_attack.py --mode mock
    python3 attack/selective_forward/run_attack.py --mode stealthy
    python3 attack/selective_forward/run_attack.py --mode detection
    python3 attack/selective_forward/run_attack.py --mode mitigation
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# ── Make sure the package root is on the path ────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from selective_forward_attack import (  # noqa: E402
    SelectiveForwardingAttack,
    SFADetector,
    SFAMitigation,
    check_blockchain_status,
    _LEDGER,
)

# ══════════════════════════════════════════════════════════════════════════
#  Minimal mock source node (used when the real dRAG network isn't running)
# ══════════════════════════════════════════════════════════════════════════

class _MockSource:
    def __init__(self, node_id: str, rng: np.random.Generator):
        self.node_id = node_id
        self._rng = rng

    def query(self, question: str, k: int = 5):
        score = float(self._rng.beta(2, 3))
        hit   = score >= 0.45
        return [], score, hit


def _make_network(n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    sources = [
        _MockSource(f"node_{i:02d}", np.random.default_rng(rng.integers(0, 2**31)))
        for i in range(n)
    ]
    for src in sources:
        _LEDGER.register(src.node_id)
    return sources


_QUERIES = [
    "What is blockchain?",
    "How does federated learning work?",
    "Explain selective forwarding attack.",
    "What is cosine similarity?",
    "How does anomaly detection work?",
    "What is Byzantine fault tolerance?",
    "Explain proof-of-work.",
    "What is differential privacy?",
    "How does RAG retrieval work?",
    "What is the Turing test?",
]


def _get_queries(n: int, rng: np.random.Generator) -> list:
    idxs = rng.integers(0, len(_QUERIES), size=n)
    return [_QUERIES[i] for i in idxs]


# ══════════════════════════════════════════════════════════════════════════
#  RUN MODES
# ══════════════════════════════════════════════════════════════════════════

def run_mock(args):
    """Baseline mock — no attack, no mitigation."""
    print("\n[mock] Baseline run — no attack")
    rng     = np.random.default_rng(args.seed)
    sources = _make_network(args.nodes, args.seed)
    queries = _get_queries(args.queries, rng)

    hits = 0
    for q in queries:
        for src in sources[:args.max_hops]:
            _, score, hit = src.query(q)
            if hit:
                hits += 1
                break

    print(f"  Hit rate : {hits/args.queries*100:.1f}%  ({hits}/{args.queries})")
    bc = check_blockchain_status()
    print(f"  Blockchain: {bc}")


def run_stealthy(args):
    """Stealthy gray-hole attack — 10-30 % probabilistic drop."""
    print(f"\n[stealthy] SFA ratio={args.ratio}  strategy={args.strategy}")
    rng     = np.random.default_rng(args.seed)
    sources = _make_network(args.nodes, args.seed)

    atk = SelectiveForwardingAttack(
        attack_ratio=args.ratio,
        strategy=args.strategy,
        drop_rate="stealthy",
        seed=args.seed + 999,
    )
    ssm_scores = {s.node_id: _LEDGER.score(s.node_id) for s in sources}
    atk.apply(sources, ssm_scores)

    queries  = _get_queries(args.queries, rng)
    hits     = 0
    ttl_exp  = 0

    for q in queries:
        found = False
        for hop, src in enumerate(sources, 1):
            _, score, hit = src.query(q)
            if hit:
                hits += 1
                found = True
                break
            if hop >= args.max_hops:
                ttl_exp += 1
                break

    print(f"  Compromised : {atk.num_compromised}/{args.nodes}")
    print(f"  Hit rate    : {hits/args.queries*100:.1f}%")
    print(f"  TTL exhaust : {ttl_exp/args.queries*100:.1f}%")
    print(f"  Drop rate   : {atk.effective_drop_rate*100:.1f}%  (effective)")
    bc = check_blockchain_status()
    print(f"  Blockchain  : chain_valid={bc['chain_valid']}  chain_len={bc['chain_length']}")


def run_detection(args):
    """Stealthy attack WITH anomaly detection — shows which nodes get flagged."""
    print(f"\n[detection] SFA ratio={args.ratio}  strategy={args.strategy}")
    rng      = np.random.default_rng(args.seed)
    sources  = _make_network(args.nodes, args.seed)
    detector = SFADetector()
    for src in sources:
        detector.register(src.node_id)

    atk = SelectiveForwardingAttack(
        attack_ratio=args.ratio,
        strategy=args.strategy,
        drop_rate="stealthy",
        seed=args.seed + 999,
    )
    ssm_scores = {s.node_id: _LEDGER.score(s.node_id) for s in sources}
    atk.apply(sources, ssm_scores)

    queries = _get_queries(args.queries, rng)
    hits    = 0

    for q in queries:
        for hop, src in enumerate(sources, 1):
            _, score, hit = src.query(q)
            detector.observe(src.node_id, hit)
            if hit:
                hits += 1
                break
            if hop >= args.max_hops:
                break

    det_report = detector.report()
    suspected  = [nid for nid, v in det_report.items() if v["suspected_drop"]]

    print(f"  Hit rate       : {hits/args.queries*100:.1f}%")
    print(f"  Compromised    : {atk.compromised_nodes}")
    print(f"  Suspected      : {suspected}")
    print(f"  Detection rate : {len(suspected)}/{atk.num_compromised} compromised nodes caught")
    print("\n  Per-node report:")
    for nid, v in sorted(det_report.items()):
        flag = " *** SUSPECTED" if v["suspected_drop"] else ""
        print(f"    {nid}  miss={v['miss_rate']:.2f}  susp_lvl={v['suspicion']}{flag}")


def run_mitigation(args):
    """Full pipeline — attack + detection + SSM-aware re-routing."""
    print(f"\n[mitigation] SFA ratio={args.ratio}  strategy={args.strategy}")
    rng      = np.random.default_rng(args.seed)
    sources  = _make_network(args.nodes, args.seed)
    detector = SFADetector()
    for src in sources:
        detector.register(src.node_id)

    atk = SelectiveForwardingAttack(
        attack_ratio=args.ratio,
        strategy=args.strategy,
        drop_rate="stealthy",
        seed=args.seed + 999,
    )
    ssm_scores = {s.node_id: _LEDGER.score(s.node_id) for s in sources}
    atk.apply(sources, ssm_scores)

    mitigation = SFAMitigation(sources, detector, _LEDGER, redundancy_k=3)
    queries    = _get_queries(args.queries, rng)

    hits = backup_hits = 0
    for q in queries:
        hit, hops, log = mitigation.route(q, max_hops=args.max_hops)
        if hit:
            hits += 1
            if any("BACKUP" in l for l in log):
                backup_hits += 1

    bc = check_blockchain_status()
    print(f"  Hit rate       : {hits/args.queries*100:.1f}%  (backup rescued {backup_hits})")
    print(f"  Blacklisted    : {mitigation.blacklisted_ids()}")
    print(f"  Suspected      : {mitigation.suspected_ids()}")
    print(f"  Active nodes   : {mitigation.active_count()}/{args.nodes}")
    print(f"  Blockchain     : valid={bc['chain_valid']}  len={bc['chain_length']}  "
          f"blacklisted={bc['num_blacklisted']}  avg_score={bc['avg_score']}")


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Selective Forwarding Attack runner")
    p.add_argument("--mode",     default="mock",
                   choices=["mock", "stealthy", "detection", "mitigation"],
                   help="Execution mode")
    p.add_argument("--ratio",    type=float, default=0.34, help="Attack ratio 0-1")
    p.add_argument("--strategy", default="random",
                   choices=["random", "high_ssm_score"], help="Node selection strategy")
    p.add_argument("--queries",  type=int, default=500,  help="Number of queries")
    p.add_argument("--nodes",    type=int, default=12,   help="Number of source nodes")
    p.add_argument("--max-hops", type=int, default=5,    dest="max_hops")
    p.add_argument("--seed",     type=int, default=42)
    args = p.parse_args()

    t0 = time.perf_counter()
    dispatch = {
        "mock":       run_mock,
        "stealthy":   run_stealthy,
        "detection":  run_detection,
        "mitigation": run_mitigation,
    }
    dispatch[args.mode](args)
    print(f"\n  Elapsed: {time.perf_counter()-t0:.2f}s\n")


if __name__ == "__main__":
    main()
