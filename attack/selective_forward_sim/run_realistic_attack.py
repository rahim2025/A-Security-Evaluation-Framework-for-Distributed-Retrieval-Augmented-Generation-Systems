"""
attack/selective_forward_sim/run_realistic_attack.py

CLI runner for AdvancedSelectiveForwardingAttack (realistic_attackers.py).
Mirrors run_attack.py's mock/live mode split and config-loading pattern,
but sweeps the four new realism toggles instead of strategy x ratio.

Usage
-----
  # Mock, all four realism variants at one ratio/strategy (fast, no Docker)
  python attack/selective_forward_sim/run_realistic_attack.py --mode mock

  # Just the adaptive-to-threshold attacker, larger network
  python attack/selective_forward_sim/run_realistic_attack.py --mode mock \
      --num_peers 30 --variant adaptive --blacklist_threshold 0.05

  # Live, against the real 3-node deployment
  python attack/selective_forward_sim/run_realistic_attack.py --mode live --variant collusion
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.selective_forward_sim.network_sim import MockRAGNetwork  # noqa: E402
from attack.selective_forward_sim.live_network import LiveRAGNetwork  # noqa: E402
from attack.selective_forward_sim.realistic_attackers import (  # noqa: E402
    AdvancedSelectiveForwardingAttack,
)

LOG_DIR = os.path.join(_ROOT, "attack_logs")
os.makedirs(LOG_DIR, exist_ok=True)

VARIANTS = ["delay", "drift", "collusion", "adaptive", "combined"]


def build_attack(variant: str, args: argparse.Namespace) -> AdvancedSelectiveForwardingAttack:
    kwargs: Dict[str, Any] = dict(attack_ratio=args.ratio, drop_rate=args.drop_rate, seed=args.seed)
    if variant in ("delay", "combined"):
        kwargs["delay_range"] = (args.delay_lo, args.delay_hi)
    if variant in ("drift", "combined"):
        kwargs["drift"] = {"period_queries": args.drift_period, "amplitude": args.drift_amplitude}
    if variant in ("collusion", "combined"):
        kwargs["collusion"] = True
    if variant in ("adaptive", "combined"):
        kwargs["adaptive_to_threshold"] = {
            "blacklist_threshold": args.blacklist_threshold, "safety_margin": args.safety_margin,
        }
    return AdvancedSelectiveForwardingAttack(**kwargs)


def run_variant(variant: str, args: argparse.Namespace) -> Dict[str, Any]:
    if args.mode == "mock":
        network = MockRAGNetwork(
            num_peers=args.num_peers, num_attachments=args.num_attachments,
            num_query_neighbor=args.num_query_neighbor, query_ttl=args.max_ttl,
            peer_hit_prob=args.peer_hit_prob, seed=args.seed,
        )
    else:
        network = LiveRAGNetwork(query_ttl=args.max_ttl)

    attack = build_attack(variant, args)
    info = attack.apply(network, strategy=args.strategy)

    results = []
    for i in range(args.num_queries):
        q = f"query_{i}"
        start = network.peers[0].peer_id if args.mode == "mock" else None
        r = network.topic_aware_query(q) if args.mode == "live" else network.topic_aware_query(q, start=None)
        results.append(r)
    attack.revert(network)

    metrics = attack.collect_metrics(results, args.max_ttl)
    metrics["variant"] = variant
    metrics["final_response_rate_per_compromised_peer"] = attack.per_peer_final_response_rate()
    metrics["mean_injected_delay"] = (
        sum(attack.injected_delays) / len(attack.injected_delays) if attack.injected_delays else 0.0
    )
    metrics["info"] = info
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mock", "live"], default="mock")
    p.add_argument("--variant", choices=VARIANTS + ["all"], default="all")
    p.add_argument("--num_peers", type=int, default=20)
    p.add_argument("--num_attachments", type=int, default=4)
    p.add_argument("--num_query_neighbor", type=int, default=4)
    p.add_argument("--max_ttl", type=int, default=6)
    p.add_argument("--peer_hit_prob", type=float, default=0.4)
    p.add_argument("--num_queries", type=int, default=200)
    p.add_argument("--strategy", choices=["random", "high_connectivity"], default="high_connectivity")
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--drop_rate", default="stealthy", help="float in [0,1], or 'stealthy'")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay_lo", type=float, default=0.05)
    p.add_argument("--delay_hi", type=float, default=0.5)
    p.add_argument("--drift_period", type=int, default=50)
    p.add_argument("--drift_amplitude", type=float, default=0.2)
    p.add_argument("--blacklist_threshold", type=float, default=0.05,
                    help="must match defense/sfa_sim_defense's configured value for a fair test")
    p.add_argument("--safety_margin", type=float, default=0.05)
    args = p.parse_args()
    try:
        args.drop_rate = float(args.drop_rate)
    except ValueError:
        pass  # keep as "stealthy"
    return args


def main() -> None:
    args = parse_args()
    variants = VARIANTS if args.variant == "all" else [args.variant]

    all_results: List[Dict[str, Any]] = []
    for variant in variants:
        print(f"\n=== variant={variant} mode={args.mode} ===")
        result = run_variant(variant, args)
        print(json.dumps({k: v for k, v in result.items() if k != "info"}, indent=2, default=str))
        all_results.append(result)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOG_DIR, f"sfa_realistic_{args.mode}_{ts}.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump({"timestamp": ts, "config": vars(args), "results": all_results}, fh, indent=2, default=str)
    print(f"\nLog saved -> {log_path}")


if __name__ == "__main__":
    main()
