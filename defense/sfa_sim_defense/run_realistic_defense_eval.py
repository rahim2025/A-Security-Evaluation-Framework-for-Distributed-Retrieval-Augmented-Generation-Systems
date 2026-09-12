"""
defense/sfa_sim_defense/run_realistic_defense_eval.py

Validates SelectiveForwardingDefense (unmodified -- reused exactly as-is,
see class import below) against AdvancedSelectiveForwardingAttack's
realistic attacker variants (attack/selective_forward_sim/
realistic_attackers.py: delay, drift, collusion, adaptive-to-threshold),
per the NAACL improvement proposal's Selective Forwarding Attack item 3:
"Validate the reputation/blacklist and redundant-probe defense on live
traffic. Report false positives, detection delay, extra latency, network
cost, and how the defense behaves when honest peers naturally have low
relevance." See reports/updated_reports_safin/ for the full writeup this
belongs to, including why this file exists despite this project's default
attack-only scoping (explicit user request, documented there).

Reuses two already-existing, unmodified classes rather than building new
detection/mitigation logic:
  - defense.sfa_sim_defense.selective_forwarding_defense.SelectiveForwardingDefense
  - attack.selective_forward_sim.realistic_attackers.AdvancedSelectiveForwardingAttack
This script is purely an EVALUATION HARNESS around both -- it adds
instrumentation (timing, per-query blacklist-state snapshots, known-
ground-truth false-positive computation), not new attack or defense
mechanisms.

Metrics reported, mapped directly to the proposal's asks
---------------------------------------------------------
  false_positives           -- peers the defense blacklisted that were
                                NOT actually compromised (ground truth is
                                known here, since this script controls the
                                attack). Computed for every scenario,
                                including the "honest peers, no attack at
                                all" stress test below.
  detection_delay_queries    -- for each ACTUALLY compromised peer, the
                                index (in top-level queries issued) at
                                which it first appears in
                                defense.blacklisted_peers. None if never
                                blacklisted during the run. Coarser than
                                an exact per-interaction count (this
                                script does not modify
                                SelectiveForwardingDefense to add a
                                blacklist-event callback -- it polls
                                `defense.blacklisted_peers` after every
                                top-level query instead), but requires no
                                changes to the reused defense class.
  extra_latency_seconds      -- (mean wall-clock time per query, defended)
                                minus (mean wall-clock time per query,
                                undefended) for the IDENTICAL attack
                                configuration and seed -- isolates the
                                defense's own overhead (redundant probes,
                                bypass checks) from the attack's own
                                injected delay (realistic_attackers.py's
                                delay_range), which is present in both the
                                defended and undefended runs equally.
  network_cost                -- defense.get_stats()'s total_bypasses,
                                redundant_probes, redundant_probe_hits,
                                plus mean_hops_per_query (defended vs.
                                undefended) as a coarser proxy for total
                                HTTP calls issued (mock mode: hops are
                                exactly peer-query calls; live mode: same,
                                against the real drag_data_source
                                containers).
  low_relevance_honest_peers  -- a SEPARATE scenario (no attack applied at
                                all) where every honest peer's hit
                                probability is deliberately set low
                                (--low_relevance_hit_prob, default 0.1)
                                instead of the normal baseline
                                (--peer_hit_prob, default 0.4-0.95
                                depending on mode) -- a pure false-positive
                                stress test for the exact ambiguity
                                SelectiveForwardingDefense's own docstring
                                names: "a single fixed absolute
                                blacklist_threshold cannot tell 'this peer
                                is actively dropping queries' apart from
                                'the whole population's natural response
                                rate is just low'." Any peer blacklisted
                                here is unconditionally a false positive.

Usage
-----
  # Mock, all four attacker variants + the low-relevance-honest stress test
  python defense/sfa_sim_defense/run_realistic_defense_eval.py --mode mock

  # Live, single variant, against the real 3-node deployment
  python defense/sfa_sim_defense/run_realistic_defense_eval.py --mode live --variant adaptive
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.selective_forward_sim.network_sim import MockRAGNetwork  # noqa: E402
from attack.selective_forward_sim.live_network import LiveRAGNetwork  # noqa: E402
from attack.selective_forward_sim.realistic_attackers import (  # noqa: E402
    AdvancedSelectiveForwardingAttack,
)
from defense.sfa_sim_defense.selective_forwarding_defense import (  # noqa: E402
    SelectiveForwardingDefense,
)

LOG_DIR = os.path.join(_ROOT, "defense_logs")
os.makedirs(LOG_DIR, exist_ok=True)

VARIANTS = ["delay", "drift", "collusion", "adaptive", "combined"]
# This project's documented multi-seed convention (.claude/CLAUDE.md).
DEFAULT_SEEDS = [0, 42, 123]


def build_network(args: argparse.Namespace, seed: int, hit_prob: Optional[float] = None):
    if args.mode == "mock":
        return MockRAGNetwork(
            num_peers=args.num_peers, num_attachments=args.num_attachments,
            num_query_neighbor=args.num_query_neighbor, query_ttl=args.max_ttl,
            peer_hit_prob=hit_prob if hit_prob is not None else args.peer_hit_prob,
            seed=seed,
        )
    return LiveRAGNetwork(query_ttl=args.max_ttl)


def build_attack(variant: Optional[str], args: argparse.Namespace, seed: int) -> Optional[AdvancedSelectiveForwardingAttack]:
    if variant is None:
        return None
    kwargs: Dict[str, Any] = dict(attack_ratio=args.ratio, drop_rate=args.drop_rate, seed=seed)
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


def build_defense_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "blacklist_threshold": args.blacklist_threshold,
        "min_queries_before_blacklist": args.min_queries_before_blacklist,
        "suspicion_threshold": args.suspicion_threshold,
        "reputation_decay": args.reputation_decay,
        "min_peers_for_validation": args.min_peers_for_validation,
        "redundancy_k": args.redundancy_k,
        "max_blacklist_fraction": args.max_blacklist_fraction,
        "detection_mode": args.detection_mode,
    }


def run_one(
    args: argparse.Namespace, seed: int, variant: Optional[str],
    with_defense: bool, hit_prob: Optional[float] = None,
) -> Dict[str, Any]:
    """One full sweep: build network, optionally apply attack, optionally
    apply defense, issue args.num_queries top-level queries, tear down."""
    network = build_network(args, seed, hit_prob=hit_prob)
    attack = build_attack(variant, args, seed)
    info = attack.apply(network, strategy=args.strategy) if attack else {}

    defense = SelectiveForwardingDefense(build_defense_config(args)) if with_defense else None
    if defense is not None:
        defense.apply(network)

    per_query_wall_times: List[float] = []
    blacklist_first_seen_at: Dict[int, int] = {}
    results = []
    for i in range(args.num_queries):
        t0 = time.perf_counter()
        r = network.topic_aware_query(f"query_{i}")
        per_query_wall_times.append(time.perf_counter() - t0)
        results.append(r)
        if defense is not None:
            for pid in defense.blacklisted_peers:
                if pid not in blacklist_first_seen_at:
                    blacklist_first_seen_at[pid] = i

    if attack:
        attack.revert(network)
    if defense is not None:
        defense.remove(network)

    total = len(results)
    answered = sum(1 for r in results if r.answer and r.is_query_hit)
    hit_rate = answered / total if total else 0.0
    avg_hops = sum(r.num_hops for r in results) / total if total else 0.0
    mean_latency = sum(per_query_wall_times) / total if total else 0.0

    compromised = set(attack.compromised_ids) if attack else set()
    defense_stats = defense.get_stats() if defense is not None else {}
    blacklisted = set(defense_stats.get("blacklisted_peers", []))
    false_positives = sorted(blacklisted - compromised)
    true_positives = sorted(blacklisted & compromised)
    detection_delay = {
        pid: blacklist_first_seen_at.get(pid) for pid in sorted(compromised)
    }

    return {
        "seed": seed, "variant": variant, "with_defense": with_defense,
        "honest_hit_prob": hit_prob if hit_prob is not None else args.peer_hit_prob,
        "hit_rate": hit_rate, "avg_hops_per_query": avg_hops, "mean_latency_seconds": mean_latency,
        "compromised_ids": sorted(compromised),
        "blacklisted_ids": sorted(blacklisted),
        "false_positives": false_positives,
        "true_positives": true_positives,
        "detection_delay_queries": detection_delay,
        "defense_stats": defense_stats,
        "attack_info": {k: v for k, v in info.items() if k != "per_peer_drop_rates"},
    }


def run_scenario(args: argparse.Namespace, variant: Optional[str], seeds: List[int],
                  hit_prob: Optional[float] = None) -> Dict[str, Any]:
    per_seed = []
    for seed in seeds:
        undefended = run_one(args, seed, variant, with_defense=False, hit_prob=hit_prob)
        defended = run_one(args, seed, variant, with_defense=True, hit_prob=hit_prob)
        per_seed.append({
            "seed": seed,
            "undefended": undefended,
            "defended": defended,
            "hit_rate_recovery": defended["hit_rate"] - undefended["hit_rate"],
            "extra_latency_seconds": defended["mean_latency_seconds"] - undefended["mean_latency_seconds"],
        })
        print(f"  seed={seed}  undefended hit_rate={undefended['hit_rate']:.3f}  "
              f"defended hit_rate={defended['hit_rate']:.3f}  "
              f"(recovery {defended['hit_rate']-undefended['hit_rate']:+.3f})  "
              f"FPs={defended['false_positives']}  "
              f"detection_delay={defended['detection_delay_queries']}")

    def _mean(key_path):
        vals = [_dig(r, key_path) for r in per_seed]
        return sum(vals) / len(vals) if vals else 0.0

    def _dig(d, path):
        for k in path:
            d = d[k]
        return d

    summary = {
        "variant": variant,
        "n_seeds": len(seeds),
        "mean_hit_rate_recovery": _mean(["hit_rate_recovery"]),
        "mean_extra_latency_seconds": _mean(["extra_latency_seconds"]),
        "total_false_positives": sum(len(r["defended"]["false_positives"]) for r in per_seed),
        "total_true_positives": sum(len(r["defended"]["true_positives"]) for r in per_seed),
        "mean_bypasses": _mean(["defended", "defense_stats", "total_bypasses"]),
        "mean_redundant_probes": _mean(["defended", "defense_stats", "redundant_probes"]),
        "mean_redundant_probe_hits": _mean(["defended", "defense_stats", "redundant_probe_hits"]),
    }
    return {"per_seed": per_seed, "summary": summary}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mock", "live"], default="mock")
    p.add_argument("--variant", choices=VARIANTS + ["all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--num_peers", type=int, default=20)
    p.add_argument("--num_attachments", type=int, default=4)
    p.add_argument("--num_query_neighbor", type=int, default=4)
    p.add_argument("--max_ttl", type=int, default=6)
    p.add_argument("--peer_hit_prob", type=float, default=0.4)
    p.add_argument("--low_relevance_hit_prob", type=float, default=0.1,
                    help="honest-peer hit probability for the pure false-positive stress "
                         "test (no attack applied at all)")
    p.add_argument("--num_queries", type=int, default=200)
    p.add_argument("--strategy", choices=["random", "high_connectivity"], default="high_connectivity")
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--drop_rate", default="stealthy")
    p.add_argument("--delay_lo", type=float, default=0.05)
    p.add_argument("--delay_hi", type=float, default=0.5)
    p.add_argument("--drift_period", type=int, default=50)
    p.add_argument("--drift_amplitude", type=float, default=0.2)
    p.add_argument("--safety_margin", type=float, default=0.05)
    # Defense config -- mirrors config/sfa_sim_defense.yaml's field names/defaults exactly.
    p.add_argument("--blacklist_threshold", type=float, default=0.05)
    p.add_argument("--min_queries_before_blacklist", type=int, default=15)
    p.add_argument("--suspicion_threshold", type=float, default=0.10)
    p.add_argument("--reputation_decay", type=float, default=0.70)
    p.add_argument("--min_peers_for_validation", type=int, default=2)
    p.add_argument("--redundancy_k", type=int, default=2)
    p.add_argument("--max_blacklist_fraction", type=float, default=0.5)
    p.add_argument("--detection_mode", choices=["threshold", "binomial"], default="threshold")
    p.add_argument("--skip_low_relevance_test", action="store_true")
    args = p.parse_args()
    try:
        args.drop_rate = float(args.drop_rate)
    except ValueError:
        pass
    return args


def main() -> None:
    args = parse_args()
    variants = VARIANTS if args.variant == "all" else [args.variant]

    all_scenarios: Dict[str, Any] = {}
    for variant in variants:
        print(f"\n=== Defense validation: variant={variant} mode={args.mode} ===")
        all_scenarios[variant] = run_scenario(args, variant, args.seeds)

    if not args.skip_low_relevance_test:
        print(f"\n=== Defense validation: low-relevance HONEST peers, NO attack "
              f"(pure false-positive stress test, hit_prob={args.low_relevance_hit_prob}) ===")
        all_scenarios["low_relevance_honest_no_attack"] = run_scenario(
            args, variant=None, seeds=args.seeds, hit_prob=args.low_relevance_hit_prob,
        )

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, scenario in all_scenarios.items():
        s = scenario["summary"]
        print(f"  {name:<28s}  recovery={s['mean_hit_rate_recovery']:+.3f}  "
              f"extra_latency={s['mean_extra_latency_seconds']:+.4f}s  "
              f"FPs={s['total_false_positives']}  TPs={s['total_true_positives']}  "
              f"bypasses={s['mean_bypasses']:.1f}  redundant_probes={s['mean_redundant_probes']:.1f}")

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOG_DIR, f"sfa_realistic_defense_{args.mode}_{ts}.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump({"timestamp": ts, "config": vars(args), "scenarios": all_scenarios},
                   fh, indent=2, default=str)
    print(f"\nLog saved -> {log_path}")


if __name__ == "__main__":
    main()
