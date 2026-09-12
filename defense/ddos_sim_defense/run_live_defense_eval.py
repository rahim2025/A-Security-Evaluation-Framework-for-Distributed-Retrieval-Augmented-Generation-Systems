"""
defense/ddos_sim_defense/run_live_defense_eval.py

Closes the gap reports/ddos_attack.md section 13 names explicitly: "A
countermeasure exists and measurably recovers a meaningful fraction of
lost hit-rate, but it operates only on the simulation layer, not the live
deployment." This script runs a REAL concurrent flood
(attack/ddos_sim/live_flood.TrafficFlood, reused unmodified) against the
real drag_data_source containers while three defense conditions are
compared on the SAME live traffic, at the SAME severity:

  - no_defense            -- baseline, nothing installed
  - ddos_defense           -- the EXISTING defense/ddos_sim_defense/
                              DDoSDefense (reputation/blacklist/bypass/
                              redundant-probe), reused unmodified, now
                              actually tested against a real flood for
                              the first time
  - live_client_defense    -- the NEW defense/ddos_sim_defense/
                              live_client_defense.LiveClientDefense
                              (rate limiting, circuit breaker, queue
                              limit, caching), see that module's docstring

Two measurement layers, mirroring attack/ddos_sim/run_live_evaluation.py's
own two-layer pattern (hop_network for retrieval, query_llm for real
generation) -- IMPORTANT ASYMMETRY, stated up front rather than left to
be discovered in the numbers:

  1. Retrieval layer (`LiveRAGNetwork.topic_aware_query()`, defense
     installed via the existing `_sfa_defense` hook): hit_rate, avg_hops,
     mean latency, plus each defense's own get_stats() (false blocks,
     bypasses, redundant probes, defense overhead). This layer DOES
     benefit from whichever defense is installed -- it is the routing
     path the defense actually wraps.
  2. End-to-end generation layer (`query_llm()`, hitting the real
     `drag_llm_service` /query endpoint exactly like
     run_live_evaluation.py does): answer quality via
     attack.ddos_sim.nlg_metrics (reused unmodified). This layer does
     NOT benefit from either defense -- drag_llm_service makes its own,
     separate HTTP calls to the data sources, which neither DDoSDefense
     nor LiveClientDefense wraps (both only wrap THIS script's own
     LiveRAGNetwork client). Expect layer-2 numbers to be statistically
     indistinguishable across all three defense conditions; that is the
     expected, correctly-explained result of where these defenses' reach
     actually ends, not a bug. A defense that protected layer 2 as well
     would require wrapping drag_llm_service's own request path, i.e.
     modifying drag_llm_service/app/server.py -- deliberately out of
     scope (".claude/CLAUDE.md": do not modify the core system unless
     strictly necessary for instrumentation).

False blocks
-------------
Measured in a dedicated POST-recovery phase: after `flood.stop()` and a
cooldown, ground truth says every source should be healthy again. Any
query in this phase that the defense still bypasses (rate-limited,
circuit still open, blacklisted) is unambiguously a false block --
mirrors defense/sfa_sim_defense's false-positive framing (ground truth
known because this script controls the attack).

Usage
-----
  python defense/ddos_sim_defense/run_live_defense_eval.py --severity high
  python defense/ddos_sim_defense/run_live_defense_eval.py --defenses no_defense ddos_defense
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

import requests  # noqa: E402

from attack.selective_forward_sim.live_network import LiveRAGNetwork, DEFAULT_SOURCE_URLS  # noqa: E402
from attack.ddos_sim.live_flood import TrafficFlood  # noqa: E402
from attack.ddos_sim.nlg_metrics import score_answer, average_metrics  # noqa: E402
from attack.ddos_sim.run_live_evaluation import (  # noqa: E402
    LLM_SERVICE_URL, SEVERITY_TIERS, load_pubmedqa_qa_pairs, query_llm,
)
from defense.ddos_sim_defense.ddos_defense import DDoSDefense  # noqa: E402
from defense.ddos_sim_defense.live_client_defense import LiveClientDefense  # noqa: E402

LOG_DIR = os.path.join(_ROOT, "defense_logs")
os.makedirs(LOG_DIR, exist_ok=True)

DEFENSE_BUILDERS = {
    "no_defense": lambda args: None,
    "ddos_defense": lambda args: DDoSDefense({
        "deprioritize_threshold": args.deprioritize_threshold,
        "min_queries_before_action": args.min_queries_before_action,
        "redundancy_k": args.redundancy_k,
    }),
    "live_client_defense": lambda args: LiveClientDefense({
        "rate_limit_capacity": args.rate_limit_capacity,
        "rate_limit_refill_per_sec": args.rate_limit_refill_per_sec,
        "circuit_failure_threshold": args.circuit_failure_threshold,
        "circuit_open_seconds": args.circuit_open_seconds,
        "max_concurrent_per_peer": args.max_concurrent_per_peer,
        "cache_ttl_seconds": args.cache_ttl_seconds,
        "redundancy_k": args.redundancy_k,
    }),
}


def retrieval_phase(net: LiveRAGNetwork, questions: List[str], threshold: float = 0.5) -> Dict[str, Any]:
    latencies, hits, hops = [], 0, []
    for q in questions:
        t0 = time.monotonic()
        r = net.topic_aware_query(q, threshold)
        latencies.append(time.monotonic() - t0)
        hops.append(r.num_hops)
        if r.is_query_hit:
            hits += 1
    n = len(questions)
    return {
        "hit_rate": hits / n if n else 0.0,
        "avg_hops": sum(hops) / n if n else 0.0,
        "mean_latency_seconds": sum(latencies) / n if n else 0.0,
        "n_questions": n,
    }


def generation_phase(qa_pairs) -> Dict[str, Any]:
    per_question = []
    successes = 0
    for q, gold in qa_pairs:
        response, ok = query_llm(q)
        if ok:
            successes += 1
        metrics = score_answer(response, [gold], compute_semantic=False)
        per_question.append(metrics)
    n = len(qa_pairs)
    agg = average_metrics(per_question)
    agg["query_success_rate"] = successes / n if n else 0.0
    return agg


def run_one_defense(args: argparse.Namespace, defense_name: str, qa_pairs) -> Dict[str, Any]:
    print(f"\n{'=' * 62}\n  defense={defense_name}  severity={args.severity}\n{'=' * 62}")
    tier = SEVERITY_TIERS[args.severity]
    net = LiveRAGNetwork(use_onchain_scores=False, query_ttl=args.max_ttl)
    defense = DEFENSE_BUILDERS[defense_name](args)
    if defense is not None:
        defense.apply(net)

    print("  --- baseline (no flood) ---")
    baseline_retrieval = retrieval_phase(net, [q for q, _ in qa_pairs])
    baseline_generation = generation_phase(qa_pairs)
    print(f"    retrieval hit_rate={baseline_retrieval['hit_rate']:.3f}  "
          f"generation f1={baseline_generation.get('f1', 0):.3f}")

    print(f"  --- under flood ({tier['num_sources']}/3 sources x {tier['workers_per_source']} workers) ---")
    flood = TrafficFlood(DEFAULT_SOURCE_URLS, workers_per_source=tier["workers_per_source"])
    targets = list(DEFAULT_SOURCE_URLS.keys())[: tier["num_sources"]]
    flood.start(targets)
    time.sleep(args.flood_ramp_s)
    try:
        under_flood_retrieval = retrieval_phase(net, [q for q, _ in qa_pairs])
        under_flood_generation = generation_phase(qa_pairs)
    finally:
        flood_stats = flood.stop()
    print(f"    retrieval hit_rate={under_flood_retrieval['hit_rate']:.3f}  "
          f"generation f1={under_flood_generation.get('f1', 0):.3f}  flood_stats={flood_stats}")

    print(f"  --- post-recovery (cooldown {args.recovery_cooldown_s:.0f}s, false-block check) ---")
    time.sleep(args.recovery_cooldown_s)
    post_recovery_retrieval = retrieval_phase(net, [q for q, _ in qa_pairs])
    defense_stats = defense.get_stats() if defense is not None else {}
    false_blocks = (
        defense_stats.get("total_rate_limited", 0) + defense_stats.get("total_circuit_blocked", 0)
        + defense_stats.get("blacklisted_count", 0)
    ) if defense is not None else 0
    print(f"    post-recovery hit_rate={post_recovery_retrieval['hit_rate']:.3f}  "
          f"(any block here is a false block -- flood has stopped)")

    if defense is not None:
        defense.remove(net)

    return {
        "defense": defense_name,
        "baseline": {"retrieval": baseline_retrieval, "generation": baseline_generation},
        "under_flood": {"retrieval": under_flood_retrieval, "generation": under_flood_generation,
                         "flood_stats": flood_stats},
        "post_recovery": {"retrieval": post_recovery_retrieval},
        "defense_stats": defense_stats,
        "hit_rate_recovery_under_flood": under_flood_retrieval["hit_rate"] - baseline_retrieval["hit_rate"],
        "extra_latency_under_flood": under_flood_retrieval["mean_latency_seconds"] - baseline_retrieval["mean_latency_seconds"],
        "post_recovery_false_block_indicators": false_blocks,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--defenses", nargs="+", default=["no_defense", "ddos_defense", "live_client_defense"],
                    choices=list(DEFENSE_BUILDERS))
    p.add_argument("--severity", choices=list(SEVERITY_TIERS), default="high")
    p.add_argument("--num_questions", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_ttl", type=int, default=6)
    p.add_argument("--flood_ramp_s", type=float, default=1.0)
    p.add_argument("--recovery_cooldown_s", type=float, default=60.0)
    p.add_argument("--tier_cooldown_s", type=float, default=60.0,
                    help="cooldown between successive defense conditions, same rationale as "
                         "run_live_evaluation.py's --tier_cooldown_s")
    # DDoSDefense config
    p.add_argument("--deprioritize_threshold", type=float, default=0.5)
    p.add_argument("--min_queries_before_action", type=int, default=5)
    # LiveClientDefense config
    p.add_argument("--rate_limit_capacity", type=float, default=20)
    p.add_argument("--rate_limit_refill_per_sec", type=float, default=5.0)
    p.add_argument("--circuit_failure_threshold", type=int, default=5)
    p.add_argument("--circuit_open_seconds", type=float, default=15.0)
    p.add_argument("--max_concurrent_per_peer", type=int, default=3)
    p.add_argument("--cache_ttl_seconds", type=float, default=5.0)
    p.add_argument("--redundancy_k", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    qa_pairs = load_pubmedqa_qa_pairs(args.num_questions, args.seed)
    print(f"[+] {len(qa_pairs)} PubMedQA questions loaded")

    reachability = LiveRAGNetwork(use_onchain_scores=False).ping_all()
    print(f"[+] Source reachability: {reachability}")
    if not any(reachability.values()):
        print("[!] No Docker data-source nodes reachable. Start with: docker compose up -d")
        sys.exit(1)

    all_results: List[Dict[str, Any]] = []
    for i, defense_name in enumerate(args.defenses):
        if i > 0:
            print(f"\n[+] Cooling down {args.tier_cooldown_s:.0f}s before next defense condition...")
            time.sleep(args.tier_cooldown_s)
        all_results.append(run_one_defense(args, defense_name, qa_pairs))

    print("\n" + "=" * 62 + "\n  SUMMARY\n" + "=" * 62)
    for r in all_results:
        print(f"  {r['defense']:<20s}  recovery={r['hit_rate_recovery_under_flood']:+.3f}  "
              f"extra_latency={r['extra_latency_under_flood']:+.4f}s  "
              f"false_blocks(post-recovery)={r['post_recovery_false_block_indicators']}")

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOG_DIR, f"ddos_live_defense_eval_{ts}.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump({"timestamp": ts, "config": vars(args), "results": all_results}, fh, indent=2, default=str)
    print(f"\nLog saved -> {log_path}")


if __name__ == "__main__":
    main()
