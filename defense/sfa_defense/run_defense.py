"""
defense/sfa_defense/run_defense.py

Evaluate the Selective Forwarding (SFA) defense against the live Reliable-dRAG
data sources, using the same HF-SQuAD-seeded workload as attack/selective_forward
(rajpurkar/squad train split, filtered to the questions actually answerable
from the running corpus -- see load_squad_questions()).

The detector (EWMA + binomial anomaly test) and mitigation (suspicion-aware
re-routing + blacklist + redundancy fallback) already exist in
attack/selective_forward/selective_forward_attack.py (SFADetector,
SFAMitigation) -- this module doesn't reimplement them, it runs the existing
attack/selective_forward/run_attack.py pipeline modes back-to-back
(baseline -> stealthy attack -> detection -> mitigation) and reports one
consolidated before/after comparison, mirroring defense/ssm_defense and
defense/mia_defense.

Two fixes from the original version, both verified against the live
containers (see README.md):
  1. SFADetector's thresholds were calibrated for a generic many-node mock
     topology (honest miss rate ~68%), not the real 3-source deployment
     (measured honest miss rate: 0.000/180 real queries) -- detection could
     never fire regardless of attack strength. Recalibrated in
     selective_forward_attack.py; detection now genuinely catches
     compromised nodes given enough samples (n_questions >= ~100).
  2. Mitigated accuracy (SFAMitigation.route(), first-hit routing) isn't
     comparable to baseline/attacked accuracy (_evaluate(), aggregate-all-
     sources). This module now also runs naive_route() -- the same
     first-hit mechanism without suspicion-awareness -- as the fair
     "undefended routing" comparison point for mitigation. That gap only
     shows up when the hop budget is below the source count (see the
     dedicated hop-limited phase below); at max_hops == total sources,
     every routing strategy eventually tries everyone, so there's nothing
     to isolate.

Usage
-----
  python defense/sfa_defense/run_defense.py
  python defense/sfa_defense/run_defense.py --ratio 0.34 --strategy high_ssm_score
  python defense/sfa_defense/run_defense.py --seed 0   # repeat --seed 1, 2 for thesis variance
  python defense/sfa_defense/run_defense.py --n_questions 150   # more margin above the detector's 40-query warm-up window
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.selective_forward.run_attack import (  # noqa: E402
    API_KEY,
    check_blockchain_status,
    run_baseline,
    run_detection,
    run_mitigation,
    run_stealthy,
)

LOG_DIR = os.path.join(_ROOT, "defense_logs")
os.makedirs(LOG_DIR, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFA Defense evaluation | Reliable-dRAG")
    p.add_argument("--ratio", type=float, default=0.34, help="Attack ratio 0-1")
    p.add_argument("--strategy", default="random", choices=["random", "high_ssm_score"])
    p.add_argument("--n_questions", type=int, default=100,
                   help="SFADetector needs >= 40 observations per node to warm up; "
                        "default raised from the attack module's 50 for a safer margin")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max_hops", type=int, default=3)
    p.add_argument("--seed", type=int, default=42, help="Use 0,1,2 for thesis variance")
    p.add_argument("--api_key", default=API_KEY)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 62)
    print("  SFA Defense Evaluation | Reliable-dRAG")
    print("=" * 62)
    print(f"  Attack ratio  : {args.ratio}")
    print(f"  Strategy      : {args.strategy}")
    print(f"  Questions     : {args.n_questions}")
    print(f"  Seed          : {args.seed}")

    # run_baseline/run_stealthy/run_detection/run_mitigation all read the
    # same attribute names off an args namespace -- reuse one shared object.
    ns = SimpleNamespace(
        ratio=args.ratio, strategy=args.strategy, n_questions=args.n_questions,
        k=args.k, max_hops=args.max_hops, seed=args.seed, api_key=args.api_key,
        nodes=12,
    )

    print("\n" + "-" * 62)
    print("  [1/5] Baseline (no attack)")
    print("-" * 62)
    baseline = run_baseline(ns)

    print("\n" + "-" * 62)
    print("  [2/5] Stealthy attack, undefended (no detection/mitigation)")
    print("-" * 62)
    undefended = run_stealthy(ns)

    print("\n" + "-" * 62)
    print("  [3/5] Attack + EWMA/binomial detection")
    print("-" * 62)
    detection = run_detection(ns)

    print("\n" + "-" * 62)
    print(f"  [4/5] Attack + detection + mitigation (max_hops={args.max_hops}, current deployment config)")
    print("-" * 62)
    mitigation = run_mitigation(ns)

    # At max_hops == total sources (today's actual deployment: n_retrievers=3
    # for 3 sources), every routing strategy eventually tries everyone, so
    # naive and mitigated routing converge and there's nothing to isolate --
    # confirmed empirically. The PDF's threat model ("system prioritises
    # high-scoring sources... forwarding-dropper can cause maximum damage")
    # only bites when the hop budget is below the source count, so also run
    # a dedicated hop-limited comparison to show the routing intelligence's
    # actual value.
    print("\n" + "-" * 62)
    print("  [5/5] Attack + mitigation, hop-limited (max_hops=1, worst-case routing scenario)")
    print("-" * 62)
    ns_hop_limited = SimpleNamespace(**{**vars(ns), "max_hops": 1})
    mitigation_hop_limited = run_mitigation(ns_hop_limited)

    bc = check_blockchain_status()

    baseline_acc = (baseline or {}).get("metrics", {}).get("accuracy")
    undefended_acc = (undefended or {}).get("attacked")
    detection_acc = (detection or {}).get("accuracy")
    detection_rate = (detection or {}).get("detection_rate")
    mitigation_acc = (mitigation or {}).get("accuracy")
    naive_acc = (mitigation or {}).get("naive_routing_accuracy")
    hop_mitigation_acc = (mitigation_hop_limited or {}).get("accuracy")
    hop_naive_acc = (mitigation_hop_limited or {}).get("naive_routing_accuracy")

    print("\n" + "=" * 62)
    print("  SFA Defense — Summary")
    print("=" * 62)
    print(f"  Baseline accuracy (no attack)              : {baseline_acc}")
    print(f"  Undefended accuracy (attack only)          : {undefended_acc}")
    print(f"  Detection-only accuracy                    : {detection_acc}")
    print(f"  Detection rate (compromised caught)        : {detection_rate}")
    print(f"  --- at max_hops={args.max_hops} (current deployment config) ---")
    print(f"  Naive routing accuracy, same attack        : {naive_acc}")
    print(f"  Mitigated routing accuracy                 : {mitigation_acc}")
    print(f"  --- at max_hops=1 (hop-limited worst case) ---")
    print(f"  Naive routing accuracy, same attack        : {hop_naive_acc}")
    print(f"  Mitigated routing accuracy                 : {hop_mitigation_acc}")
    print(f"  Blacklisted nodes                          : {(mitigation or {}).get('blacklisted')}")
    print(f"  Suspected nodes                             : {(mitigation or {}).get('suspected')}")
    print(f"  Blockchain ledger                          : valid={bc.get('chain_valid')}  "
          f"blacklisted={bc.get('num_blacklisted')}")
    print("=" * 62)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log: Dict[str, Any] = {
        "defense_type": "selective_forwarding_mitigation",
        "timestamp": ts,
        "config": vars(args),
        "baseline": baseline,
        "undefended_attack": undefended,
        "detection": detection,
        "mitigation": mitigation,
        "mitigation_hop_limited": mitigation_hop_limited,
        "blockchain_status": bc,
    }
    log_path = os.path.join(LOG_DIR, f"defense_{ts}_sfa_seed{args.seed}.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)
    print(f"\n  Log saved -> {log_path}")


if __name__ == "__main__":
    main()
