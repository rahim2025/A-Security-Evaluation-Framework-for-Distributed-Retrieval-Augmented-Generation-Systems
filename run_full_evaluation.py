#!/usr/bin/env python3
"""
run_full_evaluation.py
======================
Single entry point for the FedRAG Security Evaluation Framework.

This script requires NO GPU, NO Hugging Face model downloads, and NO Ollama
server.  It uses the built-in ``FedRAGSimulator`` (hash-based embeddings) to
demonstrate every attack and defense end-to-end.

Usage
-----
# Run all attacks with defenses enabled (default)
python run_full_evaluation.py

# Run all attacks WITHOUT defenses to measure raw attack effectiveness
python run_full_evaluation.py --no-defense

# Target a specific attack
python run_full_evaluation.py --attack data_poisoning

# Customise the simulation
python run_full_evaluation.py \\
    --num-clients 20 \\
    --num-examples 500 \\
    --seed 42 \\
    --poison-type answer_swap \\
    --node-attack-type byzantine \\
    --output-dir my_results/

# Load your own JSONL dataset (query / response / topic columns required)
python run_full_evaluation.py --dataset-jsonl path/to/data.jsonl

Output
------
results/
  security_report.json     — raw results for every attack
  SECURITY_REPORT.md       — human-readable Markdown summary
  attack_matrix.csv        — flat CSV (one row per attack)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _print_per_client_table(dp_result: dict) -> None:
    """Print a per-client F1 breakdown table for the data poisoning result."""
    breakdown = dp_result.get("per_client_breakdown")
    if not breakdown:
        return
    malicious_ids = set(dp_result.get("malicious_clients", []))
    ratio = dp_result.get("poisoning_ratio", "?")

    W = 90
    print()
    print("  ┌─ Data Poisoning: Per-Client F1 Breakdown " + "─" * (W - 44) + "┐")
    print(f"  │  Poisoning ratio = {ratio:.0%}   "
          f"✗ = malicious client   → = direction of change"
          + " " * (W - 60) + "│")
    hdr = f"  │  {'Client':>6}  {'Role':^9}  {'Baseline F1':>11}  "
    hdr += f"{'Attacked F1':>11}  {'Δ F1':>8}  {'Defended F1':>11}  {'Recovery':>8}  │"
    print("  ├" + "─" * (W - 2) + "┤")
    print(hdr)
    print("  ├" + "─" * (W - 2) + "┤")

    for cid in sorted(breakdown.keys()):
        b = breakdown[cid]
        role = "✗ MALICIOUS" if b["malicious"] else "  clean    "
        base = b["baseline_f1"]
        atk  = b["attacked_f1"]
        defe = b["defended_f1"]
        delta = b["delta_f1"]
        rec   = b["defense_recovery_f1"]
        arrow = "↓" if delta < -0.001 else ("↑" if delta > 0.001 else "─")
        delta_str = f"{arrow}{abs(delta):.4f}"
        rec_str = f"+{rec:.4f}" if rec > 0.0001 else f"{rec:.4f}"
        print(f"  │  {cid:>6}  {role:^9}  {base:>11.4f}  "
              f"{atk:>11.4f}  {delta_str:>8}  {defe:>11.4f}  {rec_str:>8}  │")

    print("  └" + "─" * (W - 2) + "┘")

    gb = dp_result.get("baseline", {}).get("f1", 0)
    ga = dp_result.get("attacked", {}).get("f1", 0)
    gd = (dp_result.get("defended") or {}).get("f1", 0)
    print(f"  Global system F1 : {gb:.4f} (baseline) → {ga:.4f} (attacked)  "
          f"Δ={ga-gb:+.4f}   Defended → {gd:.4f}   Recovery={gd-ga:+.4f}")
    print()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FedRAG Security Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--num-clients", type=int, default=10,
        help="Number of federated clients (default: 10)",
    )
    p.add_argument(
        "--num-examples", type=int, default=200,
        help="Total training examples before IID split (default: 200)",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (default: 0)",
    )
    p.add_argument(
        "--malicious-ratio", type=float, default=0.2,
        help="Fraction of clients that are malicious (default: 0.2)",
    )
    p.add_argument(
        "--poisoning-ratio", type=float, default=0.2,
        help="Fraction of malicious client data to poison (default: 0.2)",
    )
    p.add_argument(
        "--poison-type",
        choices=["wrong_answer", "misleading", "noise", "answer_swap"],
        default="wrong_answer",
        help="Poison type for data poisoning attack (default: wrong_answer)",
    )
    p.add_argument(
        "--membership-threshold", type=float, default=0.8,
        help="Similarity threshold for membership inference (default: 0.8)",
    )
    p.add_argument(
        "--extraction-top-k", type=int, default=3,
        help="Top-k retrieval for knowledge extraction (default: 3)",
    )
    p.add_argument(
        "--rate-limit", type=int, default=20,
        help="Max queries per client for rate-limiter defense (default: 20)",
    )
    p.add_argument(
        "--node-attack-type",
        choices=["node_removal", "byzantine", "partition", "ddos", "sybil"],
        default="node_removal",
        help="Node availability attack type (default: node_removal)",
    )
    p.add_argument(
        "--node-attack-ratio", type=float, default=0.3,
        help="Fraction of nodes to target (default: 0.3)",
    )
    p.add_argument(
        "--no-defense", action="store_true",
        help="Disable all defenses (measure raw attack effectiveness)",
    )
    p.add_argument(
        "--attack",
        choices=["data_poisoning", "membership_inference",
                 "knowledge_extraction", "node_availability", "all"],
        default="all",
        help="Which attack to run (default: all)",
    )
    p.add_argument(
        "--dataset-jsonl", type=str, default=None,
        help="Path to a JSONL dataset (query/response/topic). "
             "Uses synthetic data if not provided.",
    )
    p.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for reports (default: results/)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress messages",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Add repo root to path so security_framework is importable regardless
    # of where the script is invoked from.
    repo_root = Path(__file__).parent.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Also add the installed fed_rag package location so attacks/defenses work.
    src_path = repo_root / "src"
    if src_path.is_dir() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from security_framework import FedRAGSimulator, SecurityPipeline

    if not args.quiet:
        print("\n" + "=" * 60)
        print("  FedRAG Security Evaluation Framework")
        print("=" * 60)
        print(f"  Clients      : {args.num_clients}")
        print(f"  Examples     : {args.num_examples}")
        print(f"  Seed         : {args.seed}")
        print(f"  Defenses     : {'OFF' if args.no_defense else 'ON'}")
        print(f"  Attack(s)    : {args.attack}")
        print(f"  Dataset      : {args.dataset_jsonl or 'synthetic'}")
        print("=" * 60 + "\n")

    sim = FedRAGSimulator(
        num_clients=args.num_clients,
        num_examples=args.num_examples,
        seed=args.seed,
        dataset_jsonl=args.dataset_jsonl,
    )

    pipeline = SecurityPipeline(
        sim,
        malicious_ratio=args.malicious_ratio,
        quarantine_threshold=args.quarantine_threshold,
        quarantine_threshold=args.quarantine_threshold,
        poisoning_ratio=args.poisoning_ratio,
        poison_type=args.poison_type,
        membership_threshold=args.membership_threshold,
        extraction_top_k=args.extraction_top_k,
        rate_limit=args.rate_limit,
        node_attack_type=args.node_attack_type,
        node_attack_ratio=args.node_attack_ratio,
        defense_enabled=not args.no_defense,
    )

    if args.attack == "all":
        report = pipeline.run_all()
        report.print_summary()
        _print_per_client_table(report._results.get("data_poisoning", {}))
        report.save(args.output_dir)
    else:
        result = pipeline.run_attack(args.attack)
        import json
        print(json.dumps(result, indent=2))
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.attack}_result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(f"\nSaved to {out / f'{args.attack}_result.json'}")


if __name__ == "__main__":
    main()
