#!/usr/bin/env python3
"""
run_full_evaluation.py
======================
Single entry point for the FedRAG Security Evaluation Framework.

Architecture: True Per-Client Local InMemoryKnowledgeStore
-----------------------------------------------------------
Each federated client owns its own private local knowledge store.
Queries are answered by fanning out to every active client's local store
and aggregating the globally best answer at the server — matching the
FedRAG paper's architectural table (private per-client, non-replicated stores).

Attack flow (data poisoning example with 150 clients, 50 malicious):
  1. 50 clients are selected as malicious (--malicious-ratio 0.33).
  2. Each malicious client independently poisons ITS OWN local store.
  3. The server evaluates by querying ALL 150 client stores (50 poisoned + 100 clean).
  4. Server-side defense inspects each client's data and quarantines bad actors.
  5. Final evaluation excludes quarantined clients.

No GPU · No Hugging Face downloads · No Ollama server required.

Usage
-----
# Default: 10 clients, 200 examples, all attacks, defenses ON
python run_full_evaluation.py

# Scale up — 150 clients, 1500 examples, 50 malicious (1/3 ratio)
python run_full_evaluation.py --num-clients 150 --num-examples 1500 \\
    --malicious-ratio 0.33 --poisoning-ratio 0.5

# All attacks without defenses (measure raw impact)
python run_full_evaluation.py --no-defense

# Single attack
python run_full_evaluation.py --attack data_poisoning

# Custom scenario
python run_full_evaluation.py \\
    --num-clients 20 \\
    --num-examples 500 \\
    --seed 42 \\
    --malicious-ratio 0.25 \\
    --poisoning-ratio 0.3 \\
    --quarantine-threshold 0.25 \\
    --poison-type answer_swap \\
    --node-attack-type byzantine \\
    --output-dir my_results/

# Use your own JSONL dataset (query / response / topic columns required)
python run_full_evaluation.py --dataset-jsonl path/to/data.jsonl

Output
------
results/
  security_report.json   — raw results for every attack
  SECURITY_REPORT.md     — human-readable Markdown summary
  attack_matrix.csv      — flat CSV (one row per attack)
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
    hdr = (
        f"  │  {'Client':>6}  {'Role':^9}  {'Baseline F1':>11}  "
        f"{'Attacked F1':>11}  {'Δ F1':>8}  {'Defended F1':>11}  {'Recovery':>8}  │"
    )
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
        description="FedRAG Security Evaluation Framework — per-client local stores",
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
        help="Fraction of clients that are malicious (default: 0.2). "
             "E.g. 0.33 means 50 of 150 clients are poisoned.",
    )
    p.add_argument(
        "--poisoning-ratio", type=float, default=0.2,
        help="Fraction of each malicious client's LOCAL data to poison (default: 0.2)",
    )
    p.add_argument(
        "--quarantine-threshold", type=float, default=0.25,
        help="Defense sensitivity: risk score to trigger client quarantine (default: 0.25)",
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

    repo_root = Path(__file__).parent.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    src_path = repo_root / "src"
    if src_path.is_dir() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from security_framework import FedRAGSimulator, SecurityPipeline

    if not args.quiet:
        n_malicious = max(1, int(args.num_clients * args.malicious_ratio))
        print("\n" + "=" * 68)
        print("  FedRAG Security Evaluation — True Per-Client Local Stores")
        print("=" * 68)
        print(f"  Clients          : {args.num_clients}")
        print(f"  Examples         : {args.num_examples}")
        print(f"  Seed             : {args.seed}")
        print(f"  Defenses         : {'OFF' if args.no_defense else 'ON'}")
        print(f"  Attack(s)        : {args.attack}")
        print(f"  Malicious clients: {n_malicious}/{args.num_clients} "
              f"({args.malicious_ratio:.0%}) — each poisons ITS OWN local store")
        print(f"  Poisoning ratio  : {args.poisoning_ratio:.0%} of each client's data")
        print(f"  Quarantine thresh: {args.quarantine_threshold}")
        print(f"  Dataset          : {args.dataset_jsonl or 'synthetic'}")
        print("=" * 68 + "\n")

    sim = FedRAGSimulator(
        num_clients=args.num_clients,
        num_examples=args.num_examples,
        seed=args.seed,
        dataset_jsonl=args.dataset_jsonl,
    )

    pipeline = SecurityPipeline(
        sim,
        malicious_ratio=args.malicious_ratio,
        poisoning_ratio=args.poisoning_ratio,
        poison_type=args.poison_type,
        quarantine_threshold=args.quarantine_threshold,
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
