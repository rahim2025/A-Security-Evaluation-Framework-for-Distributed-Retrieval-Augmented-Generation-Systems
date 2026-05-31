#!/usr/bin/env python3
"""
run_client_degradation_sweep.py
================================
Sweep the poisoning ratio from 0 to 100% and show exactly how each client's
local F1 AND the entire global system F1 degrade step by step.

This script gives you the full picture:
  • Which clients are most vulnerable?
  • At what poisoning ratio does the system start to visibly fail?
  • How does the defense hold up as poisoning intensity increases?

No GPU · No Hugging Face downloads · No Ollama required.

USAGE
-----
  # Full sweep (0% → 100% in 10 steps, all clients poisoned)
  python run_client_degradation_sweep.py

  # Sweep only 3 poison ratios
  python run_client_degradation_sweep.py --ratios 0.2 0.5 0.8

  # Choose which clients are malicious (comma-separated IDs)
  python run_client_degradation_sweep.py --malicious-clients 0 2 5

  # Use a malicious ratio instead of explicit IDs
  python run_client_degradation_sweep.py --malicious-ratio 0.4

  # Scale up
  python run_client_degradation_sweep.py --num-clients 20 --num-examples 500

OUTPUT
------
  results/sweep/
    degradation_sweep.csv      ← ratio × per-client F1 + global F1 (Excel-ready)
    degradation_sweep.md       ← human-readable Markdown table
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_REPO = Path(__file__).parent.resolve()
for _p in (_REPO, _REPO / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _bar(f1: float, width: int = 20) -> str:
    filled = round(f1 * width)
    return "█" * filled + "░" * (width - filled)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FedRAG — Per-Client & Global F1 Degradation Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--num-clients",  type=int,   default=10)
    p.add_argument("--num-examples", type=int,   default=200)
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument(
        "--ratios", nargs="+", type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Poisoning ratios to sweep (default: 0.0 → 1.0 in 0.1 steps)",
    )
    p.add_argument(
        "--malicious-clients", nargs="+", type=int, default=None,
        help="Explicit list of client IDs to poison (overrides --malicious-ratio)",
    )
    p.add_argument(
        "--malicious-ratio", type=float, default=0.5,
        help="Fraction of clients to poison when --malicious-clients not given (default: 0.5)",
    )
    p.add_argument("--output-dir", type=str, default="results/sweep")
    p.add_argument("--dataset-jsonl", type=str, default=None)
    p.add_argument("--no-defense", action="store_true",
                   help="Skip defense evaluation (faster)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from security_framework.simulator import FedRAGSimulator
    from security_framework.global_impact import (
        _evaluate_global,
        _evaluate_per_client_global,
    )
    from fed_rag.attacks import DataPoisoningAttack
    from fed_rag.defenses import ClientDataPoisoningDefense

    dataset_path = None
    if args.dataset_jsonl:
        p = Path(args.dataset_jsonl)
        if not p.exists():
            print(f"\n  ERROR: Dataset file not found: {args.dataset_jsonl}")
            print(f"  Omit --dataset-jsonl to use synthetic data.\n")
            sys.exit(1)
        dataset_path = str(p)

    sim = FedRAGSimulator(
        num_clients=args.num_clients,
        num_examples=args.num_examples,
        seed=args.seed,
        dataset_jsonl=dataset_path,
    )

    import random
    rng = random.Random(args.seed)

    if args.malicious_clients is not None:
        malicious_ids = sorted(args.malicious_clients)
    else:
        n_mal = max(1, int(sim.num_clients * args.malicious_ratio))
        malicious_ids = sorted(rng.sample(range(sim.num_clients), n_mal))

    dataset = sim.dataset
    clients = sim.clients

    W = 80
    print()
    print("╔" + "═" * (W - 2) + "╗")
    print("║" + "  FedRAG — Per-Client & Global F1 Degradation Sweep".center(W - 2) + "║")
    print("╠" + "═" * (W - 2) + "╣")
    print(f"║  Clients       : {sim.num_clients:<{W-20}}║")
    print(f"║  Examples      : {sim.num_examples:<{W-20}}║")
    print(f"║  Seed          : {args.seed:<{W-20}}║")
    print(f"║  Malicious IDs : {str(malicious_ids):<{W-20}}║")
    print(f"║  Ratios        : {str(args.ratios):<{W-20}}║")
    print(f"║  Defense       : {'OFF' if args.no_defense else 'ON':<{W-20}}║")
    print("╚" + "═" * (W - 2) + "╝")
    print()

    baseline_global = _evaluate_global(dataset, clients, sim.embedding_dim)
    baseline_per_client = _evaluate_per_client_global(dataset, clients, sim.embedding_dim)

    print(f"  Baseline → Global F1 = {baseline_global['f1']:.4f}")
    for cid in sorted(baseline_per_client):
        f1 = baseline_per_client[cid]["f1"]
        role = "✗MAL" if cid in malicious_ids else "  ok"
        print(f"    Client {cid:>2} [{role}]  F1={f1:.4f}  {_bar(f1)}")
    print()

    all_rows: list[dict] = []

    for ratio in sorted(args.ratios):
        t0 = time.perf_counter()
        orig_examples = {c.client_id: list(c.examples) for c in clients}

        for cid in malicious_ids:
            if ratio == 0.0:
                continue
            atk = DataPoisoningAttack(
                poisoning_ratio=ratio,
                poison_type="wrong_answer",
                mode="replace",
                amplification_factor=2,
                seed=args.seed + cid,
            )
            result = atk.execute(clients[cid].examples)
            clients[cid].examples = result.poisoned_examples

        sim.rebuild_stores()
        attacked_global = _evaluate_global(dataset, clients, sim.embedding_dim)
        attacked_per = _evaluate_per_client_global(dataset, clients, sim.embedding_dim)

        defended_global = None
        defended_per: dict = {}
        if not args.no_defense and ratio > 0.0:
            all_data = [c.examples for c in clients]
            defense = ClientDataPoisoningDefense(quarantine_threshold=0.25)
            defended_data, inspections = defense.sanitize(all_data)
            quarantined = [r.client_id for r in inspections if r.quarantined]
            for i, examples in enumerate(defended_data):
                clients[i].examples = examples
            sim.rebuild_stores()
            defended_global = _evaluate_global(dataset, clients, sim.embedding_dim)
            defended_per = _evaluate_per_client_global(dataset, clients, sim.embedding_dim)

        for c in clients:
            c.examples = orig_examples[c.client_id]
        sim.rebuild_stores()

        runtime = time.perf_counter() - t0

        row: dict = {
            "poisoning_ratio": ratio,
            "global_baseline_f1": round(baseline_global["f1"], 4),
            "global_attacked_f1": round(attacked_global["f1"], 4),
            "global_delta_f1":    round(attacked_global["f1"] - baseline_global["f1"], 4),
        }
        if defended_global:
            row["global_defended_f1"]  = round(defended_global["f1"], 4)
            row["global_recovery_f1"]  = round(defended_global["f1"] - attacked_global["f1"], 4)

        for cid in sorted(attacked_per):
            role = "MAL" if cid in malicious_ids else "clean"
            row[f"c{cid}_{role}_baseline_f1"] = round(baseline_per_client[cid]["f1"], 4)
            row[f"c{cid}_{role}_attacked_f1"] = round(attacked_per[cid]["f1"], 4)
            row[f"c{cid}_{role}_delta_f1"]    = round(
                attacked_per[cid]["f1"] - baseline_per_client[cid]["f1"], 4)
            if defended_per:
                row[f"c{cid}_{role}_defended_f1"] = round(defended_per[cid]["f1"], 4)

        all_rows.append(row)

        g_atk = attacked_global["f1"]
        g_def = defended_global["f1"] if defended_global else None
        def_str = f"  →  Defended: {g_def:.4f}" if g_def is not None else ""
        print(f"  ratio={ratio:.0%}  Global F1: {baseline_global['f1']:.4f} → {g_atk:.4f}"
              f"  Δ={g_atk-baseline_global['f1']:+.4f}{def_str}  ({runtime:.2f}s)")

        for cid in sorted(attacked_per):
            a_f1 = attacked_per[cid]["f1"]
            b_f1 = baseline_per_client[cid]["f1"]
            d_f1 = defended_per[cid]["f1"] if defended_per else None
            role  = "✗MAL" if cid in malicious_ids else "  ok"
            delta = a_f1 - b_f1
            bar_b = _bar(b_f1)
            bar_a = _bar(a_f1)
            def_part = f"  defended={d_f1:.4f}" if d_f1 is not None else ""
            print(f"    Client {cid:>2} [{role}]  "
                  f"base={b_f1:.4f} [{bar_b}]  "
                  f"atk={a_f1:.4f} [{bar_a}]  Δ={delta:+.4f}{def_part}")
        print()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "degradation_sweep.csv"
    all_fields: list[str] = []
    seen: set[str] = set()
    for row in all_rows:
        for k in row:
            if k not in seen:
                all_fields.append(k)
                seen.add(k)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Wrote {csv_path}")

    md_path = out / "degradation_sweep.md"
    lines = [
        "# FedRAG — Per-Client & Global F1 Degradation Sweep",
        "",
        f"**Clients:** {sim.num_clients}  |  "
        f"**Examples:** {sim.num_examples}  |  "
        f"**Seed:** {args.seed}",
        f"**Malicious clients:** {malicious_ids}",
        "",
        "## Global System F1 vs Poisoning Ratio",
        "",
        "| Ratio | Baseline F1 | Attacked F1 | Δ F1 | Defended F1 | Recovery |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in all_rows:
        def_f1 = row.get("global_defended_f1", "–")
        rec     = row.get("global_recovery_f1", "–")
        lines.append(
            f"| {row['poisoning_ratio']:.0%} "
            f"| {row['global_baseline_f1']:.4f} "
            f"| {row['global_attacked_f1']:.4f} "
            f"| {row['global_delta_f1']:+.4f} "
            f"| {def_f1} "
            f"| {rec} |"
        )
    lines += ["", "---", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {md_path}")

    print("\nDone.")
    print(f"  {csv_path}   ← open in Excel for full per-client × ratio breakdown")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
