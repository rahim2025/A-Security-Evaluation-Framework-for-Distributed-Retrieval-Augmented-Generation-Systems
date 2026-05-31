#!/usr/bin/env python3
"""
run_global_impact_analysis.py
==============================
Entry point for the FedRAG Global System Impact Analysis.

Architecture — True Per-Client Local InMemoryKnowledgeStore
-----------------------------------------------------------
Each federated client owns its own PRIVATE local knowledge store.
Queries are answered by fanning out to every active client's local store
and aggregating the globally best answer at the server.

This matches the FedRAG paper's architectural table:
  - private per-client stores (non-replicated)
  - server-side aggregation of per-client responses
  - per-client attack: each malicious client poisons only its own local store

Attack flow example (data poisoning with 150 clients, 50 malicious):
  1. 50 clients selected as malicious.
  2. Each malicious client independently poisons ITS OWN local store.
  3. Server evaluates by querying ALL 150 stores (50 poisoned + 100 clean).
  4. Defense inspects each client's data; quarantines bad actors.
  5. Final evaluation excludes quarantined clients.

No GPU · No Hugging Face downloads · No Ollama server required.

────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────

  # Full analysis — all attacks, all intensities
  python run_global_impact_analysis.py

  # Quick test: fewer clients/examples
  python run_global_impact_analysis.py --num-clients 6 --num-examples 60

  # Scale up — 150 clients, 50 poisoned (1/3 ratio)
  python run_global_impact_analysis.py \\
      --num-clients 150 --num-examples 1500 --intensities medium heavy

  # Only data poisoning
  python run_global_impact_analysis.py --attacks data_poisoning

  # Only medium intensity
  python run_global_impact_analysis.py --intensities medium

  # Use your own JSONL dataset (query / response / topic columns required)
  python run_global_impact_analysis.py --dataset-jsonl /path/to/data.jsonl

  # Custom output directory
  python run_global_impact_analysis.py --output-dir my_results/

────────────────────────────────────────────────────────────────────
OUTPUT FILES  (default: results/global/)
────────────────────────────────────────────────────────────────────

  global_impact_results.json       Full raw results for every scenario
  GLOBAL_IMPACT_REPORT.md          Human-readable Markdown report
  global_attack_matrix.csv         Flat CSV — one row per scenario
  cascade_data_poisoning.csv       F1 degradation as each new client is poisoned
  cascade_kb_extraction.csv        KB exposure as each new client is targeted
  cascade_node_availability.csv    F1 degradation as each node is removed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).parent.resolve()
for _p in (_REPO, _REPO / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _print_global_per_client_table(report: "GlobalImpactReport") -> None:  # type: ignore[name-defined]
    """Print a per-client F1 table for every data-poisoning intensity level."""
    dp_scenarios = [s for s in report.scenarios if s.attack_name == "Data Poisoning"]
    if not dp_scenarios:
        return

    W = 100
    print()
    print("  ╔" + "═" * (W - 2) + "╗")
    print("  ║" + "  Per-Client Breakdown  ·  Data Poisoning  ·  (each client evaluated in isolation)".center(W - 2) + "║")
    print("  ╚" + "═" * (W - 2) + "╝")

    for scenario in dp_scenarios:
        if not scenario.per_client:
            continue
        lvl = scenario.intensity.upper()
        n_aff = scenario.affected_clients
        n_tot = scenario.num_clients
        print()
        print(f"  ┌─ Intensity: {lvl}  ({n_aff}/{n_tot} clients poisoned, each in its own local store) "
              + "─" * max(0, W - 62) + "┐")
        hdr = f"  │  {'Client':>6}  {'Role':^11}  {'Baseline F1':>11}  {'Attacked F1':>11}  {'Δ F1':>8}  {'Defended F1':>11}  │"
        print(hdr)
        print("  ├" + "─" * (W - 2) + "┤")

        for cid in sorted(scenario.per_client.keys()):
            b = scenario.per_client[cid]
            role = "✗ MALICIOUS" if b["malicious"] else "  clean    "
            delta = b["delta_f1"]
            arrow = "↓" if delta < -0.001 else ("↑" if delta > 0.001 else "─")
            delta_str = f"{arrow}{abs(delta):.4f}"
            print(f"  │  {cid:>6}  {role:^11}  "
                  f"{b['baseline_f1']:>11.4f}  {b['attacked_f1']:>11.4f}  "
                  f"{delta_str:>8}  {b['defended_f1']:>11.4f}  │")

        bl_f1 = scenario.baseline["f1"]
        atk_f1 = scenario.attacked["f1"]
        def_f1 = (scenario.defended or {}).get("f1", 0)
        print("  ├" + "─" * (W - 2) + "┤")
        print(f"  │  {'GLOBAL':>6}  {'system':^11}  "
              f"{bl_f1:>11.4f}  {atk_f1:>11.4f}  "
              f"{atk_f1-bl_f1:>+8.4f}  {def_f1:>11.4f}  │")
        print("  └" + "─" * (W - 2) + "┘")
    print()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FedRAG — Global System Impact Analysis (per-client local stores)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--num-clients", type=int, default=10,
        help="Number of federated clients (default: 10)",
    )
    p.add_argument(
        "--num-examples", type=int, default=200,
        help="Total (query, response) examples, split IID across clients (default: 200)",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for full reproducibility (default: 0)",
    )
    p.add_argument(
        "--poisoning-ratio", type=float, default=0.5,
        help="Fraction of each malicious client's LOCAL data to poison (default: 0.5). "
             "Range 0.0–1.0.",
    )
    p.add_argument(
        "--quarantine-threshold", type=float, default=0.5,
        help="Defense sensitivity: risk score to trigger client quarantine (default: 0.5).",
    )
    p.add_argument(
        "--attacks", nargs="+",
        choices=["data_poisoning", "kb_extraction", "node_availability"],
        default=["data_poisoning", "kb_extraction", "node_availability"],
        help="Which attacks to evaluate (default: all three)",
    )
    p.add_argument(
        "--node-attack-types", nargs="+",
        choices=["node_removal", "byzantine", "ddos", "partition", "sybil"],
        default=["node_removal", "byzantine", "ddos"],
        help="Node-availability attack variants (default: node_removal byzantine ddos)",
    )
    p.add_argument(
        "--intensities", nargs="+",
        choices=["light", "medium", "heavy"],
        default=["light", "medium", "heavy"],
        help="Attack intensity levels (default: light medium heavy)",
    )
    p.add_argument(
        "--dataset-jsonl", type=str, default=None,
        help="Path to a JSONL dataset (query/response/topic). "
             "Uses synthetic data when omitted.",
    )
    p.add_argument(
        "--output-dir", type=str, default="results/global",
        help="Directory to write all output files (default: results/global/)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from security_framework import FedRAGSimulator
    from security_framework.global_impact import GlobalImpactEvaluator
    from security_framework.global_reporter import GlobalImpactReporter

    if not args.quiet:
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║     FedRAG — Global System Impact Analysis                       ║")
        print("║     Architecture: True Per-Client Local InMemoryKnowledgeStore   ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print(f"║  Clients       : {args.num_clients:<48}║")
        print(f"║  Examples      : {args.num_examples:<48}║")
        print(f"║  Seed          : {args.seed:<48}║")
        print(f"║  Attacks       : {', '.join(args.attacks):<48}║")
        print(f"║  Intensities   : {', '.join(args.intensities):<48}║")
        print(f"║  Poison ratio  : {args.poisoning_ratio:<48}║")
        print(f"║  Quarantine    : {args.quarantine_threshold:<48}║")
        print(f"║  Dataset       : {str(args.dataset_jsonl or 'synthetic'):<48}║")
        print(f"║  Output dir    : {args.output_dir:<48}║")
        print("╚══════════════════════════════════════════════════════════════════╝")

    dataset_path = None
    if args.dataset_jsonl:
        p = Path(args.dataset_jsonl)
        if not p.exists():
            print(
                f"\n  ERROR: Dataset file not found: {args.dataset_jsonl}\n"
                f"\n  Omit --dataset-jsonl to use synthetic data:\n"
                f"\n      python run_global_impact_analysis.py\n"
            )
            sys.exit(1)
        dataset_path = str(p)

    sim = FedRAGSimulator(
        num_clients=args.num_clients,
        num_examples=args.num_examples,
        seed=args.seed,
        dataset_jsonl=dataset_path,
    )

    node_types = args.node_attack_types if "node_availability" in args.attacks else []
    _SKIP_DP = "data_poisoning" not in args.attacks
    _SKIP_KB = "kb_extraction" not in args.attacks
    _SKIP_NA = "node_availability" not in args.attacks

    from security_framework import global_impact as _gi

    _orig_run = _gi.GlobalImpactEvaluator.run

    def _filtered_run(self_inner):  # type: ignore[no-untyped-def]
        from security_framework.global_impact import GlobalImpactReport

        report = GlobalImpactReport(simulator_config=self_inner.sim.to_dict())

        print("\n[GlobalImpactEvaluator] Computing baseline global performance "
              "(true per-client fan-out)...")
        self_inner._baseline = _gi._evaluate_global(
            self_inner.sim.dataset, self_inner.sim.clients, self_inner.sim.embedding_dim
        )
        b = self_inner._baseline
        print(f"  Baseline → EM={b['em']:.4f}  F1={b['f1']:.4f}  BLEU={b['bleu']:.4f}")

        if not _SKIP_DP:
            print("\n[GlobalImpactEvaluator] Running Data Poisoning scenarios "
                  "(per-client local stores)...")
            report.scenarios.extend(self_inner._run_data_poisoning())
            report.cascade_tables["data_poisoning"] = self_inner._cascade_table_poisoning()

        if not _SKIP_KB:
            print("\n[GlobalImpactEvaluator] Running KB Extraction scenarios...")
            report.scenarios.extend(self_inner._run_kb_extraction())
            report.cascade_tables["kb_extraction"] = self_inner._cascade_table_extraction()

        if not _SKIP_NA:
            print("\n[GlobalImpactEvaluator] Running Node Availability scenarios...")
            report.scenarios.extend(self_inner._run_node_availability())
            report.cascade_tables["node_availability"] = self_inner._cascade_table_node_availability()

        print("\n[GlobalImpactEvaluator] All scenarios complete.")
        return report

    _gi.GlobalImpactEvaluator.run = _filtered_run  # type: ignore[method-assign]

    evaluator = GlobalImpactEvaluator(
        sim,
        node_attack_types=node_types,
        intensities=args.intensities,
        seed=args.seed,
        poisoning_ratio=args.poisoning_ratio,
        quarantine_threshold=args.quarantine_threshold,
    )
    report = evaluator.run()

    reporter = GlobalImpactReporter(report)
    reporter.print_summary()
    _print_global_per_client_table(report)

    print(f"\n[Saving results to {args.output_dir}/]")
    reporter.save(args.output_dir)

    print("\nDone. Key output files:")
    print(f"  {args.output_dir}/GLOBAL_IMPACT_REPORT.md   ← human-readable report")
    print(f"  {args.output_dir}/global_attack_matrix.csv  ← flat CSV")
    print(f"  {args.output_dir}/global_impact_results.json← raw JSON")
    print(f"  {args.output_dir}/cascade_*.csv             ← per-client degradation cascades")


if __name__ == "__main__":
    main()
