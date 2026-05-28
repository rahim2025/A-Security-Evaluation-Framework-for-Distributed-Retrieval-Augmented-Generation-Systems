"""
run_all.py
----------
Main entry point for the DRAG Scalability Testing Suite.

Usage examples
--------------
# Run ALL four dimensions:
    python -m scalability_tests.run_all

# Run only specific dimensions:
    python -m scalability_tests.run_all --dimensions volume network

# Run and regenerate plots from existing results only:
    python -m scalability_tests.run_all --plots-only

# Run a quick smoke-test (tiny configs, fast):
    python -m scalability_tests.run_all --quick

# Print a summary table:
    python -m scalability_tests.run_all --summarise
"""
import argparse
import sys
from typing import List

from scalability_tests.config_generator import (
    ExperimentConfig,
    volume_experiments,
    network_experiments,
    attack_experiments,
    dataset_experiments,
)
from scalability_tests.runner import run_experiment
from scalability_tests.metrics_collector import save_results, summarise, export_json
from scalability_tests.visualizer import plot_all


# ---------------------------------------------------------------------------
# Quick (smoke-test) overrides — very small configs so the full suite
# completes in minutes even without a GPU
# ---------------------------------------------------------------------------
def _make_quick(exps: List[ExperimentConfig]) -> List[ExperimentConfig]:
    """Shrink every experiment to the smallest viable config."""
    quick = []
    for e in exps:
        e.cli_overrides["data.num_samples"] = min(
            e.cli_overrides.get("data.num_samples", 10), 10
        )
        if "rag.num_peers" in e.cli_overrides:
            e.cli_overrides["rag.num_peers"] = min(e.cli_overrides["rag.num_peers"], 10)
        quick.append(e)
    # For volume, keep only 3 points
    if exps and exps[0].dimension == "volume":
        return quick[:3]
    return quick


DIMENSION_MAP = {
    "volume":  volume_experiments,
    "network": network_experiments,
    "attack":  attack_experiments,
    "dataset": dataset_experiments,
}

ALL_DIMENSIONS = list(DIMENSION_MAP.keys())


def main():
    parser = argparse.ArgumentParser(
        description="DRAG Scalability Testing Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        choices=ALL_DIMENSIONS,
        default=ALL_DIMENSIONS,
        metavar="DIM",
        help=f"Which dimensions to test. Options: {', '.join(ALL_DIMENSIONS)}. Default: all.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: use tiny configs to verify the pipeline runs end-to-end.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip running experiments and just regenerate plots from existing results.",
    )
    parser.add_argument(
        "--summarise",
        action="store_true",
        help="Print a summary table of all saved results and exit.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-experiment subprocess timeout in seconds (default: 600).",
    )
    args = parser.parse_args()

    # ── Summary only ──────────────────────────────────────────────────────
    if args.summarise:
        summarise()
        return 0

    # ── Plots only ────────────────────────────────────────────────────────
    if args.plots_only:
        print("[run_all] Regenerating plots from existing results …")
        plot_all()
        return 0

    # ── Collect experiments ───────────────────────────────────────────────
    experiments: List[ExperimentConfig] = []
    for dim in args.dimensions:
        exps = DIMENSION_MAP[dim]()
        if args.quick:
            exps = _make_quick(exps)
        experiments.extend(exps)

    total = len(experiments)
    print(f"\n{'#'*65}")
    print(f"  DRAG SCALABILITY TEST SUITE")
    print(f"  Dimensions : {', '.join(args.dimensions)}")
    print(f"  Experiments: {total}")
    print(f"  Quick mode : {args.quick}")
    print(f"  Timeout    : {args.timeout}s / experiment")
    print(f"{'#'*65}\n")

    # ── Run experiments ───────────────────────────────────────────────────
    all_results = []
    failed      = []

    for idx, exp in enumerate(experiments, 1):
        print(f"\n[{idx}/{total}] Running {exp.experiment_id} …")
        result = run_experiment(exp, timeout=args.timeout)
        all_results.append(result)

        if result.get("status", "").startswith(("error", "exception", "timeout")):
            failed.append(exp.experiment_id)

        # Save incrementally so partial results are not lost on crash
        save_results([result], append=True)

    # ── Final report ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  DONE  —  {total} experiments, {len(failed)} failures")
    if failed:
        print(f"  Failed IDs: {', '.join(failed)}")
    print(f"{'='*65}\n")

    summarise()

    # ── Generate plots ────────────────────────────────────────────────────
    print("\n[run_all] Generating plots …")
    plot_all()

    # ── Export JSON ───────────────────────────────────────────────────────
    export_json()

    print("\n[run_all] All done. Results in scalability_tests/results/")
    print("          Plots    in scalability_tests/plots/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
