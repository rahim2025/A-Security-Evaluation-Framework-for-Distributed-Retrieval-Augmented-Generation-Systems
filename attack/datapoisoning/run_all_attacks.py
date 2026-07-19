"""
Run every combination of data-poisoning attack strategy x poison_type
against the live Reliable-dRAG services and summarize the results.

Each entry in ATTACK_MATRIX is a full `--evaluate` pass of run_attack.py:
clean baseline -> inject poison -> attacked responses -> reset to clean.
Sources are restored to their clean state after every run (Phase 5 of
run_attack.py), so combinations are independent and safe to run back-to-back.

Usage
-----
python attack/datapoisoning/run_all_attacks.py                      # single unseeded run per combo (legacy)
python attack/datapoisoning/run_all_attacks.py --seeds 0 42 123     # 3 seeded runs per combo, mean +/- std reported
python attack/datapoisoning/run_all_attacks.py --dry-run            # print commands only
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.abspath(os.path.join(HERE, '..', '..', 'attack_logs'))

# (strategy, poison_type, extra_args, label)
ATTACK_MATRIX = [
    ("random",           "noise",        [],                                        "random / noise"),
    ("random",           "answer_swap",  [],                                        "random / answer_swap"),
    ("targeted",         "wrong_answer", ["--targets", "sources_100"],               "targeted(sources_100) / wrong_answer"),
    ("targeted",         "misleading",   ["--targets", "sources_0", "sources_20"],   "targeted(sources_0,sources_20) / misleading"),
    ("data_rich",        "wrong_answer", [],                                        "data_rich / wrong_answer"),
    ("data_rich",        "noise",        [],                                        "data_rich / noise"),
]


def run_one(strategy, poison_type, extra_args, label, run_attack_path, seed=None, no_query_aware=False):
    cmd = [
        sys.executable, run_attack_path,
        "--strategy", strategy,
        "--poison-type", poison_type,
        *extra_args,
        "--evaluate",
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if no_query_aware:
        cmd += ["--no-query-aware"]

    seed_note = f" (seed={seed})" if seed is not None else ""
    print("\n" + "=" * 70)
    print(f"RUNNING: {label}{seed_note}")
    print(f"  cmd: {' '.join(cmd)}")
    print("=" * 70)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    before = set(glob.glob(os.path.join(LOG_DIR, "*.json")))
    start = time.time()
    result = subprocess.run(cmd, env=env, cwd=HERE)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ⚠ run failed (exit code {result.returncode}) after {elapsed:.0f}s")
        return {"label": label, "strategy": strategy, "poison_type": poison_type, "seed": seed,
                "error": f"exit code {result.returncode}", "elapsed_sec": round(elapsed, 1)}

    after = set(glob.glob(os.path.join(LOG_DIR, "*.json")))
    new_logs = sorted(after - before)
    if not new_logs:
        print("  ⚠ no new log file found")
        return {"label": label, "strategy": strategy, "poison_type": poison_type, "seed": seed,
                "error": "no log produced", "elapsed_sec": round(elapsed, 1)}

    log_path = new_logs[-1]
    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    ev = log.get("evaluation", {})
    return {
        "label": label,
        "strategy": strategy,
        "poison_type": poison_type,
        "seed": seed,
        "query_aware": not no_query_aware,
        "log_path": log_path,
        "clean_accuracy": ev.get("clean_accuracy"),
        "attacked_accuracy": ev.get("attacked_accuracy"),
        "degradation_pct": ev.get("accuracy_degradation_pct"),
        "is_successful": ev.get("is_successful"),
        "elapsed_sec": round(elapsed, 1),
    }


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "std": None}
    if len(values) < 2:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def aggregate_seed_runs(runs):
    """Aggregate per-seed results for a single combo (same label) into mean/std stats."""
    ok_runs = [r for r in runs if "error" not in r]
    n = len(runs)
    n_ok = len(ok_runs)
    if not ok_runs:
        return {"n_runs": n, "n_ok": 0, "seeds": [r.get("seed") for r in runs], "error": "all runs failed"}

    return {
        "n_runs": n,
        "n_ok": n_ok,
        "seeds": [r.get("seed") for r in ok_runs],
        "clean_accuracy": _mean_std([r["clean_accuracy"] for r in ok_runs]),
        "attacked_accuracy": _mean_std([r["attacked_accuracy"] for r in ok_runs]),
        "degradation_pct": _mean_std([r["degradation_pct"] for r in ok_runs]),
        "success_rate": sum(1 for r in ok_runs if r.get("is_successful")) / n_ok,
    }


def main():
    parser = argparse.ArgumentParser(description="Run every data-poisoning attack variant and summarize results")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without executing them")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Run each combo once per seed (e.g. --seeds 0 42 123) and report "
                             "mean +/- std instead of a single unseeded run. See "
                             "problems/data_poisoning_gaps.md, B4/B5, for why this matters.")
    parser.add_argument("--no-query-aware", action="store_true",
                        help="Run the whole matrix black-box (attacker doesn't know the eval "
                             "questions), instead of the default oracle-knowledge tier. See "
                             "problems/data_poisoning_gaps.md, B2.")
    args = parser.parse_args()

    run_attack_path = os.path.join(HERE, "run_attack.py")
    seeds = args.seeds if args.seeds else [None]

    if args.dry_run:
        for strategy, poison_type, extra_args, label in ATTACK_MATRIX:
            for seed in seeds:
                cmd = [sys.executable, run_attack_path, "--strategy", strategy,
                       "--poison-type", poison_type, *extra_args, "--evaluate"]
                if seed is not None:
                    cmd += ["--seed", str(seed)]
                if args.no_query_aware:
                    cmd += ["--no-query-aware"]
                seed_note = f" (seed={seed})" if seed is not None else ""
                print(f"[{label}{seed_note}] {' '.join(cmd)}")
        return

    results = []
    for strategy, poison_type, extra_args, label in ATTACK_MATRIX:
        for seed in seeds:
            results.append(run_one(strategy, poison_type, extra_args, label, run_attack_path,
                                    seed=seed, no_query_aware=args.no_query_aware))

    print("\n\n" + "=" * 100)
    print("SUMMARY")
    print(f"Threat tier: {'black-box (--no-query-aware)' if args.no_query_aware else 'oracle-knowledge (default)'}")
    print("=" * 100)

    if args.seeds:
        # Multi-seed mode: aggregate per label and report mean +/- std.
        grouped = {}
        for r in results:
            grouped.setdefault(r["label"], []).append(r)
        aggregated = {label: aggregate_seed_runs(runs) for label, runs in grouped.items()}

        header = f"{'Attack':<45} {'Seeds':>6} {'Clean':>8} {'Attacked (mean+/-std)':>24} {'Drop % (mean+/-std)':>22} {'Success rate':>13}"
        print(header)
        print("-" * 120)
        for strategy, poison_type, extra_args, label in ATTACK_MATRIX:
            agg = aggregated.get(label, {})
            if agg.get("n_ok", 0) == 0:
                print(f"{label:<45} ERROR: all {agg.get('n_runs', 0)} seed runs failed")
                continue
            clean = agg["clean_accuracy"]["mean"]
            att = agg["attacked_accuracy"]
            drop = agg["degradation_pct"]
            clean_s = f"{clean:.1%}" if clean is not None else "-"
            att_s = f"{att['mean']:.1%} +/- {att['std']:.1%}" if att["mean"] is not None else "-"
            drop_s = f"{drop['mean']:.1f} +/- {drop['std']:.1f}" if drop["mean"] is not None else "-"
            success_s = f"{agg['success_rate']:.0%} ({agg['n_ok']}/{agg['n_ok']} runs)"
            print(f"{label:<45} {agg['n_ok']:>6} {clean_s:>8} {att_s:>24} {drop_s:>22} {success_s:>13}")
        print("=" * 100)

        os.makedirs(LOG_DIR, exist_ok=True)
        summary_path = os.path.join(LOG_DIR, f"attack_matrix_summary_seeds_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"seeds": args.seeds, "query_aware": not args.no_query_aware,
                        "runs": results, "aggregated": aggregated}, f, indent=2)
        print(f"\nFull summary saved -> {summary_path}")
    else:
        # Legacy single-run-per-combo mode: unchanged behavior/format.
        header = f"{'Attack':<45} {'Clean':>8} {'Attacked':>10} {'Drop %':>8} {'Success':>8} {'Time(s)':>8}"
        print(header)
        print("-" * 100)
        for r in results:
            if "error" in r:
                print(f"{r['label']:<45} ERROR: {r['error']}")
                continue
            clean = f"{r['clean_accuracy']:.1%}" if r['clean_accuracy'] is not None else "-"
            attacked = f"{r['attacked_accuracy']:.1%}" if r['attacked_accuracy'] is not None else "-"
            drop = f"{r['degradation_pct']:.1f}" if r['degradation_pct'] is not None else "-"
            success = "YES" if r.get("is_successful") else "no"
            print(f"{r['label']:<45} {clean:>8} {attacked:>10} {drop:>8} {success:>8} {r['elapsed_sec']:>8}")
        print("=" * 100)

        os.makedirs(LOG_DIR, exist_ok=True)
        summary_path = os.path.join(LOG_DIR, f"attack_matrix_summary_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
