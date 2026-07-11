"""
Run every combination of data-poisoning attack strategy x poison_type
against the live Reliable-dRAG services and summarize the results.

Each entry in ATTACK_MATRIX is a full `--evaluate` pass of run_attack.py:
clean baseline -> inject poison -> attacked responses -> reset to clean.
Sources are restored to their clean state after every run (Phase 5 of
run_attack.py), so combinations are independent and safe to run back-to-back.

Usage
-----
python attack/datapoisoning/run_all_attacks.py            # run the full matrix
python attack/datapoisoning/run_all_attacks.py --dry-run  # print commands only
"""

import argparse
import glob
import json
import os
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
    ("high_reliability",  "wrong_answer", [],                                       "high_reliability / wrong_answer"),
    ("high_reliability",  "noise",        [],                                       "high_reliability / noise"),
]


def run_one(strategy, poison_type, extra_args, label, run_attack_path):
    cmd = [
        sys.executable, run_attack_path,
        "--strategy", strategy,
        "--poison-type", poison_type,
        *extra_args,
        "--evaluate",
    ]
    print("\n" + "=" * 70)
    print(f"RUNNING: {label}")
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
        return {"label": label, "strategy": strategy, "poison_type": poison_type,
                "error": f"exit code {result.returncode}", "elapsed_sec": round(elapsed, 1)}

    after = set(glob.glob(os.path.join(LOG_DIR, "*.json")))
    new_logs = sorted(after - before)
    if not new_logs:
        print("  ⚠ no new log file found")
        return {"label": label, "strategy": strategy, "poison_type": poison_type,
                "error": "no log produced", "elapsed_sec": round(elapsed, 1)}

    log_path = new_logs[-1]
    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    ev = log.get("evaluation", {})
    return {
        "label": label,
        "strategy": strategy,
        "poison_type": poison_type,
        "log_path": log_path,
        "clean_accuracy": ev.get("clean_accuracy"),
        "attacked_accuracy": ev.get("attacked_accuracy"),
        "degradation_pct": ev.get("accuracy_degradation_pct"),
        "is_successful": ev.get("is_successful"),
        "elapsed_sec": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Run every data-poisoning attack variant and summarize results")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without executing them")
    args = parser.parse_args()

    run_attack_path = os.path.join(HERE, "run_attack.py")

    if args.dry_run:
        for strategy, poison_type, extra_args, label in ATTACK_MATRIX:
            cmd = [sys.executable, run_attack_path, "--strategy", strategy,
                   "--poison-type", poison_type, *extra_args, "--evaluate"]
            print(f"[{label}] {' '.join(cmd)}")
        return

    results = []
    for strategy, poison_type, extra_args, label in ATTACK_MATRIX:
        results.append(run_one(strategy, poison_type, extra_args, label, run_attack_path))

    print("\n\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
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
