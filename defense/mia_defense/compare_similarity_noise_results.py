"""
defense/mia_defense/compare_similarity_noise_results.py

Reads every defense_logs/similarity_noise_*.json produced by
run_similarity_noise_defense.py and prints a single comparison table:
one row per --config_label run (e.g. no_defense, eps_0.1, eps_1.0,
eps_5.0), columns = MIA AUC-ROC and task F1/EM, mean +/- std over that
run's seeds.

Usage
-----
  python defense/mia_defense/compare_similarity_noise_results.py
  python defense/mia_defense/compare_similarity_noise_results.py --logs_dir defense_logs
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> None:
    p = argparse.ArgumentParser()
    _here = os.path.dirname(os.path.abspath(__file__))
    _default_logs = os.path.abspath(os.path.join(_here, "..", "..", "defense_logs"))
    p.add_argument("--logs_dir", default=_default_logs)
    args = p.parse_args()

    pattern = os.path.join(args.logs_dir, "similarity_noise_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No logs found matching {pattern}. Run run_similarity_noise_defense.py first.")
        return

    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            log = json.load(fh)
        s = log["summary"]
        rows.append({
            "config_label": s["config_label"],
            "n_seeds": s["n_seeds"],
            "mia_auc_roc": f"{s['mia_auc_roc_mean']:.4f} +/- {s['mia_auc_roc_std']:.4f}",
            "task_f1": f"{s['task_f1_mean']:.4f} +/- {s['task_f1_std']:.4f}",
            "task_em": f"{s['task_exact_match_mean']:.4f} +/- {s['task_exact_match_std']:.4f}",
            "source_file": os.path.basename(path),
            "timestamp": log.get("timestamp", ""),
        })

    # One row per config_label -- if a label was run more than once, keep
    # the most recent (paths sorted ascending by filename, which embeds a
    # sortable timestamp).
    by_label = {}
    for r in rows:
        by_label[r["config_label"]] = r
    rows = list(by_label.values())

    headers = ["config_label", "n_seeds", "mia_auc_roc", "task_f1", "task_em", "timestamp"]
    widths = {h: max(len(h), max((len(str(r[h])) for r in rows), default=0)) for h in headers}

    def _fmt_row(vals):
        return "  ".join(str(v).ljust(widths[h]) for h, v in zip(headers, vals))

    print(_fmt_row(headers))
    print(_fmt_row(["-" * widths[h] for h in headers]))
    for r in rows:
        print(_fmt_row([r[h] for h in headers]))

    print(f"\n{len(rows)} config(s) found across {len(paths)} log file(s) in {args.logs_dir}")
    print("Note: 'no_defense' (or whatever label was used for a disabled/baseline run) "
          "is the reference row -- every other row's mia_auc_roc drop and task_f1/em "
          "drop relative to it are the actual privacy-vs-quality tradeoff this defense "
          "produces at that epsilon.")


if __name__ == "__main__":
    main()
