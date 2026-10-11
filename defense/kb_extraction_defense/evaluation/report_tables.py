"""
defense/kb_extraction_defense/evaluation/report_tables.py

Prints the README tables (markdown) straight from the result JSONs in
defense_logs/kb_extraction_defense/offline/, so no number in the write-up is typed by hand.

    python defense/kb_extraction_defense/evaluation/report_tables.py --source 1
"""
from __future__ import annotations

import argparse
import json
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
RES = os.path.join(_ROOT, "defense_logs", "kb_extraction_defense", "offline")


def load(name, tag):
    p = os.path.join(RES, f"{name}_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def pm(a):
    return f"{a['mean']:.3f} ± {a['std']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=1)
    tag = f"source{ap.parse_args().source}"

    cal = load("calibrate", tag)
    if cal:
        print(f"### Calibration ({tag})  budget B = {cal['client_budget']}   chosen = {cal['chosen']}\n")

    cmp_ = load("compare", tag)
    if cmp_:
        att = cmp_["attack"]
        names = list(att["none"].keys())
        print("### Extraction rate ER after 600 query attempts (mean ± std over seeds 0/42/123)\n")
        print("| attacker | " + " | ".join(att.keys()) + " |")
        print("|---|" + "---|" * len(att))
        for an in names:
            print(f"| {an} | " + " | ".join(pm(att[m][an]["er_final"]) for m in att) + " |")
        print("\n### First flagged query / ER at first flag (guard)\n")
        print("| attacker | first flag (query #) | ER at first flag |\n|---|---|---|")
        for an in names:
            g = att["guard"][an]
            print(f"| {an} | {g['first_flag']['mean']:.1f} | {g['er_at_first_flag']['mean']:.3f} |")
        print("\n### Legitimate-user impact (mean over 3 seeds; 'affected' = at least one degraded/frozen query)\n")
        leg = cmp_["legit"]
        classes = [k for k in leg["guard"][0].keys() if not k.startswith("_")]
        print("| defense | " + " | ".join(classes) + " | protected FP | hit-rate drop |")
        print("|---|" + "---|" * (len(classes) + 2))
        for m, per_seed in leg.items():
            row = []
            for cl in classes:
                row.append(f"{sum(s[cl]['any_affected'] for s in per_seed) / len(per_seed):.3f}")
            fp = sum(s["_protected"]["any_affected"] for s in per_seed) / len(per_seed)
            hd = sum(s["_protected"]["hit_drop"] for s in per_seed) / len(per_seed)
            print(f"| {m} | " + " | ".join(row) + f" | {fp:.3f} | {hd:.3f} |")

    ab = load("ablate", tag)
    if ab:
        att = ab["attack"]
        names = list(next(iter(att.values())).keys())
        print("\n### Ablation: ER_final (mean over seeds) and protected-legit impact\n")
        print("| variant | " + " | ".join(names) + " | protected FP | hit-rate drop |")
        print("|---|" + "---|" * (len(names) + 2))
        for v in att:
            lg = ab["legit"][v]
            print(f"| {v} | " + " | ".join(f"{att[v][n]['er_final']['mean']:.3f}" for n in names)
                  + f" | {lg['any_affected']:.3f} | {lg['hit_drop']:.3f} |")

    sy = load("sybil", tag)
    if sy:
        print("\n### Sybil (identity splitting): ER_final of snow_frontier\n")
        print("| identities | per-client budget only | + global budget |\n|---|---|---|")
        for sn, v in sy.items():
            print(f"| {sn} | {pm(v['per-client budget only'])} | {pm(v['per-client + global budget'])} |")

    sc = load("scale", tag)
    if sc:
        print("\n### Budget bound vs corpus size\n")
        print("| corpus | N | B | bound B/N | frontier undefended → budget | local+dilution3 undefended → budget |\n|---|---|---|---|---|---|")
        for k, v in sc.items():
            a = v["attackers"]
            print(f"| {k} | {v['N']} | {v['budget']} | {v['bound']:.3f} | "
                  f"{a['snow_frontier']['undefended']['mean']:.3f} → {a['snow_frontier']['budget_only']['mean']:.3f} | "
                  f"{a['snow_local_dil3']['undefended']['mean']:.3f} → {a['snow_local_dil3']['budget_only']['mean']:.3f} |")


if __name__ == "__main__":
    main()
