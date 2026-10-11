"""
defense/kb_extraction_defense/evaluation/offline_eval.py

Offline (in-process) evaluation of KB-extraction defenses on the real corpora,
real embedding model and real retrieval rule. Multi-seed (0, 42, 123), with
calibration seeds disjoint from test seeds.

  python defense/kb_extraction_defense/evaluation/offline_eval.py threat      # undefended exposure
  python defense/kb_extraction_defense/evaluation/offline_eval.py signals     # per-signal AUROC
  python defense/kb_extraction_defense/evaluation/offline_eval.py calibrate   # thresholds from legit data
  python defense/kb_extraction_defense/evaluation/offline_eval.py compare     # defenses x attackers x seeds
  python defense/kb_extraction_defense/evaluation/offline_eval.py ablate      # which mechanism earns its keep
  python defense/kb_extraction_defense/evaluation/offline_eval.py scale       # leakage bound vs corpus size
  python defense/kb_extraction_defense/evaluation/offline_eval.py sybil       # identity splitting
  python defense/kb_extraction_defense/evaluation/offline_eval.py all

Results -> defense_logs/kb_extraction_defense/offline/<experiment>_<source>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.kb_extraction.adaptive_attack import (  # noqa: E402
    SnowballCrawler, StaticProbes, Trace,
)
from defense.kb_extraction_defense.evaluation import legit_workloads as LW  # noqa: E402
from defense.kb_extraction_defense.evaluation.eval_common import (  # noqa: E402
    Corpus, Embedder, load_corpus, union_corpus,
)
from defense.kb_extraction_defense.extraction_guard import ALLOW, FREEZE, GuardConfig  # noqa: E402
from defense.kb_extraction_defense.evaluation.sim_server import SimServer, SimTransport  # noqa: E402

TEST_SEEDS = (0, 42, 123)
CAL_SEED = 1001
CHECKPOINTS = (25, 50, 100, 200, 400, 600)
MAX_Q = 600
OUT_DIR = os.path.join(_ROOT, "defense_logs", "kb_extraction_defense", "offline")
PROTECTED = ("focused1", "focused2", "focused3")     # legitimate classes the defense must not disturb


# ----------------------------------------------------------------------------- setup
def public_pool(source: int, n: int = 4000, seed: int = 0) -> List[str]:
    """Generic public questions an attacker could obtain without knowing the corpus."""
    from datasets import load_dataset
    if source == 0:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        qs = [x["question"].strip() for x in ds]
    else:
        ds = load_dataset("rajpurkar/squad", split="train")
        qs = [x["question"].strip() for x in ds]
    return random.Random(seed).sample(qs, min(n, len(qs)))


def make_attackers(c: Corpus, E: Embedder, pool: List[str]) -> Dict[str, Callable[[int], object]]:
    def snow(sel, dil):
        def f(seed):
            seeds = random.Random(seed).sample(pool, 8)
            return SnowballCrawler(sel, seeds, embed_fn=E.embed, dilution=dil)
        return f

    def static(seed):
        qs = [q.question for q in random.Random(seed).sample(c.qa, min(54, len(c.qa)))]
        return StaticProbes(qs)

    return {
        "static54": static,
        "snow_random": snow("random", 0),
        "snow_frontier": snow("frontier", 0),
        "snow_local": snow("local", 0),
        "snow_frontier_dil3": snow("frontier", 3),
        "snow_local_dil3": snow("local", 3),
        "snow_dwell": snow("dwell", 0),
        "snow_dwell_dil3": snow("dwell", 3),
    }


def build_server(c: Corpus, E: Embedder, spec: dict) -> SimServer:
    mode = spec["mode"]
    gcfg = GuardConfig(**spec.get("guard", {})) if mode == "guard" else None
    disp = bool(gcfg and gcfg.dispersion_thresh is not None)
    return SimServer(c, E, mode, guard_cfg=gcfg, use_dispersion=disp, v1_cfg=spec.get("v1"))


# ----------------------------------------------------------------------------- runs
def run_attack(c: Corpus, E: Embedder, spec: dict, attacker, seed: int, max_q: int = MAX_Q,
               sybil: int = 1, qps: float = 1.0) -> dict:
    server = build_server(c, E, spec)
    tp = SimTransport(server, lambda n: f"atk{n % sybil}", qps=qps)
    tr: Trace = attacker.run(tp, max_q, seed)
    N = len(c)
    er = {q: round(len(tr.extracted_ids(q)) / N, 4) for q in CHECKPOINTS}
    first_flag = next((i for i, (st, ac) in enumerate(tp.log) if ac != ALLOW or st != "ok"), None)
    first_freeze = next((i for i, (st, ac) in enumerate(tp.log) if ac == FREEZE or st != "ok"), None)
    er_at_flag = round(len(tr.extracted_ids(first_flag)) / N, 4) if first_flag is not None else None
    return {
        "er_final": round(len(tr.extracted_ids()) / N, 4),
        "er_curve": er,
        "attempts": len(tr.records),
        "first_flag": first_flag,
        "first_freeze": first_freeze,
        "er_at_first_flag": er_at_flag,
        "unique_docs": len(tr.extracted_ids()),
    }


def run_legit(c: Corpus, E: Embedder, spec: dict, sess: LW.Session) -> dict:
    server = build_server(c, E, spec)
    base = build_server(c, E, {"mode": "none"})
    affected = frozen = 0
    hit_def = hit_base = 0
    seen = set()
    for i, q in enumerate(sess.queries):
        r = server.query("u", q.question, 5, now=float(i))
        b = base.query("u", q.question, 5, now=float(i))
        ids = {x["id"] for x in r.results}
        seen |= ids
        if r.action != ALLOW or r.status != "ok":
            affected += 1
        if r.action == FREEZE or r.status != "ok":
            frozen += 1
        hit_def += q.doc_id in ids
        hit_base += q.doc_id in {x["id"] for x in b.results}
    n = len(sess.queries)
    return {"affected_frac": affected / n, "any_affected": affected > 0, "frozen_frac": frozen / n,
            "any_frozen": frozen > 0, "hit_def": hit_def / n, "hit_base": hit_base / n, "unique_docs": len(seen)}


def legit_class(s: LW.Session) -> str:
    if s.kind == "focused":
        return f"focused{s.params['n_topics']}"
    if s.kind == "browsing":
        return f"browsing{s.params['stay_p']}"
    return s.kind


def legit_summary(c, E, spec, suite) -> dict:
    by = defaultdict(list)
    for s in suite:
        by[legit_class(s)].append(run_legit(c, E, spec, s))
    out = {}
    for k, rs in sorted(by.items()):
        out[k] = {
            "sessions": len(rs),
            "any_affected": round(float(np.mean([r["any_affected"] for r in rs])), 4),
            "any_frozen": round(float(np.mean([r["any_frozen"] for r in rs])), 4),
            "queries_affected": round(float(np.mean([r["affected_frac"] for r in rs])), 4),
            "hit_base": round(float(np.mean([r["hit_base"] for r in rs])), 4),
            "hit_def": round(float(np.mean([r["hit_def"] for r in rs])), 4),
            "unique_docs_mean": round(float(np.mean([r["unique_docs"] for r in rs])), 1),
            "unique_docs_p99": float(np.percentile([r["unique_docs"] for r in rs], 99)),
        }
    prot = [r for k, rs in by.items() if k in PROTECTED for r in rs]
    out["_protected"] = {
        "sessions": len(prot),
        "any_affected": round(float(np.mean([r["any_affected"] for r in prot])), 4),
        "any_frozen": round(float(np.mean([r["any_frozen"] for r in prot])), 4),
        "hit_drop": round(float(np.mean([r["hit_base"] - r["hit_def"] for r in prot])), 4),
    }
    return out


def agg(vals: Sequence[float]) -> dict:
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4), "n": len(v)}


def attack_matrix(c, E, specs: Dict[str, dict], attackers: Dict[str, Callable], seeds, names=None,
                  **kw) -> dict:
    out = {}
    for sname, spec in specs.items():
        out[sname] = {}
        for an, mk in attackers.items():
            if names and an not in names:
                continue
            runs = [run_attack(c, E, spec, mk(s), s, **kw) for s in seeds]
            out[sname][an] = {
                "er_final": agg([r["er_final"] for r in runs]),
                "er_at_100": agg([r["er_curve"].get(100) for r in runs]),
                "er_at_first_flag": agg([r["er_at_first_flag"] for r in runs]),
                "first_flag": agg([r["first_flag"] for r in runs]),
                "unique_docs": agg([r["unique_docs"] for r in runs]),
                "runs": runs,
            }
    return out


def save(name: str, obj: dict, tag: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, f"{name}_{tag}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    return p


def fmt(a: dict) -> str:
    return "  n/a " if a["mean"] is None else f"{a['mean']:.3f}±{a['std']:.3f}"


# ----------------------------------------------------------------------------- experiments
def exp_threat(c, E, attackers, tag):
    """What an attacker recovers with only the deployed controls (5-result cap, 60/min)."""
    specs = {"none": {"mode": "none"}, "ratelimit60": {"mode": "ratelimit"}}
    res = attack_matrix(c, E, specs, attackers, TEST_SEEDS)
    print(f"\n[threat] corpus N={len(c)}  (ER = fraction of corpus recovered after {MAX_Q} attempts)")
    for sname in specs:
        for an, r in res[sname].items():
            curve = {q: agg([x["er_curve"].get(q) for x in r["runs"]])["mean"] for q in CHECKPOINTS}
            print(f"  {sname:12s} {an:20s} ER_final={fmt(r['er_final'])}  curve={curve}")
    print("  rate limit: 60 queries/min => 100 attempts = ~1.7 min, 600 attempts = 10 min "
          "(the limiter bounds speed, not total recovery)")
    print("  saved", save("threat", res, tag))
    return res


def feature_rows(c, E, suite, attackers, seeds, ck=(30, 60, 100, 200)):
    """Server-observable features at checkpoints, for legit sessions and attacker traces."""
    rows = []
    none = build_server(c, E, {"mode": "none"})

    def feats(texts, idlists, label, kind):
        embs = E.embed(texts)
        seen, news = set(), []
        for i, ids in enumerate(idlists):
            news.append(len([d for d in ids if d not in seen]))
            seen.update(ids)
            q = i + 1
            if q in ck:
                w = embs[max(0, q - 30):q]
                rows.append({"label": label, "kind": kind, "q": q,
                             "velocity": sum(news[max(0, q - 30):q]) / min(30, q),
                             "dispersion": float(1 - np.linalg.norm(w.sum(0)) / len(w)),
                             "unique_docs": len(seen)})
    for s in suite:
        ids = [[r["id"] for r in none.query("u", q.question, 5).results] for q in s.queries]
        feats([q.question for q in s.queries], ids, 0, legit_class(s))
    for an, mk in attackers.items():
        for sd in seeds:
            tr = mk(sd).run(SimTransport(none, lambda n: "a"), max(ck), sd)
            feats([r.text for r in tr.records], [r.doc_ids for r in tr.records], 1, an)
    return rows


def exp_signals(c, E, attackers, tag):
    from sklearn.metrics import roc_auc_score
    suite = LW.build_suite(c, CAL_SEED, n_per_cell=10)
    rows = feature_rows(c, E, suite, attackers, TEST_SEEDS)
    out = {}
    print(f"\n[signals] AUROC of each server-observable feature, legit (protected classes) vs attacker")
    print("   (0.5 = no information; evaluated at window checkpoints of q queries)")
    for q in (30, 60, 100, 200):
        R = [r for r in rows if r["q"] == q and (r["label"] == 1 or r["kind"] in PROTECTED)]
        y = np.array([r["label"] for r in R])
        line = {}
        for feat in ("velocity", "dispersion", "unique_docs"):
            line[feat] = round(float(roc_auc_score(y, [r[feat] for r in R])), 3)
        per = {}
        for an in sorted({r["kind"] for r in R if r["label"] == 1}):
            sub = [r for r in R if r["label"] == 0 or r["kind"] == an]
            yy = np.array([r["label"] for r in sub])
            per[an] = {f: round(float(roc_auc_score(yy, [r[f] for r in sub])), 3)
                       for f in ("velocity", "dispersion", "unique_docs")}
        out[q] = {"pooled": line, "per_attacker": per}
        print(f"  q={q:3d}  pooled {line}")
        for an, v in per.items():
            print(f"        vs {an:20s} {v}")
    print("  saved", save("signals", out, tag))
    return out


def legit_unique_quantile(c, E, suite, q=0.995) -> int:
    none = build_server(c, E, {"mode": "none"})
    u = []
    for s in suite:
        if legit_class(s) in PROTECTED:
            seen = set()
            for x in s.queries:
                seen |= {r["id"] for r in none.query("u", x.question, 5).results}
            u.append(len(seen))
    return int(math.ceil(np.quantile(u, q)))


def exp_calibrate(c, E, attackers, tag, n_clusters: int = 16):
    """Derive thresholds from legitimate data only (never from the test seeds)."""
    suite = LW.build_suite(c, CAL_SEED, n_per_cell=20)
    B = legit_unique_quantile(c, E, suite)
    print(f"\n[calibrate] corpus N={len(c)}: protected legit working set  p99.5 = {B} unique docs "
          f"(= {B / len(c):.1%} of the corpus)  -> client_budget")
    cal_attackers = {k: attackers[k] for k in ("snow_random", "snow_frontier")}
    grid = []
    combos = []
    for K in (6, 10, 16):
        for tc in (3, 4, 5, 6, 7, 8, 10):
            if tc > K:
                continue
            combos.append(dict(velocity_thresh=None, dispersion_thresh=None, breadth_thresh=tc, n_clusters=K))
            combos.append(dict(velocity_thresh=1.5, dispersion_thresh=None, breadth_thresh=tc, n_clusters=K))
    for td in (0.6,):
        combos.append(dict(velocity_thresh=1.5, dispersion_thresh=td, breadth_thresh=None, n_clusters=n_clusters))
    # fallback candidate: unique-doc budget only. Needs no corpus structure, so it is always deployable.
    combos.append(dict(velocity_thresh=None, dispersion_thresh=None, breadth_thresh=None, n_clusters=n_clusters,
                       escalate_after_flags=0))
    for combo in combos:
        combo = dict(combo)
        guard = dict(window=30, min_queries=10, degrade_k=1, cooldown_queries=20,
                     **{"escalate_after_flags": combo.pop("escalate_after_flags", 3)},
                     client_budget=B, epoch_seconds=1e9, **combo)
        combo = {**combo, "escalate_after_flags": guard["escalate_after_flags"]}
        spec = {"mode": "guard", "guard": guard}
        leg = legit_summary(c, E, spec, [s for s in suite if legit_class(s) in PROTECTED])
        atk = [run_attack(c, E, spec, mk(CAL_SEED + i), CAL_SEED + i) for mk in cal_attackers.values() for i in range(2)]
        grid.append({**combo,
                     "fp_protected": leg["_protected"]["any_affected"],
                     "hit_drop": leg["_protected"]["hit_drop"],
                     "er_final": round(float(np.mean([a["er_final"] for a in atk])), 4),
                     "er_at_flag": round(float(np.mean([a["er_at_first_flag"] or 1 for a in atk])), 4),
                     "first_flag": round(float(np.mean([a["first_flag"] if a["first_flag"] is not None else MAX_Q for a in atk])), 1)})
    print("   vel   disp  breadth  K | FP(protected) hit_drop | attacker ER_final  ER@flag  first_flag")
    for g in grid:
        print(f"  {str(g['velocity_thresh']):5s} {str(g['dispersion_thresh']):5s} {str(g['breadth_thresh']):5s} {g['n_clusters']:3d} |  {g['fp_protected']:.3f}       {g['hit_drop']:.3f}  |"
              f"  {g['er_final']:.3f}            {g['er_at_flag']:.3f}   {g['first_flag']}")
    ok = [g for g in grid if g["fp_protected"] <= 0.05]
    best = min(ok, key=lambda g: (g["er_final"], g["er_at_flag"])) if ok else None
    out = {"client_budget": B, "grid": grid, "chosen": best}
    print("  chosen (FP<=5% on protected classes, then minimum attacker ER):", best)
    print("  saved", save("calibrate", out, tag))
    return out


def tuned_spec(cal: dict, **over) -> dict:
    b = cal["chosen"]
    g = dict(window=30, min_queries=10, velocity_thresh=b["velocity_thresh"],
             dispersion_thresh=b["dispersion_thresh"], breadth_thresh=b["breadth_thresh"],
             n_clusters=b.get("n_clusters", 16), degrade_k=1, cooldown_queries=20,
             escalate_after_flags=b.get("escalate_after_flags", 3), client_budget=cal["client_budget"],
             epoch_seconds=1e9)
    g.update(over)
    return {"mode": "guard", "guard": g}


def exp_compare(c, E, attackers, cal, tag):
    specs = {
        "none": {"mode": "none"},
        "ratelimit60": {"mode": "ratelimit"},
        "oracle_v1_default12": {"mode": "oracle_v1", "v1": dict(window_size=30, max_topics_per_window=12,
                                                                 min_queries_before_check=15, cooldown_queries=20)},
        "oracle_v1_tuned5": {"mode": "oracle_v1", "v1": dict(window_size=30, max_topics_per_window=5,
                                                              min_queries_before_check=8, cooldown_queries=20)},
        "guard": tuned_spec(cal),
    }
    res = attack_matrix(c, E, specs, attackers, TEST_SEEDS)
    legit = {}
    for seed in TEST_SEEDS:
        suite = LW.build_suite(c, seed, n_per_cell=10)
        for sname, spec in specs.items():
            if sname == "ratelimit60":
                continue
            legit.setdefault(sname, []).append(legit_summary(c, E, spec, suite))
    print(f"\n[compare] corpus N={len(c)}  ER_final after {MAX_Q} attempts, mean±std over seeds {TEST_SEEDS}")
    names = list(attackers)
    print("  defense               | " + " | ".join(f"{n[:16]:16s}" for n in names))
    for sname in specs:
        print(f"  {sname:21s} | " + " | ".join(f"{fmt(res[sname][n]['er_final']):16s}" for n in names))
    print("\n  legit impact (sessions with ANY degraded/frozen query), protected = focused 1-3 topics:")
    for sname, per_seed in legit.items():
        prot = np.mean([s["_protected"]["any_affected"] for s in per_seed])
        drop = np.mean([s["_protected"]["hit_drop"] for s in per_seed])
        brw = np.mean([np.mean([s[k]["any_affected"] for k in s if k.startswith("browsing")]) for s in per_seed])
        foc5 = np.mean([s["focused5"]["any_affected"] for s in per_seed])
        print(f"  {sname:21s} protected FP={prot:.3f}  hit-rate drop={drop:.3f}  focused5 affected={foc5:.3f}  browsing affected={brw:.3f}")
    out = {"attack": res, "legit": legit, "specs": specs}
    print("  saved", save("compare", out, tag))
    return out


def exp_ablate(c, E, attackers, cal, tag):
    base = tuned_spec(cal)["guard"]
    mods = {
        "budget only": dict(velocity_thresh=None, dispersion_thresh=None, breadth_thresh=None),
        "velocity+dispersion+budget": dict(velocity_thresh=1.5, dispersion_thresh=0.6, breadth_thresh=None),
        "breadth only (no budget)": dict(client_budget=None),
        "breadth+budget, no escalation": dict(escalate_after_flags=0),
        "breadth+velocity+budget": dict(velocity_thresh=1.5),
        "full: breadth+budget+escalation": {},
    }
    specs = {k: {"mode": "guard", "guard": {**base, **v}} for k, v in mods.items()}
    res = attack_matrix(c, E, specs, attackers, TEST_SEEDS)
    legit = {}
    suite = LW.build_suite(c, TEST_SEEDS[0], n_per_cell=10)
    for sname, spec in specs.items():
        legit[sname] = legit_summary(c, E, spec, suite)["_protected"]
    names = list(attackers)
    print(f"\n[ablate] ER_final (lower is better) and protected-legit FP, corpus N={len(c)}")
    print("  variant                     | " + " | ".join(f"{n[:14]:14s}" for n in names) + " | FP_prot")
    for sname in specs:
        print(f"  {sname:27s} | " + " | ".join(f"{fmt(res[sname][n]['er_final']):14s}" for n in names)
              + f" | {legit[sname]['any_affected']:.3f}")
    out = {"attack": res, "legit": legit}
    print("  saved", save("ablate", out, tag))
    return out


def exp_sybil(c, E, attackers, cal, tag):
    base = tuned_spec(cal)["guard"]
    gb = int(round(1.0 * base["client_budget"]))
    specs = {"per-client budget only": {"mode": "guard", "guard": base},
             "per-client + global budget": {"mode": "guard", "guard": {**base, "global_budget": gb}}}
    out = {}
    print(f"\n[sybil] attacker rotates S identities (round robin), snow_frontier; N={len(c)}; global budget={gb}")
    for S in (1, 4, 16):
        out[S] = {}
        for sname, spec in specs.items():
            runs = [run_attack(c, E, spec, attackers["snow_frontier"](s), s, sybil=S) for s in TEST_SEEDS]
            out[S][sname] = agg([r["er_final"] for r in runs])
            print(f"  S={S:2d} {sname:28s} ER_final={fmt(out[S][sname])}")
    print("  saved", save("sybil", out, tag))
    return out


def exp_scale(Es, attackers_by_n, tag):
    """B fixed (set from the legitimate working set), N grows: leakage bound B/N shrinks."""
    out = {}
    print("\n[scale] same budget, growing federated corpus (union of real sources)")
    for label, (c, E, attackers) in attackers_by_n.items():
        suite = LW.build_suite(c, CAL_SEED, n_per_cell=10)
        B = legit_unique_quantile(c, E, suite)
        spec = {"mode": "guard", "guard": dict(window=30, min_queries=10, velocity_thresh=None,
                                               dispersion_thresh=None, client_budget=B, epoch_seconds=1e9,
                                               escalate_after_flags=0)}
        res = {}
        for an in ("snow_frontier", "snow_local_dil3"):
            none_runs = [run_attack(c, E, {"mode": "none"}, attackers[an](s), s, max_q=2500) for s in TEST_SEEDS]
            g_runs = [run_attack(c, E, spec, attackers[an](s), s, max_q=2500) for s in TEST_SEEDS]
            res[an] = {"undefended": agg([r["er_final"] for r in none_runs]), "budget_only": agg([r["er_final"] for r in g_runs])}
        out[label] = {"N": len(c), "budget": B, "bound": round(B / len(c), 4), "attackers": res}
        print(f"  {label:10s} N={len(c):5d} B={B:4d} bound=B/N={B / len(c):.3f}  " +
              "  ".join(f"{an}: undef={r['undefended']['mean']:.3f} guard={r['budget_only']['mean']:.3f}" for an, r in res.items()))
    print("  saved", save("scale", out, tag))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["threat", "signals", "calibrate", "compare", "ablate", "sybil", "scale", "all"])
    ap.add_argument("--source", type=int, default=1, choices=[0, 1, 2])
    args = ap.parse_args()

    E = Embedder()
    c = load_corpus(args.source, E)
    pool = public_pool(args.source)
    attackers = make_attackers(c, E, pool)
    tag = f"source{args.source}"
    t0 = time.time()
    cal = None
    exps = ["threat", "signals", "calibrate", "compare", "ablate", "sybil", "scale"] if args.experiment == "all" else [args.experiment]
    for ex in exps:
        if ex == "threat":
            exp_threat(c, E, attackers, tag)
        elif ex == "signals":
            exp_signals(c, E, attackers, tag)
        elif ex == "calibrate":
            cal = exp_calibrate(c, E, attackers, tag)
        else:
            if cal is None:
                p = os.path.join(OUT_DIR, f"calibrate_{tag}.json")
                cal = json.load(open(p)) if os.path.exists(p) else exp_calibrate(c, E, attackers, tag)
            if ex == "compare":
                exp_compare(c, E, attackers, cal, tag)
            elif ex == "ablate":
                exp_ablate(c, E, attackers, cal, tag)
            elif ex == "sybil":
                exp_sybil(c, E, attackers, cal, tag)
            elif ex == "scale":
                corpora = {"N=500": c}
                others = [load_corpus(s, E) for s in (0, 1, 2) if s != args.source]
                corpora["N=1000"] = union_corpus([c, others[0]])
                corpora["N=1500"] = union_corpus([c] + others)
                byn = {}
                for lab, cc in corpora.items():
                    byn[lab] = (cc, E, make_attackers(cc, E, pool))
                exp_scale(E, byn, tag)
        E.save()
    print(f"\n[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
