"""
defense/kb_extraction_defense/evaluation/live_eval.py

Live validation of the KB-extraction guard against the real Docker data sources,
over real HTTP, with the same adaptive attackers used in offline_eval.py.

Before / after on ONE deployment, no container recreate between them:
  after  (defended)   plain X-API-Key  -> the caller is an ordinary key holder: guarded.
  before (undefended) X-API-Key + X-Orchestrator-Token -> exempt, i.e. exactly the old
                      behaviour (full top-5 documents to anyone who asks).
Both go through the identical retrieval code; only the guard differs.

Preconditions
  docker compose up -d   (data sources rebuilt with the guard; see README "Running")
  RATE_LIMIT_DEFAULT raised on the data sources (e.g. "100000 per minute") so the
  deployed Flask-Limiter (60/min) does not contaminate the measurement: a rate-limited
  request is indistinguishable from a guard-blocked one. This script aborts if it sees a 429.

  python defense/kb_extraction_defense/evaluation/live_eval.py --source 1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.kb_extraction.adaptive_attack import QueryResult, SnowballCrawler, StaticProbes  # noqa: E402
from defense.kb_extraction_defense.evaluation import legit_workloads as LW  # noqa: E402
from defense.kb_extraction_defense.evaluation.eval_common import Embedder, load_corpus  # noqa: E402
from defense.kb_extraction_defense.evaluation.offline_eval import TEST_SEEDS, agg, fmt, public_pool  # noqa: E402

API_KEY = os.getenv("API_KEY", "reliable-derag-secret-2026")
ORCH_TOKEN = os.getenv("ORCHESTRATOR_TOKEN", "orch-secret-2026")
URLS = {0: "http://localhost:8001", 1: "http://localhost:8002", 2: "http://localhost:8003"}
OUT = os.path.join(_ROOT, "defense_logs", "kb_extraction_defense", "live")


class RateLimited(RuntimeError):
    pass


class HttpTransport:
    def __init__(self, base: str, trusted: bool):
        self.base = base
        self.h = {"Content-Type": "application/json", "X-API-Key": API_KEY}
        if trusted:
            self.h["X-Orchestrator-Token"] = ORCH_TOKEN
        self.actions: List[str] = []

    def query(self, text: str, k: int = 5) -> QueryResult:
        r = requests.post(f"{self.base}/query", headers=self.h, json={"query": text, "k": k}, timeout=30)
        if r.status_code == 429:
            raise RateLimited("HTTP 429 (Flask-Limiter): raise RATE_LIMIT_DEFAULT on the data sources")
        self.actions.append(r.headers.get("X-Guard-Action", "n/a"))
        if r.status_code == 403:
            return QueryResult("blocked", [])
        r.raise_for_status()
        return QueryResult("ok", [{"id": d["id"], "text": d["text"]} for d in r.json().get("results", [])])


def reset(base: str) -> None:
    """Guard-only reset (the corpus is never modified by this evaluation)."""
    r = requests.post(f"{base}/extraction_guard/reset", headers={"X-API-Key": API_KEY}, timeout=60)
    r.raise_for_status()


def guard_stats(base: str) -> dict:
    return requests.get(f"{base}/extraction_guard/stats", headers={"X-API-Key": API_KEY}, timeout=30).json()


def run_attack(base, attacker, seed, trusted, n_corpus, max_q):
    tp = HttpTransport(base, trusted)
    tr = attacker.run(tp, max_q, seed)
    first = next((i for i, a in enumerate(tp.actions) if a not in ("allow", "n/a")), None)
    return {"er_final": round(len(tr.extracted_ids()) / n_corpus, 4), "attempts": len(tr.records),
            "first_flag": first, "docs": len(tr.extracted_ids())}


def run_legit(base, sess, trusted=False):
    tp = HttpTransport(base, trusted=trusted)
    hit = 0
    for q in sess.queries:
        res = tp.query(q.question, 5)
        # live servers return integer doc ids, the evaluation corpus uses string ids
        hit += str(q.doc_id) in {str(d["id"]) for d in res.docs}
    aff = sum(1 for a in tp.actions if a not in ("allow", "n/a"))
    return {"hit": hit / len(sess.queries), "affected": aff / len(sess.queries), "any_affected": aff > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--max_q", type=int, default=600)
    ap.add_argument("--legit_per_cell", type=int, default=2)
    args = ap.parse_args()
    base = URLS[args.source]

    E = Embedder()
    c = load_corpus(args.source, E)
    pool = public_pool(args.source)
    N = len(c)

    def snow(sel, dil):
        return lambda s: SnowballCrawler(sel, random.Random(s).sample(pool, 8), embed_fn=E.embed, dilution=dil)

    attackers = {
        "static54": lambda s: StaticProbes([q.question for q in random.Random(s).sample(c.qa, 54)]),
        "snow_random": snow("random", 0),
        "snow_frontier": snow("frontier", 0),
        "snow_local_dil3": snow("local", 3),
    }
    info = requests.get(f"{base}/info", headers={"X-API-Key": API_KEY}, timeout=30).json()
    print(f"target {base}  docs={info['total_docs']}  guard={guard_stats(base).get('enabled')}")
    out = {"source": args.source, "N": N, "attack": {}, "legit": {}}

    for an, mk in attackers.items():
        for mode, trusted in (("undefended(token)", True), ("guarded", False)):
            runs = []
            for s in TEST_SEEDS:
                reset(base)
                runs.append(run_attack(base, mk(s), s, trusted, N, args.max_q))
            out["attack"][f"{an}|{mode}"] = {"er_final": agg([r["er_final"] for r in runs]),
                                             "first_flag": agg([r["first_flag"] for r in runs]), "runs": runs}
            print(f"  {an:18s} {mode:18s} ER_final={fmt(out['attack'][f'{an}|{mode}']['er_final'])}  "
                  f"first_flag={out['attack'][f'{an}|{mode}']['first_flag']['mean']}")

    # legitimate focused sessions, live
    rng = random.Random(7)
    sess = [LW.focused(c, rng, nt, ln) for nt in (1, 2, 3) for ln in (20, 50, 100, 200) for _ in range(args.legit_per_cell)]
    reset(base)
    rs, rs_base = [], []
    for s in sess:
        reset(base)
        rs.append(run_legit(base, s))
        reset(base)
        rs_base.append(run_legit(base, s, trusted=True))
    out["legit"] = {"sessions": len(rs), "mean_hit_undefended": round(sum(r["hit"] for r in rs_base) / len(rs_base), 4), "any_affected": round(sum(r["any_affected"] for r in rs) / len(rs), 4),
                    "mean_affected_frac": round(sum(r["affected"] for r in rs) / len(rs), 4),
                    "mean_hit": round(sum(r["hit"] for r in rs) / len(rs), 4)}
    print("  legit focused sessions (live):", out["legit"])
    out["guard_stats_final"] = guard_stats(base)
    reset(base)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"live_source{args.source}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(out, open(p, "w"), indent=1, default=str)
    print("saved", p)


if __name__ == "__main__":
    main()
