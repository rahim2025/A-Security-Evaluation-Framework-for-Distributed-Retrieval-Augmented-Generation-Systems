"""
defense/mia_defense/run_defense.py
Evaluate the MIA response-sanitization defense against the live Reliable-dRAG
LLM service, using the exact same member/non-member QA sampling as
attack/Mia_attack: real SQuAD questions, member context loaded into the
corpus, non-member context genuinely absent from every source.

Usage
-----
  python defense/mia_defense/run_defense.py
  python defense/mia_defense/run_defense.py --seed 0
  python defense/mia_defense/run_defense.py --n_members 25 --n_nonmembers 25
  python defense/mia_defense/run_defense.py --api_key reliable-derag-secret-2026
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.Mia_attack.mia_attack import (  # noqa: E402
    CORPUS_JSONL,
    DEFAULT_MEMBERS,
    DEFAULT_NONMEMBERS,
    LLM_SERVICE_URL,
)
from mia_defense import MIADefenseEvaluator, MAX_OVERLAP_WORDS, MAX_RESPONSE_WORDS  # noqa: E402

LOG_DIR = os.path.join(_ROOT, "defense_logs")
os.makedirs(LOG_DIR, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MIA Defense evaluation | Reliable-dRAG")
    p.add_argument("--llm_url", default=LLM_SERVICE_URL)
    p.add_argument("--corpus_jsonl", default=CORPUS_JSONL)
    p.add_argument("--n_members", type=int, default=DEFAULT_MEMBERS)
    p.add_argument("--n_nonmembers", type=int, default=DEFAULT_NONMEMBERS)
    p.add_argument("--probes_per_doc", type=int, default=4)
    p.add_argument("--threshold_pct", type=int, default=50)
    p.add_argument("--api_key", default="")
    p.add_argument("--seed", type=int, default=42, help="Run 0,1,2 for thesis mean +/- std")
    p.add_argument("--max_response_words", type=int, default=MAX_RESPONSE_WORDS)
    p.add_argument("--max_overlap_words", type=int, default=MAX_OVERLAP_WORDS)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 62)
    print("  MIA Defense Evaluation | Reliable-dRAG")
    print("=" * 62)
    print(f"  LLM service URL   : {args.llm_url}")
    print(f"  Corpus JSONL      : {args.corpus_jsonl}")
    print(f"  Members           : {args.n_members}")
    print(f"  Non-members       : {args.n_nonmembers}")
    print(f"  RNG seed          : {args.seed}")
    print(f"  Max response words: {args.max_response_words}")
    print(f"  Max overlap words : {args.max_overlap_words}")

    evaluator = MIADefenseEvaluator(
        llm_service_url=args.llm_url,
        corpus_jsonl=args.corpus_jsonl,
        n_members=args.n_members,
        n_nonmembers=args.n_nonmembers,
        probes_per_doc=args.probes_per_doc,
        threshold_percentile=args.threshold_pct,
        api_key=args.api_key,
        random_seed=args.seed,
        max_response_words=args.max_response_words,
        max_overlap_words=args.max_overlap_words,
    )
    result = evaluator.run()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log = {
        "timestamp": ts,
        "defense_type": "mia_response_sanitization",
        "config": {
            "llm_service_url": args.llm_url,
            "corpus_jsonl": args.corpus_jsonl,
            "n_members": args.n_members,
            "n_nonmembers": args.n_nonmembers,
            "probes_per_doc": args.probes_per_doc,
            "threshold_pct": args.threshold_pct,
            "random_seed": args.seed,
            "max_response_words": args.max_response_words,
            "max_overlap_words": args.max_overlap_words,
        },
        "result": result,
    }
    log_path = os.path.join(LOG_DIR, f"defense_{ts}_mia_seed{args.seed}.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)
    print(f"\n  Log saved -> {log_path}")


if __name__ == "__main__":
    main()
