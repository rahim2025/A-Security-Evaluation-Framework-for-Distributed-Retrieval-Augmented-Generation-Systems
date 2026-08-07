"""
attack/Mia_attack/run_consistency_pilot.py

Revision 6 pilot -- tests, empirically, an external critique's hypothesis
that _consistency_score() (mia_attack.py) is a better membership signal than
any single-response signal: a member document has a real passage the
retriever can consistently surface, so repeated identical queries should
keep landing on the same yes/no/maybe judgment; a non-member has no backing
passage, so the LLM is guessing/generating from general knowledge each time,
which should vary more across repeated queries.

This is a smaller pilot (default 10 members + 10 non-members, 5 probes each
= 100 extra live LLM calls beyond a standard single-probe run) rather than a
full 25+25 run, since the signal costs n_probes times more LLM calls than
every other signal in this project and its value was, before this pilot,
untested speculation from the critique -- not worth the full-scale cost
until a small sample confirms or refutes the basic hypothesis.

Usage
-----
  python attack/Mia_attack/run_consistency_pilot.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.Mia_attack.mia_attack import (  # noqa: E402
    CORPUS_JSONL,
    LLM_SERVICE_URL,
    _consistency_score,
    load_membership_documents,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_members", type=int, default=10)
    p.add_argument("--n_nonmembers", type=int, default=10)
    p.add_argument("--n_probes", type=int, default=5)
    p.add_argument("--llm_url", default=LLM_SERVICE_URL)
    p.add_argument("--api_key", default="")
    args = p.parse_args()

    members, non_members = load_membership_documents(
        args.n_members, args.n_nonmembers, args.seed, CORPUS_JSONL, probes_per_doc=1,
    )

    def _score_docs(documents, label):
        scores = []
        for i, qa_list in enumerate(documents, 1):
            qa = qa_list[0]  # PubMedQA: ~1 question/doc; use the first
            score = _consistency_score(
                qa["question"], qa.get("decision", ""), args.llm_url, args.api_key,
                n_probes=args.n_probes,
            )
            scores.append(score)
            print(f"  [{label}] doc {i:>2}/{len(documents)}  consistency={score:.2f}  "
                  f"q='{qa['question'][:50]}'")
        return scores

    print(f"=== Consistency-score pilot: seed={args.seed}, "
          f"n_members={args.n_members}, n_nonmembers={args.n_nonmembers}, n_probes={args.n_probes} ===")
    member_scores = _score_docs(members, "MEMBER")
    nonmember_scores = _score_docs(non_members, "NON-MEMBER")

    y_true = np.array([1] * len(member_scores) + [0] * len(nonmember_scores))
    y_scores = np.array(member_scores + nonmember_scores)
    auc = (float(roc_auc_score(y_true, y_scores))
           if len(np.unique(y_true)) > 1 and len(np.unique(y_scores)) > 1 else 0.5)

    result = {
        "seed": args.seed,
        "n_members": len(member_scores),
        "n_non_members": len(nonmember_scores),
        "n_probes": args.n_probes,
        "mean_member_consistency": round(float(np.mean(member_scores)), 4) if member_scores else 0.0,
        "mean_non_member_consistency": round(float(np.mean(nonmember_scores)), 4) if nonmember_scores else 0.0,
        "auc_roc_consistency": round(auc, 4),
    }
    print(f"\n=== Result ===\n{json.dumps(result, indent=2)}")

    out_path = os.path.join(_ROOT, "attack_logs", f"consistency_pilot_seed{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
