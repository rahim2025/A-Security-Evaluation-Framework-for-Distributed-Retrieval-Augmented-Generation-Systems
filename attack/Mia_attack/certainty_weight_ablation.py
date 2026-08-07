"""
attack/Mia_attack/certainty_weight_ablation.py

Answers problems/mia_attack_gaps.md #4's open item directly: CERTAINTY_WEIGHT
is already zeroed in the production composite (Revision 7), but a direct
CERTAINTY_WEIGHT-on ablation *of the production formula itself* -- as
opposed to the separate ABLATION_* diagnostic composite already in
mia_attack.py, which uses different weights entirely (DECISION=0.70,
LEN=0.30) and isn't a certainty-specific test -- was still open.

Runs each seed's real member/non-member probes exactly once (the expensive
part: real LLM calls), then recomputes AUC-ROC twice from the *same*
per-document (match_rate, raw_sim, length_ratio, certainty) signals already
returned by MIAAttack._probe_documents(): once at the production weights
(CERTAINTY_WEIGHT=0.0) and once with CERTAINTY_WEIGHT raised to 0.2 while
keeping DECISION_WEIGHT=1.0 (still satisfies the module's own gate
invariant: DECISION_WEIGHT > SIM_WEIGHT+CERTAINTY_WEIGHT+LEN_WEIGHT). Since
both composites are computed from identical underlying probes, this is a
clean ablation with no resampling confound -- not a rerun of the attack.

Uses fresh seeds not previously used anywhere in this project's dev/test
seed history (report's Revision 7 used 13 dev seeds + 5 fresh test seeds
865/659/693/783/154), so this ablation's seeds are held out from that
history too.
"""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.Mia_attack.mia_attack import (  # noqa: E402
    MIAAttack,
    LLM_SERVICE_URL,
    CORPUS_JSONL,
    DEFAULT_MEMBERS,
    DEFAULT_NONMEMBERS,
    load_membership_documents,
)

SEEDS = [4001, 4002, 4003]
CERTAINTY_WEIGHT_TEST = 0.20  # DECISION_WEIGHT=1.0 still exceeds 0.20, gate holds


def composite(match_rates: List[float], certainties: List[float], certainty_weight: float) -> List[float]:
    decision_weight = 1.0
    return [
        decision_weight * m + certainty_weight * c
        for m, c in zip(match_rates, certainties)
    ]


def run_seed(seed: int) -> dict:
    attack = MIAAttack(
        llm_service_url=LLM_SERVICE_URL, corpus_jsonl=CORPUS_JSONL,
        n_members=DEFAULT_MEMBERS, n_nonmembers=DEFAULT_NONMEMBERS,
        probes_per_doc=4, threshold_percentile=50, api_key="", random_seed=seed,
    )
    members, non_members = load_membership_documents(
        attack.n_members, attack.n_nonmembers, seed, attack.corpus_jsonl, attack.probes_per_doc,
    )
    _, m_match, m_sim, m_len, m_cert = attack._probe_documents(members, "MEMBER")
    _, nm_match, nm_sim, nm_len, nm_cert = attack._probe_documents(non_members, "NON-MEMBER")

    y_true = np.array([1] * len(m_match) + [0] * len(nm_match))

    prod_scores = composite(m_match + nm_match, m_cert + nm_cert, certainty_weight=0.0)
    test_scores = composite(m_match + nm_match, m_cert + nm_cert, certainty_weight=CERTAINTY_WEIGHT_TEST)

    return {
        "seed": seed,
        "auc_production_certainty0": float(roc_auc_score(y_true, prod_scores)),
        "auc_with_certainty_0.2": float(roc_auc_score(y_true, test_scores)),
    }


def main() -> None:
    results = [run_seed(s) for s in SEEDS]
    print(f"\n{'seed':>8}{'AUC (production, CERTAINTY_WEIGHT=0.0)':>42}{'AUC (CERTAINTY_WEIGHT=0.2)':>30}")
    for r in results:
        print(f"{r['seed']:>8}{r['auc_production_certainty0']:>42.4f}{r['auc_with_certainty_0.2']:>30.4f}")
    mean_prod = np.mean([r["auc_production_certainty0"] for r in results])
    mean_test = np.mean([r["auc_with_certainty_0.2"] for r in results])
    print(f"\n{'mean':>8}{mean_prod:>42.4f}{mean_test:>30.4f}")
    print(f"\nDelta (with_certainty - production): {mean_test - mean_prod:+.4f}")


if __name__ == "__main__":
    main()
