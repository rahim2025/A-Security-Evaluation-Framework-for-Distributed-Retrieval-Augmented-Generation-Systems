"""
attack/Mia_attack/run_ablation_eval.py

Revision 6 diagnostic tool -- NOT the production attack (that remains
mia_attack.py / run_attack.py, whose gated composite and weights are
unchanged). This script exists to test, empirically, three hypotheses raised
against Revision 5's held-out-seed finding (reports/MIA_Security_Analysis_Report.md
§12.10): the composite's held-out AUC (0.453, below chance) might be
explained by (a) SIM_WEIGHT/CERTAINTY_WEIGHT adding noise rather than signal,
or (b) the positional `_decision_match` gate being too easy to depress on
held-out seeds where the LLM phrases its commitment differently. Both are
testable without touching the production composite: this script computes, on
the SAME probe set as the production attack, both the production gated
composite AND an ablation composite (decision-dominant, sim/certainty
zeroed) plus `_decision_match_semantic` (full-response + synonyms) as
side-by-side diagnostics, on a documented, reproducible held-out seed list.

Held-out seed provenance (documented per Revision 5's honesty note that
seeds 7/13 were arbitrary, not formally randomized):
    seeds = [s for s in random.Random(20260709).sample(range(1000), 20)
             if s not in (0, 1, 42, 7, 13)][:8]
    -> [880, 496, 356, 820, 741, 926, 912, 662]
Master seed 20260709 is simply this experiment's run date (2026-07-09),
recorded here so the list is independently reproducible by anyone re-running
this exact line, not because the date has any other significance.

Usage
-----
  python attack/Mia_attack/run_ablation_eval.py --seed 880
  python attack/Mia_attack/run_ablation_eval.py --seed 880 --seed 496 ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.Mia_attack.mia_attack import (  # noqa: E402
    ABLATION_CERTAINTY_WEIGHT,
    ABLATION_DECISION_WEIGHT,
    ABLATION_LEN_WEIGHT,
    ABLATION_SIM_WEIGHT,
    CERTAINTY_WEIGHT,
    CORPUS_JSONL,
    DECISION_WEIGHT,
    DEFAULT_MEMBERS,
    DEFAULT_NONMEMBERS,
    LEN_WEIGHT,
    LLM_SERVICE_URL,
    SIM_WEIGHT,
    _answer_length_ratio,
    _certainty_score,
    _cosine_similarity,
    _decision_match,
    _decision_match_adaptive,
    _decision_match_semantic,
    _normalize_similarity,
    _query_llm,
    load_membership_documents,
)

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install sentence-transformers") from exc

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

HELD_OUT_SEEDS_REVISION_6 = [880, 496, 356, 820, 741, 926, 912, 662]


def _auc_or_half(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) > 1 and len(np.unique(scores)) > 1:
        return float(roc_auc_score(y_true, scores))
    return 0.5


def run_seed(seed: int, llm_url: str, api_key: str, encoder: SentenceTransformer,
             n_members: int = DEFAULT_MEMBERS, n_nonmembers: int = DEFAULT_NONMEMBERS,
             probes_per_doc: int = 4) -> Dict[str, Any]:
    members, non_members = load_membership_documents(
        n_members, n_nonmembers, seed, CORPUS_JSONL, probes_per_doc,
    )

    primary_composite, ablation_composite = [], []
    decision_positional, decision_adaptive, decision_semantic = [], [], []
    sims, lens_, certs = [], [], []

    for label, documents in (("MEMBER", members), ("NON-MEMBER", non_members)):
        print(f"  [ablation-eval] seed={seed}  {label}: {len(documents)} documents")
        for i, qa_list in enumerate(documents, 1):
            probe_sims, probe_lens, probe_certs = [], [], []
            probe_pos, probe_adapt, probe_sem = [], [], []
            for qa in qa_list:
                response = _query_llm(qa["question"], llm_url, api_key)
                gold_decision = qa.get("decision", "")
                if response.strip():
                    emb_r = encoder.encode([response], convert_to_numpy=True)[0]
                    emb_c = encoder.encode([qa["context"]], convert_to_numpy=True)[0]
                    sim = _cosine_similarity(emb_r, emb_c)
                else:
                    sim = 0.0
                probe_sims.append(sim)
                probe_lens.append(_answer_length_ratio(response, qa["answer"]))
                probe_certs.append(_certainty_score(response))
                probe_pos.append(1.0 if _decision_match(response, gold_decision) else 0.0)
                probe_adapt.append(1.0 if _decision_match_adaptive(response, gold_decision) else 0.0)
                probe_sem.append(_decision_match_semantic(response, gold_decision))

            best = int(np.argmax(probe_sims)) if probe_sims else 0
            sim = probe_sims[best] if probe_sims else 0.0
            length = probe_lens[best] if probe_lens else 0.0
            cert = probe_certs[best] if probe_certs else 0.0
            pos_rate = (sum(probe_pos) / len(probe_pos)) if probe_pos else 0.0
            adapt_rate = (sum(probe_adapt) / len(probe_adapt)) if probe_adapt else 0.0
            sem_rate = (sum(probe_sem) / len(probe_sem)) if probe_sem else 0.0

            primary_composite.append(
                DECISION_WEIGHT * pos_rate + SIM_WEIGHT * _normalize_similarity(sim)
                + CERTAINTY_WEIGHT * cert + LEN_WEIGHT * length
            )
            ablation_composite.append(
                ABLATION_DECISION_WEIGHT * pos_rate + ABLATION_SIM_WEIGHT * _normalize_similarity(sim)
                + ABLATION_CERTAINTY_WEIGHT * cert + ABLATION_LEN_WEIGHT * length
            )
            decision_positional.append(pos_rate)
            decision_adaptive.append(adapt_rate)
            decision_semantic.append(sem_rate)
            sims.append(sim)
            lens_.append(length)
            certs.append(cert)

    n_m, n_nm = len(members), len(non_members)
    y_true = np.array([1] * n_m + [0] * n_nm)

    result = {
        "seed": seed,
        "n_members": n_m,
        "n_non_members": n_nm,
        "auc_primary_composite": round(_auc_or_half(y_true, np.array(primary_composite)), 4),
        "auc_ablation_composite": round(_auc_or_half(y_true, np.array(ablation_composite)), 4),
        "auc_decision_positional": round(_auc_or_half(y_true, np.array(decision_positional)), 4),
        "auc_decision_adaptive": round(_auc_or_half(y_true, np.array(decision_adaptive)), 4),
        "auc_decision_semantic": round(_auc_or_half(y_true, np.array(decision_semantic)), 4),
        "auc_similarity": round(_auc_or_half(y_true, np.array(sims)), 4),
        "auc_length_ratio": round(_auc_or_half(y_true, np.array(lens_)), 4),
        "auc_certainty": round(_auc_or_half(y_true, np.array(certs)), 4),
    }
    print(f"  [ablation-eval] seed={seed} DONE -> {json.dumps(result)}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, action="append", default=None,
                   help="Seed(s) to evaluate; repeat flag for multiple. "
                        "Default: all of HELD_OUT_SEEDS_REVISION_6.")
    p.add_argument("--llm_url", default=LLM_SERVICE_URL)
    p.add_argument("--api_key", default="")
    p.add_argument("--n_members", type=int, default=DEFAULT_MEMBERS)
    p.add_argument("--n_nonmembers", type=int, default=DEFAULT_NONMEMBERS)
    args = p.parse_args()

    seeds = args.seed if args.seed else HELD_OUT_SEEDS_REVISION_6
    print(f"[ablation-eval] Loading encoder: {EMBEDDING_MODEL}")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    results = []
    for seed in seeds:
        results.append(run_seed(seed, args.llm_url, args.api_key, encoder,
                                 args.n_members, args.n_nonmembers))

    out_path = os.path.join(_ROOT, "attack_logs", "ablation_eval_revision6.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[ablation-eval] Wrote {out_path}")

    aucs_primary = [r["auc_primary_composite"] for r in results]
    aucs_ablation = [r["auc_ablation_composite"] for r in results]
    aucs_pos = [r["auc_decision_positional"] for r in results]
    aucs_sem = [r["auc_decision_semantic"] for r in results]
    print("\n=== Summary across seeds ===")
    print(f"Primary composite  mean AUC: {np.mean(aucs_primary):.4f}  (per-seed: {aucs_primary})")
    print(f"Ablation composite mean AUC: {np.mean(aucs_ablation):.4f}  (per-seed: {aucs_ablation})")
    print(f"decision_match (positional) mean AUC: {np.mean(aucs_pos):.4f}  (per-seed: {aucs_pos})")
    print(f"decision_match_semantic     mean AUC: {np.mean(aucs_sem):.4f}  (per-seed: {aucs_sem})")


if __name__ == "__main__":
    main()
