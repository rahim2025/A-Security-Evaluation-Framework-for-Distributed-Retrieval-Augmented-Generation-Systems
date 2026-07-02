"""
Run the MIA (Membership Inference Attack) against Reliable-dRAG.

Prerequisites
-------------
  * Docker services running:
      docker compose up -d
  * Python packages:
      pip install sentence-transformers scikit-learn numpy requests
  * Knowledge-base data present:
      data/polluted_token/sources_0.jsonl

Usage
-----
  # Standard run -- 25 members vs 25 non-members, seed 42
  python3 attack/mia/run_attack.py

  # Custom member / non-member counts
  python3 attack/mia/run_attack.py --n_members 25 --n_nonmembers 25

  # With API-key auth enabled on data sources
  python3 attack/mia/run_attack.py --api_key <YOUR_API_KEY>

  # Custom LLM URL (e.g. non-default port)
  python3 attack/mia/run_attack.py --llm_url http://localhost:9000

  # Use a different RNG seed (run >=3 seeds, report mean +/- std)
  python3 attack/mia/run_attack.py --seed 0
  python3 attack/mia/run_attack.py --seed 1
  python3 attack/mia/run_attack.py --seed 2

  # Dry-run: skip LLM calls, generate random similarity scores (sanity check)
  python3 attack/mia/run_attack.py --dry_run

Expected output (thesis -- genuine vulnerability)
-------------------------------------------------
  AUC-ROC   > 0.70
  Accuracy  > 70 %
  mean member similarity significantly higher than non-member

Comparison with random baseline
--------------------------------
  AUC-ROC ~= 0.50  -> no privacy leakage
  AUC-ROC >  0.70  -> genuine vulnerability
"""

import argparse
import datetime
import json
import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
import requests
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from attack.mia.mia_attack import (
    MIAAttack,
    LLM_SERVICE_URL,
    MEMBER_JSONL,
    DEFAULT_MEMBERS,
    DEFAULT_NONMEMBERS,
)

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "attack_logs"))


# ---------------------------------------------------------------------------
# Dry-run helper (no live services needed)
# ---------------------------------------------------------------------------

def _dry_run(n_members: int, n_nonmembers: int, seed: int) -> Dict[str, Any]:
    """
    Simulate MIA with random similarity scores.
    Useful for checking the CLI pipeline without running Docker.
    AUC-ROC should be ~= 0.50 (pure random -- no signal).
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    member_scores     = [rng.uniform(0.0, 1.0) for _ in range(n_members)]
    non_member_scores = [rng.uniform(0.0, 1.0) for _ in range(n_nonmembers)]

    y_true   = np.array([1] * n_members + [0] * n_nonmembers)
    y_scores = np.array(member_scores + non_member_scores)
    threshold = float(np.percentile(y_scores, 50))
    y_pred    = (y_scores >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    auc_roc   = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else 0.5
    accuracy  = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))

    mu_m  = float(np.mean(member_scores))
    mu_nm = float(np.mean(non_member_scores))

    print("\n[DRY-RUN] Using random similarity scores (no LLM calls made)")
    print(f"  AUC-ROC   : {auc_roc:.4f}  (expected ~= 0.50 for random)")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Confusion : TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Mean member sim    : {mu_m:.4f}")
    print(f"  Mean non-member sim: {mu_nm:.4f}")
    print(f"  Similarity delta   : {mu_m - mu_nm:+.4f}")

    return {
        "confusion_matrix":           {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "attack_accuracy":            round(accuracy,  4),
        "precision":                  round(precision, 4),
        "recall":                     round(recall,    4),
        "f1_score":                   round(f1,        4),
        "auc_roc":                    round(auc_roc,   4),
        "privacy_risk":               "DRY-RUN -- random baseline (AUC ~= 0.50)",
        "n_members_tested":           n_members,
        "n_non_members_tested":       n_nonmembers,
        "mean_member_similarity":     round(mu_m,          4),
        "mean_non_member_similarity": round(mu_nm,         4),
        "similarity_delta":           round(mu_m - mu_nm,  4),
    }


# ---------------------------------------------------------------------------
# Log saving
# ---------------------------------------------------------------------------

def save_log(log: Dict[str, Any]) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts    = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    seed  = log.get("attack_config", {}).get("random_seed", "xx")
    fname = os.path.join(LOG_DIR, f"attack_{ts}_mia_seed{seed}.json")
    with open(fname, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)
    return fname


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MIA -- Membership Inference Attack on Reliable-dRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--llm_url", default=LLM_SERVICE_URL,
        help=f"LLM service URL (default: {LLM_SERVICE_URL})",
    )
    p.add_argument(
        "--member_jsonl", default=MEMBER_JSONL,
        help="Path to JSONL file containing member passages",
    )
    p.add_argument(
        "--n_members", type=int, default=DEFAULT_MEMBERS,
        help=f"Number of member passages to probe (default: {DEFAULT_MEMBERS})",
    )
    p.add_argument(
        "--n_nonmembers", type=int, default=DEFAULT_NONMEMBERS,
        help=f"Number of non-member passages to probe (default: {DEFAULT_NONMEMBERS})",
    )
    p.add_argument(
        "--seed_words", type=int, default=12,
        help="Number of words used as probe seed (default: 12)",
    )
    p.add_argument(
        "--threshold_pct", type=int, default=50,
        help="Similarity percentile used as classification threshold (default: 50)",
    )
    p.add_argument(
        "--api_key", default="",
        help="X-API-Key header value if auth is enabled on data sources",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility (default: 42). Run seeds 0,1,2 for thesis.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Skip LLM calls; use random scores (sanity check -- AUC ~= 0.50)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 62)
    print("  MIA -- Membership Inference Attack | Reliable-dRAG")
    print("=" * 62)
    print(f"  LLM service URL : {args.llm_url}")
    print(f"  Member JSONL    : {args.member_jsonl}")
    print(f"  Members         : {args.n_members}")
    print(f"  Non-members     : {args.n_nonmembers}")
    print(f"  Seed words      : {args.seed_words}")
    print(f"  Threshold pct   : {args.threshold_pct}")
    print(f"  RNG seed        : {args.seed}")
    print(f"  Auth key        : {'<set>' if args.api_key else '<not set>'}")
    print(f"  Dry run         : {args.dry_run}")

    ts = datetime.datetime.now().isoformat()

    if args.dry_run:
        metrics = _dry_run(args.n_members, args.n_nonmembers, args.seed)
    else:
        attack = MIAAttack(
            llm_service_url      = args.llm_url,
            member_jsonl         = args.member_jsonl,
            n_members            = args.n_members,
            n_nonmembers         = args.n_nonmembers,
            seed_words           = args.seed_words,
            threshold_percentile = args.threshold_pct,
            api_key              = args.api_key,
            random_seed          = args.seed,
        )
        metrics = attack.run()

    log = {
        "timestamp":   ts,
        "attack_type": "membership_inference",
        "attack_config": {
            "llm_service_url":  args.llm_url,
            "member_jsonl":     args.member_jsonl,
            "n_members":        args.n_members,
            "n_nonmembers":     args.n_nonmembers,
            "seed_words":       args.seed_words,
            "threshold_pct":    args.threshold_pct,
            "random_seed":      args.seed,
            "dry_run":          args.dry_run,
        },
        "metrics": metrics,
    }

    log_path = save_log(log)
    print(f"\n  Log saved -> {log_path}")

    print("\n  === Thesis Summary ===")
    print(f"  AUC-ROC     : {metrics['auc_roc']:.4f}  "
          f"({'> 0.70 -> genuine vulnerability' if metrics['auc_roc'] > 0.70 else '~= 0.50 -> no signal'})")
    print(f"  Accuracy    : {metrics['attack_accuracy']:.4f}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  F1          : {metrics['f1_score']:.4f}")
    print(f"  Privacy risk: {metrics['privacy_risk']}")
    print(f"  Sim delta   : {metrics['similarity_delta']:+.4f}  "
          "(positive -> members more exposed)")


if __name__ == "__main__":
    main()
