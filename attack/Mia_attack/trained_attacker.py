"""
attack/Mia_attack/trained_attacker.py

Revision 9, Phase 1 -- "train a classifier instead of a hand-designed
heuristic" (NAACL reviewer feedback, see
reports/updated_reports_safin/README.md for the full writeup).

Why this file exists
---------------------
Revision 7 (reports/MIA_Security_Analysis_Report.md, section 2.9) fixed the
composite's generalization problem with a grid search: it searched a fixed
functional form (a weighted linear sum of four signals, weights on a 0.05
grid) for the best-generalizing weight vector, found pure `decision_match`
(weight 1.0) won, and confirmed that on 5 fresh held-out seeds (mean AUC
0.620, 95% CI [0.546, 0.695]).

That is a *tuned* heuristic, not a *trained* one -- the functional form
(linear combination) and the search procedure (grid search over a simplex)
were both fixed by hand before any data was looked at. External reviewer
feedback ahead of NAACL submission asked for a genuinely trained classifier
instead: something that can learn a non-linear decision boundary and
feature interactions from the data itself, evaluated with the same
train/test discipline Revision 7 already established.

This file does exactly that, and nothing more:
  - Reuses the *already-collected* raw per-document signal data from
    Revision 7 (`attack_logs/tune_weights_dev_signals.json` and
    `tune_weights_test_signals.json`) -- the same DEV_SEEDS (13 seeds) /
    TEST_SEEDS (5 fresh seeds) split, the same 4 raw signals
    (norm_sim, certainty, length_ratio, decision_match) per document.
    No new LLM calls are made by this script -- see
    `contrastive_probes.py` (Phase 2) for the follow-up that *does*
    require fresh live queries, to add the richer wording/contrastive
    features the reviewer also asked for.
  - Trains real supervised classifiers (logistic regression and gradient
    boosting) on that data, model-selected via Leave-One-Seed-Out
    cross-validation *within the dev set only* -- never touching
    TEST_SEEDS during selection, the same discipline `tune_weights.py`
    used for its weight grid search.
  - Locks in exactly one model (whichever wins the dev-only CV), then
    evaluates it exactly once against the 5 reserved TEST_SEEDS -- mirrors
    `tune_weights.py cmd_evaluate` bit for bit, including the "run this
    exactly once" discipline and refusing to evaluate on any seed not in
    the reserved TEST_SEEDS list.

What this script deliberately does NOT do
-------------------------------------------
- It does not add new features. The feature set is still only the four
  signals Revision 7 already computed (norm_sim, certainty, length_ratio,
  decision_match) -- a classifier over four thin scalar features is a
  real improvement over a fixed linear form (it can learn a non-linear
  boundary and interactions the grid search could never express), but it
  is not yet the "wording, confidence, similarity, and multiple
  contrastive questions per document" attacker the reviewer described.
  That is Phase 2 (`contrastive_probes.py`), which needs fresh LLM calls
  this script's data does not contain.
- It does not re-run defense evaluation against this stronger attacker.
  That is explicitly out of scope for this pass -- see
  reports/updated_reports_safin/README.md, "What this pass does NOT do".
- It has not been executed in this session. Running it requires the
  conda env with a working scikit-learn build (`/opt/miniconda3/bin/python3`
  on this machine -- the project's system Python has an architecture-
  mismatched scikit-learn install, see the README) -- but no live Docker
  services, since it only reads the JSON files Revision 7 already wrote.

Two commands, mirroring tune_weights.py's collect/search/evaluate split
(this script has no `collect` step -- it reads tune_weights.py's dev/test
JSON directly):

  python trained_attacker.py search     # Leave-One-Seed-Out CV on DEV_SEEDS
                                         # only; selects + locks one model
  python trained_attacker.py evaluate   # score the locked model against
                                         # TEST_SEEDS -- run this exactly once
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.Mia_attack.tune_weights import (  # noqa: E402
    DEV_DATA_PATH,
    DEV_SEEDS,
    TEST_DATA_PATH,
    TEST_SEEDS,
)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This script needs a working scikit-learn build. On this machine "
        "the project's system Python (/Library/Frameworks/Python.framework/"
        "Versions/3.11/bin/python3) has an architecture-mismatched "
        "scikit-learn install (arm64 interpreter, x86_64 compiled "
        "extension) -- use the conda base env instead: "
        "/opt/miniconda3/bin/python3 trained_attacker.py ...\n"
        f"Original import error: {exc}"
    ) from exc

_DATA_DIR = os.path.join(_ROOT, "attack_logs")
LOCKED_MODEL_PATH = os.path.join(_DATA_DIR, "trained_attacker_locked.pkl")
LOCKED_META_PATH = os.path.join(_DATA_DIR, "trained_attacker_locked_meta.json")
FINAL_RESULT_PATH = os.path.join(_DATA_DIR, "trained_attacker_final_result.json")

FEATURE_NAMES = ["norm_sim", "certainty", "length_ratio", "decision_match"]

# t critical values, df = n-1, two-sided 95% -- same small hardcoded table
# tune_weights.py uses, kept identical so the two scripts' CIs are computed
# the same way and are directly comparable.
_T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _rows_to_xy(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows], dtype=int)
    groups = np.array([r["seed"] for r in rows], dtype=int)
    return X, y, groups


def _mean_per_seed_auc_from_predictions(
    y: np.ndarray, scores: np.ndarray, groups: np.ndarray,
) -> float:
    """
    Same aggregation `tune_weights.py`'s `_mean_per_seed_auc` uses: compute
    AUC *within each seed* (not pooled across seeds), then average the
    per-seed AUCs. Pooling across seeds first would let a seed with a more
    separable member/non-member split dominate the metric; per-seed
    averaging treats every seed as one independent trial, matching how
    every other AUC figure in this project's reports is computed.
    """
    seeds = sorted(set(groups.tolist()))
    aucs = []
    for seed in seeds:
        mask = groups == seed
        y_s, s_s = y[mask], scores[mask]
        if len(np.unique(y_s)) > 1 and len(np.unique(s_s)) > 1:
            aucs.append(roc_auc_score(y_s, s_s))
        else:
            aucs.append(0.5)
    return float(np.mean(aucs))


def _candidate_models() -> Dict[str, Any]:
    """
    Two candidate classifier families, both defensible for ~650 rows / 4
    features / 13 groups:
      - Logistic regression: linear, low-variance, directly interpretable
        coefficients (can be reported per-signal the way the grid-searched
        weights were) -- the natural "trained version" of the existing
        linear composite.
      - Gradient boosting (shallow trees, small ensemble, regularized):
        can capture non-linear thresholds and feature interactions a
        linear model can't -- e.g. "high decision_match AND high
        certainty" mattering more than either alone -- at real risk of
        overfitting 4 features over 650 rows, which is exactly why model
        *selection* is Leave-One-Seed-Out CV on dev, not a fixed choice.
    Both are deliberately small/regularized given the dataset size; this
    is not a search over many classifier families or hyperparameters --
    a wider search would need more dev data than 650 rows across 13 seeds
    to avoid the exact overfitting-to-the-search-process risk Revision 7's
    docstring already discusses for the linear weight grid.
    """
    return {
        "logreg_l2": LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, random_state=0,
        ),
        "logreg_l2_strong": LogisticRegression(
            penalty="l2", C=0.1, max_iter=2000, random_state=0,
        ),
        "gboost_shallow": GradientBoostingClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=0,
        ),
    }


def cmd_search(args: argparse.Namespace) -> None:
    rows = _load_rows(DEV_DATA_PATH)
    seeds_present = sorted(set(r["seed"] for r in rows))
    print(f"[trained-attacker] Loaded {len(rows)} dev rows across seeds: {seeds_present}")
    assert set(seeds_present) <= set(DEV_SEEDS), (
        "Dev signal file contains a seed outside DEV_SEEDS -- refusing to "
        "search, this would silently expand the training set beyond what "
        "Revision 7 defined as the dev split."
    )

    X, y, groups = _rows_to_xy(rows)
    logo = LeaveOneGroupOut()

    print(f"\n[trained-attacker] Leave-One-Seed-Out CV over {len(_candidate_models())} "
          f"candidate models, {len(seeds_present)} folds each (one held seed per fold, "
          f"the rest used to fit) -- selection uses DEV_SEEDS only, TEST_SEEDS never touched:\n")

    results = []
    for name, model_template in _candidate_models().items():
        fold_scores = []
        for train_idx, held_idx in logo.split(X, y, groups):
            X_train, y_train = X[train_idx], y[train_idx]
            X_held, y_held = X[held_idx], y[held_idx]
            held_seed = int(groups[held_idx][0])

            scaler = StandardScaler().fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_held_s = scaler.transform(X_held)

            # Clone-by-reconstruction so each fold gets a fresh, unfit model
            # instance -- fitting the same object repeatedly across folds
            # would silently carry state between them for some estimators.
            model = type(model_template)(**model_template.get_params())
            if len(np.unique(y_train)) < 2:
                # Degenerate fold guard (shouldn't happen with 12 seeds x 50
                # docs feeding the training side, but fail safe rather than
                # crash the whole search on one bad fold).
                fold_scores.append((held_seed, 0.5))
                continue
            model.fit(X_train_s, y_train)
            probs = model.predict_proba(X_held_s)[:, 1]
            auc = (
                roc_auc_score(y_held, probs)
                if len(np.unique(y_held)) > 1 and len(np.unique(probs)) > 1
                else 0.5
            )
            fold_scores.append((held_seed, float(auc)))

        mean_auc = float(np.mean([s for _, s in fold_scores]))
        std_auc = float(np.std([s for _, s in fold_scores], ddof=1))
        results.append((mean_auc, std_auc, name, fold_scores))
        per_seed_str = ", ".join(f"{seed}:{auc:.3f}" for seed, auc in sorted(fold_scores))
        print(f"  {name:<20s}  LOSO-CV mean AUC = {mean_auc:.4f}  (std {std_auc:.4f})")
        print(f"    per-seed: {per_seed_str}")

    results.sort(key=lambda x: -x[0])
    winner_mean, winner_std, winner_name, winner_folds = results[0]
    print(f"\n[trained-attacker] WINNER (dev-only LOSO-CV): {winner_name}  "
          f"mean AUC = {winner_mean:.4f}  (std {winner_std:.4f})")
    print("[trained-attacker] Compare to Revision 7's dev-set grid-search score: 0.5723 "
          "(pure decision_match, reports/MIA_Security_Analysis_Report.md sec 2.9)")

    # Refit the winning model type on ALL of DEV_SEEDS (no held-out fold
    # this time -- LOSO-CV already estimated its generalization; this final
    # fit uses every dev row to give the locked model the most data
    # possible before its one-shot evaluation against TEST_SEEDS).
    final_scaler = StandardScaler().fit(X)
    X_all_s = final_scaler.transform(X)
    winner_template = _candidate_models()[winner_name]
    final_model = type(winner_template)(**winner_template.get_params())
    final_model.fit(X_all_s, y)

    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(LOCKED_MODEL_PATH, "wb") as fh:
        pickle.dump({"scaler": final_scaler, "model": final_model,
                     "feature_names": FEATURE_NAMES, "model_name": winner_name}, fh)

    meta = {
        "model_name": winner_name,
        "feature_names": FEATURE_NAMES,
        "dev_seeds": DEV_SEEDS,
        "loso_cv_mean_auc": round(winner_mean, 4),
        "loso_cv_std_auc": round(winner_std, 4),
        "loso_cv_per_seed": {str(s): round(a, 4) for s, a in winner_folds},
        "all_candidates": [
            {"model_name": n, "loso_cv_mean_auc": round(m, 4), "loso_cv_std_auc": round(sd, 4)}
            for m, sd, n, _ in results
        ],
        "reference_dev_score_revision7_grid_search": 0.5723,
    }
    if hasattr(final_model, "coef_"):
        meta["final_model_coefficients"] = {
            f: round(float(c), 4) for f, c in zip(FEATURE_NAMES, final_model.coef_[0])
        }
        meta["final_model_intercept"] = round(float(final_model.intercept_[0]), 4)
    with open(LOCKED_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n[trained-attacker] Locked model written -> {LOCKED_MODEL_PATH}")
    print(f"[trained-attacker] Metadata written -> {LOCKED_META_PATH}")
    print("[trained-attacker] Model is now FROZEN. Next step: evaluate -- run exactly once.")


def cmd_evaluate(args: argparse.Namespace) -> None:
    with open(LOCKED_MODEL_PATH, "rb") as fh:
        locked = pickle.load(fh)
    scaler, model, model_name = locked["scaler"], locked["model"], locked["model_name"]
    print(f"[trained-attacker] Evaluating LOCKED model (selected on DEV_SEEDS LOSO-CV only): "
          f"{model_name}")

    rows = _load_rows(TEST_DATA_PATH)
    test_seeds_present = sorted(set(r["seed"] for r in rows))
    print(f"[trained-attacker] Loaded {len(rows)} test rows across seeds: {test_seeds_present}")
    assert set(test_seeds_present) <= set(TEST_SEEDS), (
        "Test data contains a seed not in the reserved TEST_SEEDS list -- "
        "refusing to evaluate, this would not be a fair held-out test."
    )

    X, y, groups = _rows_to_xy(rows)
    X_s = scaler.transform(X)
    probs = model.predict_proba(X_s)[:, 1]

    per_seed = {}
    for seed in test_seeds_present:
        mask = groups == seed
        y_s, p_s = y[mask], probs[mask]
        auc = (
            roc_auc_score(y_s, p_s)
            if len(np.unique(y_s)) > 1 and len(np.unique(p_s)) > 1 else 0.5
        )
        per_seed[seed] = round(float(auc), 4)

    aucs = list(per_seed.values())
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
    n = len(aucs)
    se = std_auc / (n ** 0.5) if n > 1 else 0.0
    t_crit = _T_TABLE.get(n - 1, 1.96)
    ci = (round(mean_auc - t_crit * se, 4), round(mean_auc + t_crit * se, 4))

    print(f"\n[trained-attacker] TRUE HELD-OUT RESULT (never seen during search) -- "
          f"per seed: {per_seed}")
    print(f"[trained-attacker] Mean AUC on fresh test seeds: {mean_auc:.4f}   95% CI: {ci}")
    print("[trained-attacker] Compare to production linear composite (Revision 7/8, "
          "pure decision_match): mean AUC 0.6200, 95% CI [0.5455, 0.6945] "
          "(reports/MIA_Security_Analysis_Report.md sec 2.9)")

    out = {
        "model_name": model_name,
        "per_seed_auc": per_seed,
        "mean_auc": round(mean_auc, 4),
        "std_auc": round(std_auc, 4),
        "ci_95": list(ci),
        "n_test_seeds": n,
        "reference_linear_composite_mean_auc": 0.6200,
        "reference_linear_composite_ci_95": [0.5455, 0.6945],
    }
    with open(FINAL_RESULT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[trained-attacker] Wrote {FINAL_RESULT_PATH}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    c1 = sub.add_parser("search", help="Leave-One-Seed-Out CV on DEV_SEEDS, locks one model")
    c1.set_defaults(func=cmd_search)

    c2 = sub.add_parser("evaluate", help="score the locked model on TEST_SEEDS -- run once")
    c2.set_defaults(func=cmd_evaluate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
