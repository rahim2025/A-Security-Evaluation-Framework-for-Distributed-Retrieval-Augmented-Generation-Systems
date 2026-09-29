"""
attack/Mia_attack/contrastive_trained_attacker.py

Revision 9, Phase 2b -- trains + evaluates a classifier on the richer
contrastive/wording feature set `contrastive_probes.py` collects, the same
way `trained_attacker.py` (Phase 1) does for the original 4-signal data.
Kept as a separate script rather than folded into `trained_attacker.py`
because the feature set, data files, and (per `contrastive_probes.py`'s
`--seeds` override) the seed list are all different in kind, not just in
value -- see that module's docstring for why Phase 2's data collection is
a genuinely different, more expensive operation than Phase 1's.

Pipeline (run in order, on the machine with the live services)
------------------------------------------------------------------
  1. python contrastive_probes.py collect --split dev  [--seeds ...]
  2. python contrastive_probes.py collect --split test [--seeds ...]
  3. python contrastive_trained_attacker.py search      # LOSO-CV on dev only
  4. python contrastive_trained_attacker.py evaluate     # locked model, test once

Seed handling differs from trained_attacker.py by necessity
-----------------------------------------------------------------
Phase 1 asserts the collected data's seeds are a subset of the fixed
`DEV_SEEDS`/`TEST_SEEDS` lists (attack/Mia_attack/tune_weights.py) --
appropriate there, because that data was already collected once
(Revision 7) at the full 13+5-seed scale with no override mechanism.
Phase 2's collector supports an explicit `--seeds` override (e.g. a
cheaper 3-seed pilot before committing to a full run -- see
reports/updated_reports_safin/README.md's recommended next step), so this
script cannot assume any fixed seed list. What it DOES still enforce,
exactly like Phase 1: the seeds actually present in the dev file and the
seeds actually present in the test file must be disjoint -- the one
invariant that must never be violated regardless of how many seeds were
collected or which ones.
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

from attack.Mia_attack.contrastive_probes import (  # noqa: E402
    CONTRASTIVE_DEV_PATH,
    CONTRASTIVE_TEST_PATH,
)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This script needs a working scikit-learn build. If the project's "
        "system Python has an architecture-mismatched scikit-learn install "
        "(as it did in an earlier session on macOS/arm64), use a working "
        "environment instead, e.g.: "
        "/opt/miniconda3/bin/python3 contrastive_trained_attacker.py ...\n"
        f"Original import error: {exc}"
    ) from exc

_DATA_DIR = os.path.join(_ROOT, "attack_logs")
LOCKED_MODEL_PATH = os.path.join(_DATA_DIR, "contrastive_trained_attacker_locked.pkl")
LOCKED_META_PATH = os.path.join(_DATA_DIR, "contrastive_trained_attacker_locked_meta.json")
FINAL_RESULT_PATH = os.path.join(_DATA_DIR, "contrastive_trained_attacker_final_result.json")

# Must match the row shape contrastive_probes.py's _collect_contrastive_signals()
# writes -- "n_probes", "seed", "label" are metadata, not model features.
FEATURE_NAMES = [
    "norm_sim_mean", "norm_sim_max", "norm_sim_std",
    "certainty_mean", "certainty_max",
    "length_ratio_mean",
    "decision_match_rate", "decision_match_std",
    "hedge_fraction_mean", "char_length_mean", "word_count_mean", "refusal_rate",
]

_T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _load_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python contrastive_probes.py collect "
            f"--split {'dev' if path == CONTRASTIVE_DEV_PATH else 'test'}` first "
            "(needs the live LLM service up)."
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _rows_to_xy(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows], dtype=int)
    groups = np.array([r["seed"] for r in rows], dtype=int)
    return X, y, groups


def _candidate_models() -> Dict[str, Any]:
    """Same two-family, small/regularized candidate set as trained_attacker.py
    (Phase 1) -- see that module's docstring for the reasoning. Repeated here
    rather than imported, since the two scripts' feature dimensionality
    differs (4 vs. 12) and picking hyperparameters is specific to each."""
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
    rows = _load_rows(CONTRASTIVE_DEV_PATH)
    seeds_present = sorted(set(r["seed"] for r in rows))
    n_seeds = len(seeds_present)
    print(f"[contrastive-attacker] Loaded {len(rows)} dev rows across seeds: {seeds_present}")
    if n_seeds < 2:
        raise ValueError(
            f"Only {n_seeds} distinct seed(s) in the dev data -- Leave-One-Seed-Out "
            "CV needs at least 2 seeds to hold one out and train on the rest. "
            "Collect more seeds (contrastive_probes.py collect --split dev --seeds ...)."
        )

    X, y, groups = _rows_to_xy(rows)
    logo = LeaveOneGroupOut()

    print(f"\n[contrastive-attacker] Leave-One-Seed-Out CV over {len(_candidate_models())} "
          f"candidate models, {n_seeds} folds -- DEV data only, test data never touched:\n")

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

            model = type(model_template)(**model_template.get_params())
            if len(np.unique(y_train)) < 2:
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
        std_auc = float(np.std([s for _, s in fold_scores], ddof=1)) if len(fold_scores) > 1 else 0.0
        results.append((mean_auc, std_auc, name, fold_scores))
        per_seed_str = ", ".join(f"{seed}:{auc:.3f}" for seed, auc in sorted(fold_scores))
        print(f"  {name:<20s}  LOSO-CV mean AUC = {mean_auc:.4f}  (std {std_auc:.4f})")
        print(f"    per-seed: {per_seed_str}")

    results.sort(key=lambda x: -x[0])
    winner_mean, winner_std, winner_name, winner_folds = results[0]
    print(f"\n[contrastive-attacker] WINNER (dev-only LOSO-CV): {winner_name}  "
          f"mean AUC = {winner_mean:.4f}  (std {winner_std:.4f})")
    print("[contrastive-attacker] Compare to Phase 1's dev-only LOSO-CV winner "
          "(attack_logs/trained_attacker_locked_meta.json, if present) and to "
          "Revision 7's linear-composite dev score: 0.5723.")

    final_scaler = StandardScaler().fit(X)
    X_all_s = final_scaler.transform(X)
    winner_template = _candidate_models()[winner_name]
    final_model = type(winner_template)(**winner_template.get_params())
    final_model.fit(X_all_s, y)

    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(LOCKED_MODEL_PATH, "wb") as fh:
        pickle.dump({"scaler": final_scaler, "model": final_model,
                     "feature_names": FEATURE_NAMES, "model_name": winner_name,
                     "dev_seeds": seeds_present}, fh)

    meta = {
        "model_name": winner_name,
        "feature_names": FEATURE_NAMES,
        "dev_seeds": seeds_present,
        "loso_cv_mean_auc": round(winner_mean, 4),
        "loso_cv_std_auc": round(winner_std, 4),
        "loso_cv_per_seed": {str(s): round(a, 4) for s, a in winner_folds},
        "all_candidates": [
            {"model_name": n, "loso_cv_mean_auc": round(m, 4), "loso_cv_std_auc": round(sd, 4)}
            for m, sd, n, _ in results
        ],
    }
    if hasattr(final_model, "coef_"):
        meta["final_model_coefficients"] = {
            f: round(float(c), 4) for f, c in zip(FEATURE_NAMES, final_model.coef_[0])
        }
        meta["final_model_intercept"] = round(float(final_model.intercept_[0]), 4)
    if hasattr(final_model, "feature_importances_"):
        meta["final_model_feature_importances"] = {
            f: round(float(v), 4) for f, v in zip(FEATURE_NAMES, final_model.feature_importances_)
        }
    with open(LOCKED_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n[contrastive-attacker] Locked model written -> {LOCKED_MODEL_PATH}")
    print(f"[contrastive-attacker] Metadata written -> {LOCKED_META_PATH}")
    print("[contrastive-attacker] Model is now FROZEN. Next step: evaluate -- run exactly once.")


def cmd_evaluate(args: argparse.Namespace) -> None:
    with open(LOCKED_MODEL_PATH, "rb") as fh:
        locked = pickle.load(fh)
    scaler, model, model_name = locked["scaler"], locked["model"], locked["model_name"]
    dev_seeds = set(locked.get("dev_seeds", []))
    print(f"[contrastive-attacker] Evaluating LOCKED model (selected on dev LOSO-CV only): "
          f"{model_name}")

    rows = _load_rows(CONTRASTIVE_TEST_PATH)
    test_seeds_present = sorted(set(r["seed"] for r in rows))
    print(f"[contrastive-attacker] Loaded {len(rows)} test rows across seeds: {test_seeds_present}")

    overlap = dev_seeds & set(test_seeds_present)
    assert not overlap, (
        f"Test data contains seed(s) {sorted(overlap)} that were ALSO in the dev "
        "data used to select/train this model -- refusing to evaluate, this would "
        "not be a fair held-out test. Collect genuinely disjoint test seeds."
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
    t_crit = _T_TABLE.get(n - 1, 1.96) if n > 1 else float("nan")
    ci = (round(mean_auc - t_crit * se, 4), round(mean_auc + t_crit * se, 4)) if n > 1 else (None, None)

    print(f"\n[contrastive-attacker] TRUE HELD-OUT RESULT (never seen during search) -- "
          f"per seed: {per_seed}")
    if n > 1:
        print(f"[contrastive-attacker] Mean AUC on fresh test seeds: {mean_auc:.4f}   95% CI: {ci}")
    else:
        print(f"[contrastive-attacker] AUC on the single test seed: {mean_auc:.4f}   "
              "(n=1 -- no CI computable; collect more test seeds before trusting this number, "
              "same caution this project applies everywhere else, e.g. "
              "reports/MIA_Security_Analysis_Report.md Revision 5's n=2 caveat)")
    print("[contrastive-attacker] Compare to Phase 1's trained-classifier result "
          "(attack_logs/trained_attacker_final_result.json, if present) and to the "
          "linear composite: mean AUC 0.6200, 95% CI [0.5455, 0.6945].")

    out = {
        "model_name": model_name,
        "per_seed_auc": per_seed,
        "mean_auc": round(mean_auc, 4),
        "std_auc": round(std_auc, 4),
        "ci_95": list(ci) if ci[0] is not None else None,
        "n_test_seeds": n,
        "dev_seeds_used_for_training": sorted(dev_seeds),
        "reference_phase1_trained_classifier_mean_auc": 0.6658,
        "reference_linear_composite_mean_auc": 0.6200,
        "reference_linear_composite_ci_95": [0.5455, 0.6945],
    }
    with open(FINAL_RESULT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[contrastive-attacker] Wrote {FINAL_RESULT_PATH}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    c1 = sub.add_parser("search", help="Leave-One-Seed-Out CV on the collected dev data, locks one model")
    c1.set_defaults(func=cmd_search)

    c2 = sub.add_parser("evaluate", help="score the locked model on the collected test data -- run once")
    c2.set_defaults(func=cmd_evaluate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
