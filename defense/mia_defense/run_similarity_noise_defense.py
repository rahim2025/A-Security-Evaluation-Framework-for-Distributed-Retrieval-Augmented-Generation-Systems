"""
defense/mia_defense/run_similarity_noise_defense.py

Evaluates the similarity-only Laplace noise defense
(drag_llm_service/src/retriever/defense.py, `add_laplace_noise_to_similarity`)
against the live Reliable-dRAG deployment: reports the existing MIA AUC-ROC
metric (attack/Mia_attack/mia_attack.py's `MIAAttack`, unmodified) and normal
task-quality metrics (attack/ddos_sim/nlg_metrics.py's `score_answer` /
`average_metrics`, unmodified), across this project's standard multi-seed
setup, for ONE server-side configuration per run.

Why this is a different shape from run_defense.py / MIADefenseEvaluator
---------------------------------------------------------------------------
`defense/mia_defense/mia_defense.py`'s `MIADefenseEvaluator` evaluates
`sanitize_response()` / `obfuscate_decision()` / `normalize_length()` --
all response-TEXT transforms applied AFTER a single live `/query` call, so
one probe set can be scored under many "worlds" without extra LLM calls
(see that module's docstring: "there's no numeric score field to add
calibrated noise to ... unlike the original DRAG system's confidence-score
noise defense"). The similarity-only Laplace defense is the opposite
shape: it perturbs a similarity score INSIDE the retrieval pipeline,
server-side, before the LLM ever generates a response -- there is no
response to retroactively re-score under a different epsilon, because a
different epsilon can retrieve genuinely different context and therefore
produce a genuinely different generated answer. Each configuration this
script evaluates must therefore be a SEPARATE live run against a
server actually configured with that setting -- see "Operational
requirement" below.

This script reuses two already-existing, unmodified evaluators rather than
building new ones (mirrors run_defense.py's own reuse of `MIAAttack`-style
signals):
  - MIA metric: `attack.Mia_attack.mia_attack.MIAAttack` (unchanged) --
    the exact same live black-box attacker used throughout
    reports/MIA_Security_Analysis_Report.md. It queries `/query` and
    never sees server-side config; whatever similarity-noise setting the
    server is running with affects retrieval, and therefore the responses
    this attacker scores, transparently.
  - Task-quality metric: `attack.ddos_sim.nlg_metrics.score_answer` /
    `average_metrics` (unchanged) -- the same F1/exact-match implementation
    already used by attack/ddos_sim/run_live_evaluation.py, the only
    F1/EM implementation in this repository (see that module's own
    docstring). Applied here to MEMBER documents' real questions (the
    documents a legitimate retrieval-quality question would actually be
    about), scored against PubMedQA's gold `long_answer`.

Operational requirement (READ BEFORE RUNNING)
-------------------------------------------------
This script does NOT and CANNOT change server-side config for you --
`retrieval.defense.similarity_noise_*` lives in
drag_llm_service/configs/config.yaml and is loaded once at server startup
(same as every other retrieval-config value; see server.py). Before each
invocation:
  1. Edit drag_llm_service/configs/config.yaml's
     retrieval.defense.similarity_noise_enabled/_epsilon to the setting
     you intend to evaluate.
  2. Restart the drag-llm-service container so it picks up the change
     (`docker compose restart drag-llm-service` or equivalent -- this
     project's own report already establishes container restart as the
     mechanism for a live config change, see
     reports/MIA_Security_Analysis_Report.md Revision 7's infrastructure
     note).
  3. THEN run this script with --config_label describing what you just
     set (e.g. "no_defense", "eps_0.1", "eps_1.0", "eps_5.0") -- purely a
     label for the output JSON/table, not something this script verifies
     against the live server. Mislabeling is possible if step 1-2 were
     skipped or done wrong; this script prints a loud reminder at startup
     but cannot detect the mistake for you.

This script intentionally does not attempt to override config.yaml via an
API call or environment variable at request time -- the task's own
instruction was to preserve the current ranking/fusion architecture, and
adding a request-level config override would be a new capability of the
server's request contract, not a reuse of the existing one.

Usage
-----
  # after editing config.yaml + restarting the container for THIS setting:
  python defense/mia_defense/run_similarity_noise_defense.py \\
      --config_label no_defense --seeds 0 42 123

  python defense/mia_defense/run_similarity_noise_defense.py \\
      --config_label eps_1.0 --seeds 0 42 123

Then compare multiple saved runs:
  python defense/mia_defense/compare_similarity_noise_results.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.Mia_attack.mia_attack import (  # noqa: E402
    CORPUS_JSONL,
    DEFAULT_MEMBERS,
    DEFAULT_NONMEMBERS,
    LLM_SERVICE_URL,
    MIAAttack,
    _parse_llm_response,
    load_membership_documents,
)
from attack.ddos_sim.nlg_metrics import average_metrics, score_answer  # noqa: E402

import requests

LOG_DIR = os.path.join(_ROOT, "defense_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# This project's documented multi-seed convention
# (.claude/CLAUDE.md: "Multi-seed runs (at least seeds 0, 42, 123)").
DEFAULT_SEEDS = [0, 42, 123]

# Same rationale as attack/Mia_attack/tune_weights.py's _QUERY_TIMEOUT_SECONDS:
# a short timeout treated as an empty response keeps a multi-seed x
# multi-question task-quality collection tractable rather than stalling on
# a rare slow query. Independent of mia_attack.py's own 120s _query_llm,
# which MIAAttack still uses unmodified for the MIA-metric half of this
# script.
_QUALITY_QUERY_TIMEOUT_SECONDS = 30


def _query_llm_for_quality(question: str, url: str, api_key: str = "") -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        r = requests.post(
            f"{url}/query", json={"query": question}, headers=headers,
            timeout=_QUALITY_QUERY_TIMEOUT_SECONDS,
        )
        if r.status_code == 200:
            return _parse_llm_response(r.json())
    except Exception:
        pass
    return ""


def evaluate_task_quality(
    llm_url: str, corpus_jsonl: str, api_key: str, n_questions: int, seed: int,
) -> Dict[str, float]:
    """
    Normal (non-adversarial) retrieval-quality check: real questions about
    documents that ARE in the corpus (member documents -- the only
    documents a legitimate user's question could be correctly answered
    from), scored against PubMedQA's gold `long_answer` with the
    repository's existing F1/exact-match implementation
    (attack/ddos_sim/nlg_metrics.py). This is what the similarity-noise
    defense's cost is measured against: a defense that drives this to
    near-zero has "worked" for privacy in a vacuous, useless way.
    """
    members, _ = load_membership_documents(
        n_members=n_questions, n_nonmembers=0, seed=seed,
        corpus_jsonl=corpus_jsonl, probes_per_doc=1,
    )
    per_question: List[Dict[str, float]] = []
    for i, qa_list in enumerate(members, 1):
        if not qa_list:
            continue
        qa = qa_list[0]
        response = _query_llm_for_quality(qa["question"], llm_url, api_key)
        metrics = score_answer(response, [qa["answer"]], compute_semantic=False)
        per_question.append(metrics)
        if i % 5 == 0 or i == len(members):
            print(f"    [quality] {i}/{len(members)}  f1={metrics['f1']:.3f}  "
                  f"em={metrics['exact_match']:.0f}", flush=True)
    return average_metrics(per_question)


def run_one_seed(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    print(f"\n{'=' * 62}\n  seed={seed}  config_label={args.config_label}\n{'=' * 62}")

    print("\n  --- MIA metric (attack.Mia_attack.mia_attack.MIAAttack, unmodified) ---")
    attacker = MIAAttack(
        llm_service_url=args.llm_url,
        corpus_jsonl=args.corpus_jsonl,
        n_members=args.n_members,
        n_nonmembers=args.n_nonmembers,
        probes_per_doc=args.probes_per_doc,
        threshold_percentile=args.threshold_pct,
        api_key=args.api_key,
        random_seed=seed,
    )
    mia_result = attacker.run()

    print("\n  --- Task-quality metric (attack.ddos_sim.nlg_metrics, unmodified) ---")
    quality_result = evaluate_task_quality(
        args.llm_url, args.corpus_jsonl, args.api_key, args.n_quality_questions, seed,
    )

    return {
        "seed": seed,
        "mia": {
            "auc_roc": mia_result["auc_roc"],
            "privacy_risk": mia_result["privacy_risk"],
            "auc_roc_answer_match": mia_result["auc_roc_answer_match"],
            "n_members_tested": mia_result["n_members_tested"],
            "n_non_members_tested": mia_result["n_non_members_tested"],
        },
        "task_quality": quality_result,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Similarity-only Laplace noise defense evaluation | Reliable-dRAG MIA"
    )
    p.add_argument("--config_label", required=True,
                    help="Label for THIS run's server-side config, e.g. 'no_defense', "
                         "'eps_1.0' -- see module docstring's Operational requirement. "
                         "Not verified against the live server.")
    p.add_argument("--llm_url", default=LLM_SERVICE_URL)
    p.add_argument("--corpus_jsonl", default=CORPUS_JSONL)
    p.add_argument("--n_members", type=int, default=DEFAULT_MEMBERS)
    p.add_argument("--n_nonmembers", type=int, default=DEFAULT_NONMEMBERS)
    p.add_argument("--probes_per_doc", type=int, default=4)
    p.add_argument("--threshold_pct", type=int, default=50)
    p.add_argument("--api_key", default="")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help=f"default: {DEFAULT_SEEDS} (this project's documented multi-seed "
                         "convention, .claude/CLAUDE.md)")
    p.add_argument("--n_quality_questions", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 62)
    print("  Similarity-only Laplace Noise Defense Evaluation | Reliable-dRAG")
    print("=" * 62)
    print(f"  Config label      : {args.config_label}")
    print(f"  LLM service URL   : {args.llm_url}")
    print(f"  Seeds             : {args.seeds}")
    print(f"  Members/seed      : {args.n_members}   Non-members/seed: {args.n_nonmembers}")
    print(f"  Quality questions : {args.n_quality_questions}/seed")
    print()
    print("  REMINDER: this script reads whatever drag-llm-service is CURRENTLY")
    print("  running with. It does not check that config.yaml's")
    print("  retrieval.defense.similarity_noise_* matches --config_label, and it")
    print("  cannot restart the container for you. See this script's module")
    print("  docstring, 'Operational requirement', before trusting this run's label.")
    print()

    per_seed_results = []
    for seed in args.seeds:
        per_seed_results.append(run_one_seed(args, seed))

    aucs = [r["mia"]["auc_roc"] for r in per_seed_results]
    f1s = [r["task_quality"].get("f1", 0.0) for r in per_seed_results]
    ems = [r["task_quality"].get("exact_match", 0.0) for r in per_seed_results]

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def _std(xs):
        if len(xs) < 2:
            return 0.0
        m = _mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    summary = {
        "config_label": args.config_label,
        "n_seeds": len(args.seeds),
        "mia_auc_roc_mean": round(_mean(aucs), 4),
        "mia_auc_roc_std": round(_std(aucs), 4),
        "task_f1_mean": round(_mean(f1s), 4),
        "task_f1_std": round(_std(f1s), 4),
        "task_exact_match_mean": round(_mean(ems), 4),
        "task_exact_match_std": round(_std(ems), 4),
    }

    print("\n" + "=" * 62)
    print(f"  SUMMARY  config_label={args.config_label}  n_seeds={len(args.seeds)}")
    print("=" * 62)
    print(f"  MIA AUC-ROC        : {summary['mia_auc_roc_mean']:.4f} ± {summary['mia_auc_roc_std']:.4f}")
    print(f"  Task F1            : {summary['task_f1_mean']:.4f} ± {summary['task_f1_std']:.4f}")
    print(f"  Task Exact-Match   : {summary['task_exact_match_mean']:.4f} ± {summary['task_exact_match_std']:.4f}")

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log = {
        "timestamp": ts,
        "defense_type": "similarity_only_laplace_noise",
        "config": vars(args),
        "per_seed_results": per_seed_results,
        "summary": summary,
    }
    log_path = os.path.join(LOG_DIR, f"similarity_noise_{args.config_label}_{ts}.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)
    print(f"\n  Log saved -> {log_path}")


if __name__ == "__main__":
    main()
