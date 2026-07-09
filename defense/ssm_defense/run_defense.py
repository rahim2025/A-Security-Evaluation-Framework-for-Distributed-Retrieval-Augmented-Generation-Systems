"""
defense/ssm_defense/run_defense.py
Evaluates the SSM-Score defense against Reliable-dRAG.

Requires the patched DragScores contract (see drag_contract/contracts/
drag_scores.sol) to already be deployed -- see the redeploy commands in
defense/ssm_defense/README.md before running this.

Flow:
  1. Baseline accuracy on the SQuAD eval sample.
  2. Launch the same SSM-Score attack used in attack/ssm_score, and record
     which rounds the contract accepted vs. rejected.
  3. Post-attempt accuracy -- should be ~= baseline if the defense holds.
  4. Replay on-chain events and report any residual anomalies.
"""

import json
import os
import pathlib
import sys
import time
from datetime import datetime

# ── add project root to path so sibling packages resolve ──────────────────
PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from attack.ssm_score.run_attack import load_squad_eval, query_rag, measure_accuracy  # noqa: E402
from attack.ssm_score.ssm_score_attack import SSMScoreAttack  # noqa: E402
from ssm_score_defense import SSMScoreDefense, DEFAULT_DATA_SOURCES  # noqa: E402

BLOCKCHAIN_URL  = os.getenv("BLOCKCHAIN_URL",  "http://localhost:8545")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:9000")
TARGET_SOURCE   = os.getenv("TARGET_SOURCE",   "sources_100")
AMPLIFY         = int(os.getenv("AMPLIFY",     "999999"))
ROUNDS          = int(os.getenv("ROUNDS",      "5"))

EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "50"))
RANDOM_SEED      = int(os.getenv("RANDOM_SEED",      "42"))

LOG_DIR = os.path.join(PROJECT_ROOT, "defense_logs")
os.makedirs(LOG_DIR, exist_ok=True)


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file  = os.path.join(LOG_DIR, f"defense_{timestamp}_ssm_score.json")

    print(f"\n{'='*60}")
    print("  Reliable-dRAG  -  SSM-Score Defense Evaluation")
    print(f"{'='*60}")
    print(f"  target_source : {TARGET_SOURCE}")
    print(f"  blockchain    : {BLOCKCHAIN_URL}")
    print(f"  llm_service   : {LLM_SERVICE_URL}")
    print(f"  amplify       : {AMPLIFY}   rounds: {ROUNDS}")
    print()

    eval_data = load_squad_eval(n=EVAL_SAMPLE_SIZE, seed=RANDOM_SEED)

    print("\n[*] Measuring BASELINE accuracy ...")
    baseline = measure_accuracy(eval_data)
    print(f"    Baseline accuracy: {baseline['accuracy']*100:.1f}%  "
          f"({baseline['correct']}/{baseline['total']})")

    print("\n[*] Attempting SSM-Score attack against the defended contract ...")
    attacker = SSMScoreAttack(
        target_source=TARGET_SOURCE,
        blockchain_url=BLOCKCHAIN_URL,
        project_root=PROJECT_ROOT,
        amplify=AMPLIFY,
        rounds=ROUNDS,
        llm_service_url=LLM_SERVICE_URL,
    )
    scores_before = attacker.get_current_scores()
    inflate_result = attacker.inflate_scores(TARGET_SOURCE)
    scores_after = attacker.get_current_scores()

    rounds_accepted = len(inflate_result["tx_hashes"])
    rounds_blocked  = ROUNDS - rounds_accepted
    print(f"\n    Rounds accepted by contract: {rounds_accepted}/{ROUNDS}")
    print(f"    Rounds blocked by defense:   {rounds_blocked}/{ROUNDS}")

    time.sleep(3)

    print("\n[*] Measuring POST-ATTEMPT accuracy ...")
    post = measure_accuracy(eval_data)
    print(f"    Post-attempt accuracy: {post['accuracy']*100:.1f}%  "
          f"({post['correct']}/{post['total']})")

    drop = baseline["accuracy"] - post["accuracy"]
    print(f"\n    Accuracy drop: {drop*100:.1f} percentage points "
          f"(0.0 means the defense fully neutralised the attack)")

    print("\n[*] Replaying on-chain events for residual anomalies ...")
    defense = SSMScoreDefense(blockchain_url=BLOCKCHAIN_URL, project_root=PROJECT_ROOT)
    report = defense.scan(source_ids=list(DEFAULT_DATA_SOURCES.keys()))
    print(f"    Scanned {report.events_scanned} events across {len(report.sources_scanned)} sources")
    print(f"    Anomalies found: {len(report.anomalies)}")
    for a in report.anomalies:
        print(f"      [{a.kind}] {a.source_id}: {a.detail} (tx={a.transaction_hash})")

    result = {
        "defense":       "ssm_score",
        "timestamp":     timestamp,
        "config": {
            "target_source":    TARGET_SOURCE,
            "blockchain_url":   BLOCKCHAIN_URL,
            "llm_service_url":  LLM_SERVICE_URL,
            "amplify":          AMPLIFY,
            "rounds":           ROUNDS,
            "eval_sample_size": len(eval_data),
            "random_seed":      RANDOM_SEED,
        },
        "baseline":         baseline,
        "post_attempt":     post,
        "accuracy_drop":    round(drop, 4),
        "rounds_accepted":  rounds_accepted,
        "rounds_blocked":   rounds_blocked,
        "scores_before":    scores_before,
        "scores_after":     scores_after,
        "attack_tx_result": inflate_result,
        "anomaly_scan":     report.to_dict(),
    }
    with open(log_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[+] Log saved -> {log_file}")
    return result


if __name__ == "__main__":
    main()
