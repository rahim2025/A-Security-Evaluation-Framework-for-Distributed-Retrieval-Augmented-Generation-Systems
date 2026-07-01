"""
attack/ssm_score/run_attack.py
SSM-Score manipulation attack for Reliable-dRAG.
Eval questions loaded from HuggingFace rajpurkar/squad (validation split).
"""

import json
import os
import random
import sys
import time
import pathlib
from datetime import datetime

import requests
from datasets import load_dataset

# ── add attack module to path ──────────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ssm_score_attack import SSMScoreAttack

# ── Configuration ─────────────────────────────────────────────────────────────
BLOCKCHAIN_URL  = os.getenv("BLOCKCHAIN_URL",  "http://localhost:8545")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:9000")
TARGET_SOURCE   = os.getenv("TARGET_SOURCE",   "sources_100")
AMPLIFY         = int(os.getenv("AMPLIFY",     "999999"))
ROUNDS          = int(os.getenv("ROUNDS",      "5"))

EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "50"))
RANDOM_SEED      = int(os.getenv("RANDOM_SEED",      "42"))

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
LOG_DIR      = os.path.join(PROJECT_ROOT, "attack_logs")
os.makedirs(LOG_DIR, exist_ok=True)

HEADERS = {"Content-Type": "application/json"}


# ── HuggingFace SQuAD loader ──────────────────────────────────────────────────
def load_squad_eval(n=EVAL_SAMPLE_SIZE, seed=RANDOM_SEED):
    print(f"[HF] Loading rajpurkar/squad validation split ...")
    ds = load_dataset("rajpurkar/squad", split="validation")
    seen, rows = set(), []
    for item in ds:
        q = item["question"].strip()
        if q in seen:
            continue
        seen.add(q)
        ans_list = item["answers"]["text"]
        if not ans_list:
            continue
        rows.append({"question": q, "answer": ans_list[0]})
    random.seed(seed)
    sampled = random.sample(rows, min(n, len(rows)))
    print(f"[HF] Sampled {len(sampled)} questions (seed={seed})")
    return sampled


# ── Query the RAG pipeline ────────────────────────────────────────────────────
def query_rag(question, top_k=5):
    """
    POST /query -> {"response": "system\n<answer>"}
    Strip the 'system\\n' prefix before returning.
    """
    try:
        resp = requests.post(
            f"{LLM_SERVICE_URL}/query",
            headers=HEADERS,
            json={"query": question, "k": top_k},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        raw  = data.get("response") or data.get("answer") or ""
        # Strip leading role prefix e.g. "system\n", "assistant\n"
        if "\n" in raw:
            raw = raw.split("\n", 1)[-1].strip()
        return {"answer": raw, "raw": data}
    except Exception as exc:
        return {"error": str(exc), "answer": ""}


# ── Answer correctness check ──────────────────────────────────────────────────
def is_correct(predicted, gold):
    pred = predicted.lower()
    g    = gold.lower().strip()
    if g in pred:
        return True
    # All significant words from gold appear in prediction
    words = [w for w in g.split() if len(w) > 3]
    return bool(words) and all(w in pred for w in words)


# ── Measure accuracy over the eval set ───────────────────────────────────────
def measure_accuracy(eval_data):
    correct = 0
    results = []
    for item in eval_data:
        response  = query_rag(item["question"])
        predicted = response.get("answer", "")
        hit       = is_correct(predicted, item["answer"])
        if hit:
            correct += 1
        results.append({
            "question":  item["question"],
            "gold":      item["answer"],
            "predicted": predicted,
            "correct":   hit,
        })
    acc = correct / len(eval_data) if eval_data else 0.0
    return {
        "accuracy": round(acc, 4),
        "correct":  correct,
        "total":    len(eval_data),
        "details":  results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file  = os.path.join(LOG_DIR, f"attack_{timestamp}_ssm_score.json")

    print(f"\n{'='*60}")
    print("  Reliable-dRAG  -  SSM-Score Manipulation Attack")
    print(f"{'='*60}")
    print(f"  target_source : {TARGET_SOURCE}")
    print(f"  blockchain    : {BLOCKCHAIN_URL}")
    print(f"  llm_service   : {LLM_SERVICE_URL}")
    print(f"  amplify       : {AMPLIFY}   rounds: {ROUNDS}")
    print(f"  eval_sample   : {EVAL_SAMPLE_SIZE}   seed: {RANDOM_SEED}")
    print()

    # 1. Load eval data from HuggingFace
    eval_data = load_squad_eval()

    # 2. Quick sanity check — one question before full eval
    print("[*] Sanity check (1 question) ...")
    sample = query_rag(eval_data[0]["question"])
    print(f"    Q: {eval_data[0]['question']}")
    print(f"    A: {sample['answer']}")
    print(f"    Gold: {eval_data[0]['answer']}")

    # 3. Baseline accuracy BEFORE attack
    print("\n[*] Measuring BASELINE accuracy ...")
    baseline = measure_accuracy(eval_data)
    print(f"    Baseline accuracy: {baseline['accuracy']*100:.1f}%  "
          f"({baseline['correct']}/{baseline['total']})")

    # 4. Run SSM attack
    print("\n[*] Launching SSM-Score attack ...")
    attacker = SSMScoreAttack(
        target_source   = TARGET_SOURCE,
        blockchain_url  = BLOCKCHAIN_URL,
        project_root    = PROJECT_ROOT,
        amplify         = AMPLIFY,
        rounds          = ROUNDS,
        llm_service_url = LLM_SERVICE_URL,
    )
    # Step 1 — snapshot scores before attack
    scores_before = attacker.get_current_scores()
    print(f"    Scores BEFORE: {scores_before}")

    # Step 2 — inflate scores of the target source (the actual attack)
    inflate_result = attacker.inflate_scores(TARGET_SOURCE)
    print(f"    inflate_scores result: {inflate_result}")

    # Step 3 — snapshot scores after attack
    scores_after = attacker.get_current_scores()
    print(f"    Scores AFTER:  {scores_after}")

    attack_log = {
        "scores_before":  scores_before,
        "inflate_result": inflate_result,
        "scores_after":   scores_after,
    }

    # Give blockchain time to mine
    time.sleep(3)

    # 5. Post-attack accuracy
    print("\n[*] Measuring POST-ATTACK accuracy ...")
    post = measure_accuracy(eval_data)
    print(f"    Post-attack accuracy:  {post['accuracy']*100:.1f}%  "
          f"({post['correct']}/{post['total']})")

    drop = baseline["accuracy"] - post["accuracy"]
    print(f"\n    Accuracy drop: {drop*100:.1f} percentage points")

    # 6. Save full log
    report = {
        "attack":           "ssm_score",
        "timestamp":        timestamp,
        "config": {
            "target_source":    TARGET_SOURCE,
            "blockchain_url":   BLOCKCHAIN_URL,
            "llm_service_url":  LLM_SERVICE_URL,
            "amplify":          AMPLIFY,
            "rounds":           ROUNDS,
            "eval_sample_size": len(eval_data),
            "random_seed":      RANDOM_SEED,
            "eval_dataset":     "rajpurkar/squad",
        },
        "baseline":      baseline,
        "post_attack":   post,
        "accuracy_drop": round(drop, 4),
        "transactions":  attack_log,
    }
    with open(log_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Log saved -> {log_file}")
    return report


if __name__ == "__main__":
    main()
