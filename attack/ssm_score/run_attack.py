"""
attack/ssm_score/run_attack.py
SSM-Score key-forgery attack for Reliable-dRAG -- control-plane /
orchestrator-key-compromise variant (see ssm_score_attack.py's module
docstring and .claude/ssm_grounding_farming_plan.md §7). This is the
secondary SSM finding; Grounding-Farming
(attack/ssm_score/run_grounding_farming.py) is the flagship, no-privileged-
access instantiation.

AMPLIFY/INTER_ROUND_DELAY default to values that respect the deployed
transaction-layer defense's MAX_DELTA_PER_UPDATE (5,000) and
MIN_UPDATE_INTERVAL (2s) -- a defense-aware attacker throttles itself just
under both caps rather than tripping ScoreDeltaTooLarge/UpdateTooFrequent
on the first round. defense/ssm_defense/run_defense.py deliberately keeps
the old reckless defaults (999999 / 0.5s) to demonstrate the defense
blocking that naive burst -- this script does not touch that scenario.

Eval questions loaded from HuggingFace rajpurkar/squad (train split).
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
# Defense-aware defaults: stay just under drag_scores.sol's MAX_DELTA_PER_UPDATE
# (5,000) and MIN_UPDATE_INTERVAL (2s) so every round is accepted instead of
# reverting on round 1 (see ssm_score_attack.py's module docstring).
AMPLIFY            = int(os.getenv("AMPLIFY",             "4000"))
ROUNDS              = int(os.getenv("ROUNDS",              "5"))
INTER_ROUND_DELAY   = float(os.getenv("INTER_ROUND_DELAY", "2.5"))

EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "50"))
RANDOM_SEED      = int(os.getenv("RANDOM_SEED",      "42"))

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
LOG_DIR      = os.path.join(PROJECT_ROOT, "attack_logs")
os.makedirs(LOG_DIR, exist_ok=True)

HEADERS = {"Content-Type": "application/json"}


# ── HuggingFace SQuAD loader ──────────────────────────────────────────────────
def _load_corpus_contexts():
    """
    Passages actually served by the Docker data sources, restricted to the
    two sources that are still SQuAD-domain.

    sources_0.jsonl was migrated to PubMedQA content by
    data/build_pubmedqa_corpus.py (written for attack/Mia_attack's MIA fix --
    see that script's docstring) and must not be used as ground truth here
    any more. sources_20/100 remain independently token-polluted SQuAD
    variants (reports/Security_Analysis_Report.md: only 207/500 rows are
    byte-identical between them), so this pools both -- a question counts as
    "answerable" if its original context still matches, unpolluted, in
    *either* source. Matching only one would undercount how much of the
    system's combined corpus can actually answer a given question, which
    matters here because SSM-Score's accuracy metric is a system-wide
    measurement, not a single-source one (contrast attack/kb_extraction,
    which targets one source's content specifically and correctly matches
    per-source instead of pooling).

    Same collision, already hit and fixed the same way in
    attack/kb_extraction/run_attack.py and
    attack/selective_forward_sim/run_attack.py.
    """
    contexts = set()
    for fname in ("sources_20.jsonl", "sources_100.jsonl"):
        path = os.path.join(PROJECT_ROOT, "data", "polluted_token", fname)
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                html = rec.get("html", "")
                if html:
                    contexts.add(html)
    return contexts


def load_squad_eval(n=EVAL_SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    Sample QA pairs restricted to contexts actually present, unpolluted, in
    sources_20 or sources_100 (see _load_corpus_contexts -- sources_0 is
    PubMedQA now, not SQuAD, so it's excluded from matching). The data
    sources are seeded from the SQuAD *train* split; sampling from
    *validation* asks about articles never loaded into any source, pinning
    accuracy near 0 regardless of the attack.
    """
    print(f"[HF] Loading rajpurkar/squad train split ...")
    ds = load_dataset("rajpurkar/squad", split="train")
    corpus_contexts = _load_corpus_contexts()
    print(f"[HF] Matching questions against {len(corpus_contexts)} loaded source documents (sources_20/100) ...")

    seen, rows = set(), []
    for item in ds:
        if item["context"] not in corpus_contexts:
            continue
        q = item["question"].strip()
        if q in seen:
            continue
        seen.add(q)
        ans_list = item["answers"]["text"]
        if not ans_list:
            continue
        rows.append({"question": q, "answer": ans_list[0]})

    if not rows:
        raise RuntimeError(
            "No SQuAD questions matched sources_20/100 -- "
            "check data/polluted_token/sources_{20,100}.jsonl."
        )

    random.seed(seed)
    sampled = random.sample(rows, min(n, len(rows)))
    print(f"[HF] Sampled {len(sampled)} questions answerable from the running corpus (seed={seed})")
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
# drag_data_source/app/server.py rate-limits each client IP to
# RATE_LIMIT_DEFAULT (60/min by default) per source, and the orchestrator
# fans every /query out to all three sources -- a tight, unthrottled loop
# over more than ~50-60 questions can trip 429s and silently degrade into
# empty predictions that look like (but are not) an accuracy drop. Space
# queries out to stay safely under that budget (same class of issue as
# reports/SFA_Security_Analysis_Report.md sec 12.9).
QUERY_DELAY = float(os.getenv("QUERY_DELAY", "1.3"))


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
        time.sleep(QUERY_DELAY)
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
    log_file  = os.path.join(LOG_DIR, f"attack_{timestamp}_ssm_score_key_forgery_seed{RANDOM_SEED}.json")

    print(f"\n{'='*60}")
    print("  Reliable-dRAG  -  SSM-Score Key-Forgery Attack")
    print("  (control-plane / orchestrator-key-compromise -- secondary")
    print("   finding; see run_grounding_farming.py for the flagship,")
    print("   no-privileged-access SSM instantiation)")
    print(f"{'='*60}")
    print(f"  target_source     : {TARGET_SOURCE}")
    print(f"  blockchain        : {BLOCKCHAIN_URL}")
    print(f"  llm_service       : {LLM_SERVICE_URL}")
    print(f"  amplify           : {AMPLIFY}   rounds: {ROUNDS}   inter_round_delay: {INTER_ROUND_DELAY}s")
    print(f"  eval_sample       : {EVAL_SAMPLE_SIZE}   seed: {RANDOM_SEED}")
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

    # 4. Run SSM attack (defense-aware: throttled to stay under the
    #    deployed contract's MAX_DELTA_PER_UPDATE/MIN_UPDATE_INTERVAL caps)
    print("\n[*] Launching SSM-Score key-forgery attack ...")
    attacker = SSMScoreAttack(
        target_source     = TARGET_SOURCE,
        blockchain_url    = BLOCKCHAIN_URL,
        project_root      = PROJECT_ROOT,
        amplify           = AMPLIFY,
        rounds            = ROUNDS,
        llm_service_url   = LLM_SERVICE_URL,
        inter_round_delay = INTER_ROUND_DELAY,
    )
    # Step 1 — snapshot scores before attack
    scores_before = attacker.get_current_scores()
    print(f"    Scores BEFORE: {scores_before}")

    # Step 2 — inflate scores of the target source (the actual attack)
    inflate_result = attacker.inflate_scores(TARGET_SOURCE)
    print(f"    inflate_scores result: {inflate_result}")

    rounds_accepted = len(inflate_result["tx_hashes"])
    rounds_blocked  = ROUNDS - rounds_accepted
    print(f"    Rounds accepted by contract: {rounds_accepted}/{ROUNDS}")
    print(f"    Rounds blocked by defense:   {rounds_blocked}/{ROUNDS}")

    # Step 3 — snapshot scores after attack
    scores_after = attacker.get_current_scores()
    print(f"    Scores AFTER:  {scores_after}")

    attack_log = {
        "scores_before":  scores_before,
        "inflate_result": inflate_result,
        "scores_after":   scores_after,
        "rounds_accepted": rounds_accepted,
        "rounds_blocked":  rounds_blocked,
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
        "attack":       "ssm_score_key_forgery",
        "threat_model": "control-plane / orchestrator-key-compromise -- "
                         "NOT a lone malicious data source (see "
                         "ssm_score_attack.py module docstring and "
                         ".claude/ssm_grounding_farming_plan.md §7). "
                         "Secondary finding; Grounding-Farming is flagship.",
        "timestamp":        timestamp,
        "config": {
            "target_source":      TARGET_SOURCE,
            "blockchain_url":     BLOCKCHAIN_URL,
            "llm_service_url":    LLM_SERVICE_URL,
            "amplify":            AMPLIFY,
            "rounds":             ROUNDS,
            "inter_round_delay":  INTER_ROUND_DELAY,
            "eval_sample_size":   len(eval_data),
            "random_seed":        RANDOM_SEED,
            "eval_dataset":       "rajpurkar/squad",
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
