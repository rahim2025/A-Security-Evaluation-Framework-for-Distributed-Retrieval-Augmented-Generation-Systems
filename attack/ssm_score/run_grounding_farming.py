"""
attack/ssm_score/run_grounding_farming.py
Grounding-Farming SSM-Score attack for Reliable-dRAG (.claude/ssm_grounding_farming_plan.md).

Unlike the key-forgery SSM variant (ssm_score_attack.py / run_attack.py),
this attack never signs or submits a blockchain transaction itself, and
uses no key but the target source's own. It queries the LLM orchestrator
exactly as an ordinary user would (POST /query_analyze, update_scores=True)
so that the *real* orchestrator computes and submits every score update --
the farming source just makes sure its own content keeps tripping the
naive substring grounding check (server.py:856-858) on those real queries.

Eval questions loaded from HuggingFace rajpurkar/squad (train split),
matched against the corpus actually served by sources_20/sources_100 --
same per-source ground-truth pattern already fixed in
attack/ssm_score/run_attack.py (see .claude/ssm_grounding_farming_plan.md §7a).

Safety notes (see .claude/ssm_grounding_farming_plan.md and this run's
confirmation with the user before first execution):
  - The Hardhat node (drag_contract/entrypoint.sh) runs an in-memory chain
    redeployed fresh on every container start -- no volume, no persistence.
    Any on-chain score drift this script produces is fully undoable by
    restarting the hardhat-node container (which also has nothing else to
    lose right now -- confirmed via a live snapshot immediately before this
    module was written: all three sources sat at the pristine 10000/10000
    initial values with zero prior ScoreRecordUpdated events).
  - /poison and /reset only ever touch the target source's own in-memory
    retriever -- never any file on disk, never any other source, never the
    blockchain. Every other attack/defense module in this repo uses the
    identical /poison + /reset lifecycle already; nothing new is introduced
    here.
  - This script writes ONLY to attack_logs/ and reads/writes retriever
    state on the target source. It does not modify any existing file.
"""
import json
import math
import os
import random
import re
import sys
import time
import pathlib
from collections import Counter
from datetime import datetime

# Some SQuAD questions/answers contain non-ASCII characters (accents, macrons,
# etc.); Windows' default console codepage can't encode them and crashes
# mid-print (hit during the seed-0 campaign run, seed0.json was lost as a
# result even though the try/finally reset still ran correctly).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from datasets import load_dataset

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from grounding_farming_attack import GroundingFarmingAttack, DEFAULT_DATA_SOURCES

# ── Configuration ─────────────────────────────────────────────────────────────
BLOCKCHAIN_URL   = os.getenv("BLOCKCHAIN_URL",   "http://localhost:8545")
LLM_SERVICE_URL  = os.getenv("LLM_SERVICE_URL",  "http://localhost:9000")
TARGET_SOURCE    = os.getenv("TARGET_SOURCE",    "sources_100")
COMPARISON_SOURCES = [s for s in ("sources_0", "sources_20") if s != TARGET_SOURCE]

ROUNDS           = int(os.getenv("ROUNDS", "40"))
NUM_TITLES       = int(os.getenv("NUM_TITLES", "10"))
DOCS_PER_TITLE   = int(os.getenv("DOCS_PER_TITLE", "4"))
KEYWORDS_PER_DOC = int(os.getenv("KEYWORDS_PER_DOC", "4"))
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "50"))
RANDOM_SEED      = int(os.getenv("RANDOM_SEED", "42"))

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
LOG_DIR      = os.path.join(PROJECT_ROOT, "attack_logs")
os.makedirs(LOG_DIR, exist_ok=True)


def _normalize_for_match(s):
    return "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in str(s)).split()


def _norm_join(tokens):
    return " ".join([t for t in tokens if t])


def norm(s):
    return _norm_join(_normalize_for_match(s))


# ── Corpus / eval loading (same per-source ground-truth pattern as run_attack.py) ──
def _load_per_source_contexts():
    """
    Per-source context sets (not pooled) so, for each eval question, we can
    tell whether its gold context lives in sources_20, sources_100, or both --
    needed to test whether an accuracy change is concentrated on questions
    whose real answer-bearing content sits in the farmed source (displacement
    hypothesis, plan §8a) versus spread evenly (generic rerank noise).
    """
    per_source = {}
    for name, fname in (("sources_20", "sources_20.jsonl"), ("sources_100", "sources_100.jsonl")):
        path = os.path.join(PROJECT_ROOT, "data", "polluted_token", fname)
        contexts = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                html = rec.get("html", "")
                if html:
                    contexts.add(html)
        per_source[name] = contexts
    return per_source


STOPWORDS = {
    "what", "which", "when", "where", "does", "did", "was", "were", "that",
    "this", "have", "from", "with", "about", "into", "over", "after",
    "before", "during", "their", "there", "been", "being", "also", "some",
    "many", "much", "most", "other", "those", "these", "would", "could",
    "should", "will", "shall", "than", "then", "them", "such", "only",
    "were", "whom", "whose", "known",
}


# Unicode-aware word regex (§8c bug #1): the previous [A-Za-z]+ pattern
# stripped accents before comparison, so "Beyoncé" (title) never matched
# "beyonc"/"beyonce" (regex-mangled keyword) and mangled title fragments
# leaked into that title's own "distinctive" keyword list. [^\W\d_]+ keeps
# accented letters intact so exclusion actually excludes.
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _keyword_tokens(text, exclude=()):
    words = WORD_RE.findall(text.lower())
    exclude_lower = {w.lower() for w in exclude}
    return [w for w in words if len(w) > 3 and w not in STOPWORDS and w not in exclude_lower]


def load_squad_eval(n=EVAL_SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    Returns (sampled_eval_rows, short_answer_fragments, title_counts, title_keywords).

    title_keywords is harvested from the QUESTIONS of every matched row in
    the full train split (not just the n sampled for eval) -- ranked
    frequent content words per article title, used to build sub-topic
    filler clusters (plan §8b) instead of one generic per-article blurb.
    This does not reuse any exact question text as filler content, only
    aggregate word-frequency shape, consistent with .claude/CLAUDE.md's
    "no exact stored text as probes" rule (this is the inverse direction --
    content generation, not probing -- but the same spirit applies).
    """
    print("[HF] Loading rajpurkar/squad train split ...")
    ds = load_dataset("rajpurkar/squad", split="train")
    per_source_contexts = _load_per_source_contexts()
    corpus_contexts = per_source_contexts["sources_20"] | per_source_contexts["sources_100"]

    seen, rows, all_answers = set(), [], []
    title_counts = {}
    title_keyword_counters = {}
    for item in ds:
        ans_list = item["answers"]["text"]
        if ans_list:
            all_answers.extend(a for a in ans_list if a and len(a) < 40)
        if item["context"] not in corpus_contexts:
            continue
        title = item.get("title", "").replace("_", " ").strip()
        if title:
            title_counts[title] = title_counts.get(title, 0) + 1
            counter = title_keyword_counters.setdefault(title, Counter())
            counter.update(_keyword_tokens(item["question"], exclude=title.split()))
        q = item["question"].strip()
        if q in seen or not ans_list:
            continue
        seen.add(q)
        in_20 = item["context"] in per_source_contexts["sources_20"]
        in_100 = item["context"] in per_source_contexts["sources_100"]
        gold_in = ("sources_20" if in_20 else "") + ("+" if in_20 and in_100 else "") + ("sources_100" if in_100 else "")
        rows.append({"question": q, "answer": ans_list[0], "gold_context_in": gold_in})

    if not rows:
        raise RuntimeError(
            "No SQuAD questions matched sources_20/100 -- "
            "check data/polluted_token/sources_{20,100}.jsonl."
        )

    # TF-IDF-style ranking, not raw frequency (§8c bug #2): raw top-N-by-count
    # surfaced words common to nearly every title's questions ("year", "first",
    # "name", "song") rather than words distinctive to THIS title. Down-weight
    # a word by how many *different* titles' question sets it also appears in.
    num_titles_total = len(title_keyword_counters)
    title_doc_freq = Counter()
    for counter in title_keyword_counters.values():
        title_doc_freq.update(counter.keys())

    title_keywords = {}
    for title, counter in title_keyword_counters.items():
        scored = {
            w: tf * math.log((1 + num_titles_total) / (1 + title_doc_freq[w]))
            for w, tf in counter.items()
        }
        title_keywords[title] = [w for w, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:40]]

    random.seed(seed)
    sampled = random.sample(rows, min(n, len(rows)))
    print(f"[HF] Sampled {len(sampled)} matched questions; "
          f"{len(title_counts)} distinct article titles present in sources_20/100; "
          f"keyword pools harvested from the full matched question set (not just the sample)")
    return sampled, all_answers, title_counts, title_keywords


# ── Blockchain score reads (read-only; this attack never signs a transaction) ──
def get_current_scores(source_ids):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "drag_python_client"))
    from drag_python_client import DragScoresClient
    client = DragScoresClient(project_root=PROJECT_ROOT, provider_url=BLOCKCHAIN_URL)
    ids, rel, use = client.get_scores_batch(source_ids)
    return {sid: {"reliability": rel[i], "usefulness": use[i]} for i, sid in enumerate(ids)}


# ── Query the orchestrator exactly as a normal user would ────────────────────
def query_analyze(question, gold_answer, update_scores):
    payload = {"query": question, "ground_truth": [gold_answer], "update_scores": update_scores}
    r = requests.post(f"{LLM_SERVICE_URL}/query_analyze", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def is_correct(predicted, gold):
    pred = predicted.lower()
    g = gold.lower().strip()
    if g in pred:
        return True
    words = [w for w in g.split() if len(w) > 3]
    return bool(words) and all(w in pred for w in words)


def measure_accuracy(eval_data):
    correct = 0
    details = []
    for item in eval_data:
        resp = query_analyze(item["question"], item["answer"], update_scores=False)
        predicted = resp.get("response", "")
        hit = is_correct(predicted, item["answer"])
        if hit:
            correct += 1
        details.append({
            "question": item["question"],
            "gold": item["answer"],
            "gold_context_in": item.get("gold_context_in", ""),
            "predicted": predicted,
            "correct": hit,
            "sampled_sources": resp.get("sampled_sources", []),
        })
    n = len(eval_data)
    return {
        "accuracy": round(correct / n, 4) if n else 0.0,
        "correct": correct,
        "total": n,
        "details": details,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(LOG_DIR, f"attack_{timestamp}_grounding_farming_seed{RANDOM_SEED}.json")

    all_source_ids = [s["name"] for s in DEFAULT_DATA_SOURCES]
    target_cfg = next(s for s in DEFAULT_DATA_SOURCES if s["name"] == TARGET_SOURCE)

    print("=" * 70)
    print("  Reliable-dRAG  -  Grounding-Farming SSM-Score Attack")
    print("=" * 70)
    print(f"  target_source      : {TARGET_SOURCE}")
    print(f"  comparison_sources : {COMPARISON_SOURCES}  (not the dedicated equal-volume")
    print(f"                       honest control from plan §8 risk #4 -- that is a")
    print(f"                       separate follow-up; these are the other sources")
    print(f"                       already running, untouched by this attack)")
    print(f"  rounds             : {ROUNDS}")
    print(f"  num_titles         : {NUM_TITLES}   docs_per_title: {DOCS_PER_TITLE}   "
          f"keywords_per_doc: {KEYWORDS_PER_DOC}")
    print(f"  eval_sample        : {EVAL_SAMPLE_SIZE}   seed: {RANDOM_SEED}")
    print()

    # 1. Load eval data + corpus title distribution + per-title keyword pools
    eval_data, short_answers, title_counts, title_keywords = load_squad_eval()

    # 2. Snapshot on-chain scores BEFORE anything (all sources)
    scores_before = get_current_scores(all_source_ids)
    print(f"[*] Scores BEFORE: {scores_before}")

    # 3. Baseline accuracy on the eval set (no scores touched: update_scores=False)
    print("\n[*] Measuring BASELINE accuracy (update_scores=False, no chain writes) ...")
    baseline_acc = measure_accuracy(eval_data)
    print(f"    Baseline accuracy: {baseline_acc['accuracy']*100:.1f}% "
          f"({baseline_acc['correct']}/{baseline_acc['total']})")

    # 4. Build + inject sub-topic-clustered filler into the target source (§8b)
    attacker = GroundingFarmingAttack(
        target_source=target_cfg,
        data_sources=DEFAULT_DATA_SOURCES,
        num_titles=NUM_TITLES,
        docs_per_title=DOCS_PER_TITLE,
        keywords_per_doc=KEYWORDS_PER_DOC,
        llm_service_url=LLM_SERVICE_URL,
        blockchain_url=BLOCKCHAIN_URL,
    )
    filler_docs = attacker.build_filler_documents(title_counts, short_answers, title_keywords)
    print(f"\n[*] Built {len(filler_docs)} sub-topic filler docs across "
          f"{len(set(d['meta']['domain_title'] for d in filler_docs))} titles")

    inject_result = attacker.inject()
    print(f"[*] Injection result: {inject_result}")

    # Steps 5-8 run inside try/finally so a crash mid-run (e.g. the Unicode
    # console-encoding crash hit during §8c's content-design testing) can
    # never leave the target source's retriever poisoned -- reset() always
    # runs on the way out, success or failure (plan §8c operational note).
    try:
        # 5. Run ROUNDS of real queries through /query_analyze with update_scores=True.
        #    This is the actual attack: every transaction is submitted by the real
        #    orchestrator, using its own key, for a real query -- the farming
        #    source just keeps tripping the grounding check on its own filler.
        print(f"\n[*] Running {ROUNDS} rounds of real queries (update_scores=True) ...")
        round_log = []
        random.seed(RANDOM_SEED)
        for rnd in range(1, ROUNDS + 1):
            item = random.choice(eval_data)
            try:
                resp = query_analyze(item["question"], item["answer"], update_scores=True)
            except Exception as e:
                print(f"    Round {rnd}/{ROUNDS}: ERROR {e}")
                round_log.append({"round": rnd, "question": item["question"], "error": str(e)})
                continue

            sampled_sources = resp.get("sampled_sources", [])
            importance_score = resp.get("importance_score", [])
            farm_in_pool = TARGET_SOURCE in sampled_sources
            farm_selected = any(entry[0] == TARGET_SOURCE for entry in importance_score)
            current_scores = get_current_scores(all_source_ids)

            round_log.append({
                "round": rnd,
                "question": item["question"],
                "gold": item["answer"],
                "response": resp.get("response", ""),
                "correctness": resp.get("correctness"),
                "sampled_sources": sampled_sources,
                "farm_source_sampled": farm_in_pool,
                "farm_source_in_importance_score": farm_selected,
                "scores_after_round": current_scores,
            })
            print(f"    Round {rnd:3d}/{ROUNDS}: sampled={sampled_sources} "
                  f"farm_selected={farm_selected} "
                  f"R[{TARGET_SOURCE}]={current_scores.get(TARGET_SOURCE, {}).get('reliability')} "
                  f"U[{TARGET_SOURCE}]={current_scores.get(TARGET_SOURCE, {}).get('usefulness')}")
            time.sleep(0.2)

        # 6. Snapshot on-chain scores AFTER
        scores_after = get_current_scores(all_source_ids)
        print(f"\n[*] Scores AFTER: {scores_after}")

        # 7. Post-attack accuracy on the SAME eval set (secondary/sanity metric --
        #    plan §6 expects this to stay roughly flat, unlike Data Poisoning)
        print("\n[*] Measuring POST-ATTACK accuracy (update_scores=False, no chain writes) ...")
        post_acc = measure_accuracy(eval_data)
        print(f"    Post-attack accuracy: {post_acc['accuracy']*100:.1f}% "
              f"({post_acc['correct']}/{post_acc['total']})")

        # 7a. Flip diagnostic (plan §8a): is any accuracy change concentrated on
        #     questions whose gold context actually lives in the farmed source
        #     (displacement hypothesis) or spread evenly (generic rerank noise)?
        flips = []
        for b, p in zip(baseline_acc["details"], post_acc["details"]):
            if b["correct"] != p["correct"]:
                flips.append({
                    "question": b["question"],
                    "gold": b["gold"],
                    "gold_context_in": b["gold_context_in"],
                    "direction": "regressed" if b["correct"] and not p["correct"] else "improved",
                    "baseline_predicted": b["predicted"],
                    "post_predicted": p["predicted"],
                    "baseline_sampled_sources": b["sampled_sources"],
                    "post_sampled_sources": p["sampled_sources"],
                })
        print(f"\n[*] Flip diagnostic: {len(flips)} question(s) changed correctness "
              f"out of {len(eval_data)}")
        for fl in flips:
            print(f"    [{fl['direction']:>9s}] gold_in={fl['gold_context_in']:<18s} "
                  f"Q: {fl['question'][:50]!r}")
            print(f"               baseline_pred={fl['baseline_predicted']!r}  "
                  f"post_pred={fl['post_predicted']!r}")
        regressed_on_target = sum(
            1 for fl in flips if fl["direction"] == "regressed" and TARGET_SOURCE in fl["gold_context_in"]
        )
        regressed_total = sum(1 for fl in flips if fl["direction"] == "regressed")
        print(f"    Regressions whose gold context lives in {TARGET_SOURCE}: "
              f"{regressed_on_target}/{regressed_total}")

        # 8. Metrics (plan §6)
        def delta(sid, key):
            return scores_after.get(sid, {}).get(key, 0) - scores_before.get(sid, {}).get(key, 0)

        score_vs_merit_gap = {}
        for cmp_sid in COMPARISON_SOURCES:
            score_vs_merit_gap[cmp_sid] = {
                "reliability_gap": delta(TARGET_SOURCE, "reliability") - delta(cmp_sid, "reliability"),
                "usefulness_gap": delta(TARGET_SOURCE, "usefulness") - delta(cmp_sid, "usefulness"),
            }

        valid_rounds = [r for r in round_log if "error" not in r]
        misdirection_rate = (
            sum(1 for r in valid_rounds if r["farm_source_sampled"]) / len(valid_rounds)
            if valid_rounds else 0.0
        )
        false_grounding_proxy_rate = (
            sum(1 for r in valid_rounds if r["farm_source_in_importance_score"]) / len(valid_rounds)
            if valid_rounds else 0.0
        )

        print("\n" + "-" * 70)
        print(f"Reliability/usefulness delta [{TARGET_SOURCE}] : "
              f"R={delta(TARGET_SOURCE, 'reliability'):+d}  U={delta(TARGET_SOURCE, 'usefulness'):+d}")
        for cmp_sid in COMPARISON_SOURCES:
            print(f"Reliability/usefulness delta [{cmp_sid}] : "
                  f"R={delta(cmp_sid, 'reliability'):+d}  U={delta(cmp_sid, 'usefulness'):+d}")
        print(f"Score-vs-merit gap (target - comparison)   : {score_vs_merit_gap}")
        print(f"Misdirection rate (farm source sampled)    : {misdirection_rate*100:.1f}%")
        print(f"Farm source in final selection (proxy for  ")
        print(f"  false-grounding opportunity)              : {false_grounding_proxy_rate*100:.1f}%")
        print(f"Accuracy: baseline={baseline_acc['accuracy']*100:.1f}%  "
              f"post={post_acc['accuracy']*100:.1f}%  "
              f"(expected roughly flat, unlike Data Poisoning)")
    finally:
        # 9. Reset target source's retriever content (NOT on-chain scores -- see
        #    module docstring and .claude/ssm_grounding_farming_plan.md §3a for
        #    why those are a separate, deliberately-produced result, not a
        #    side effect to clean up). Always runs, even if the try block above
        #    raised, so the retriever never stays poisoned past a crash.
        reset_result = attacker.reset()
        print(f"\n[*] Reset {TARGET_SOURCE} retriever: {reset_result}")

    report = {
        "attack": "grounding_farming",
        "timestamp": timestamp,
        "config": {
            "target_source": TARGET_SOURCE,
            "comparison_sources": COMPARISON_SOURCES,
            "rounds": ROUNDS,
            "num_titles": NUM_TITLES,
            "docs_per_title": DOCS_PER_TITLE,
            "keywords_per_doc": KEYWORDS_PER_DOC,
            "eval_sample_size": len(eval_data),
            "random_seed": RANDOM_SEED,
            "eval_dataset": "rajpurkar/squad",
        },
        "filler_docs": [
            {
                "id": d["id"],
                "domain_title": d["meta"]["domain_title"],
                "subtopic_keywords": d["meta"].get("subtopic_keywords", []),
                "text": d["text"],
            }
            for d in filler_docs
        ],
        "scores_before": scores_before,
        "scores_after": scores_after,
        "score_vs_merit_gap": score_vs_merit_gap,
        "misdirection_rate": misdirection_rate,
        "false_grounding_proxy_rate": false_grounding_proxy_rate,
        "baseline_accuracy": baseline_acc,
        "post_attack_accuracy": post_acc,
        "flips": flips,
        "regressed_on_target_source": f"{regressed_on_target}/{regressed_total}",
        "round_log": round_log,
    }
    with open(log_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Log saved -> {log_file}")
    return report


if __name__ == "__main__":
    main()
