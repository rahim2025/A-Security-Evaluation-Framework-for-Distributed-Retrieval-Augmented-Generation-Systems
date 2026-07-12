"""
attack/kb_extraction/run_attack.py
Knowledge-Base Extraction attack for Reliable-dRAG.
Probe questions loaded from HuggingFace rajpurkar/squad (validation split).

Configuration fix (see reports comparing this against the idealized
modules/attacks/kb_extraction.py design doc): TOP_K used to default to 10,
but drag_data_source/app/server.py's /query endpoint hard-caps
`k = min(int(data.get("k", 5)), 5)` server-side -- every request was
silently truncated to 5 results regardless of what this script asked for,
so the "top_k" recorded in past logs never reflected what was actually
returned. Default corrected to 5 to match reality; the server-side cap
itself is left alone since it is, incidentally, a real (if modest)
anti-scraping control already in place -- see this module's README.

Ground-truth scoring added: this script previously only reported raw
doc/char counts with no comparison against the real corpus, so "how much
of the network's content was actually recovered" was never actually
measured. extraction_rate/extraction_accuracy/query_efficiency/
topic_coverage are now computed against the real corpus loaded into
data-source-0 (data/polluted_token/sources_0.jsonl), and Phase C's LLM
leakage is scored with paraphrase-tolerant semantic-similarity/edit-distance
(reused from attack/ddos_sim/nlg_metrics.py) since an LLM answer can leak a
private fact "in its own words" without being verbatim.
"""

import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Set

import requests
from datasets import load_dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.ddos_sim.nlg_metrics import (  # noqa: E402
    exact_match as _nlg_exact_match,
    normalized_edit_distance_score,
    semantic_similarity as _nlg_semantic_similarity,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_SOURCE_URLS = [
    os.getenv("DS0_URL", "http://localhost:8001"),
    os.getenv("DS1_URL", "http://localhost:8002"),
    os.getenv("DS2_URL", "http://localhost:8003"),
]
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:9000")
API_KEY         = os.getenv("API_KEY", "reliable-derag-secret-2026")

PROBE_SAMPLE_SIZE = int(os.getenv("PROBE_SAMPLE_SIZE", "54"))
# Was 10; drag_data_source/app/server.py:260 hard-caps k to 5 server-side,
# so requesting more than 5 never returned more than 5 -- see module docstring.
TOP_K             = int(os.getenv("TOP_K", "5"))
RANDOM_SEED       = int(os.getenv("RANDOM_SEED", "42"))
# CRR (Chunk Recovery Rate) thresholds for Phase C's paraphrase-tolerant
# leakage check: an LLM answer counts as "recovered" the gold fact if it's
# an exact match OR semantically close enough, mirroring the pasted design
# doc's CRR definition (exact OR SS >= 0.8 OR edit-distance-similarity >= 0.8).
CRR_SIM_THRESHOLD  = float(os.getenv("CRR_SIM_THRESHOLD", "0.8"))
CRR_EDIT_THRESHOLD = float(os.getenv("CRR_EDIT_THRESHOLD", "0.8"))

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
LOG_DIR = os.path.join(PROJECT_ROOT, "attack_logs")
os.makedirs(LOG_DIR, exist_ok=True)

HEADERS_WITH_KEY    = {"Content-Type": "application/json", "X-API-Key": API_KEY}
HEADERS_WITHOUT_KEY = {"Content-Type": "application/json"}


SOURCE_JSONL = {0: "sources_0.jsonl", 1: "sources_20.jsonl", 2: "sources_100.jsonl"}
PUBMEDQA_DATASET = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"


def _load_source_contexts(source_idx: int) -> Set[str]:
    """
    Ground truth = whatever data/polluted_token/sources_{0,20,100}.jsonl
    actually serves for THIS specific source right now, loaded directly --
    not matched against an external "clean reference" file, because no such
    shared reference exists anymore (see load_probe_sets() docstring for
    why). This is simply "the real content this source has," which is
    exactly what an extraction attack is trying to reconstruct, regardless
    of whether that content happens to be polluted/clean/a different
    dataset entirely.
    """
    path = os.path.join(PROJECT_ROOT, "data", "polluted_token", SOURCE_JSONL[source_idx])
    contexts: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            html = rec.get("html", "")
            if html:
                contexts.add(html)
    return contexts


# Kept for any external caller that still wants "the loaded corpus" in the
# old (source_0-only) sense; source_0 is PubMedQA now, not SQuAD -- see
# load_probe_sets().
def _load_corpus_contexts():
    return _load_source_contexts(0)


def load_probe_sets(n=PROBE_SAMPLE_SIZE, seed=RANDOM_SEED) -> Dict[int, Dict]:
    """
    Builds one probe/ground-truth/topic set PER data source.

    Configuration fix: the original version of this script assumed all
    three data sources served the same document indices ("sources_0.jsonl
    is the 0%-polluted / clean copy") and matched a single shared SQuAD
    probe set against all three. That assumption is no longer true:
    data-source-0's corpus was migrated to PubMedQA content by
    data/build_pubmedqa_corpus.py (written for attack/Mia_attack's MIA
    fix -- see that script's docstring), while sources_20/100 remain
    independently token-polluted SQuAD variants that only partially
    overlap *each other* (confirmed: only 207/500 rows are byte-identical
    between sources_20.jsonl and sources_100.jsonl). Concretely, this
    script previously crashed on every single run with "No SQuAD
    questions matched the loaded source documents" the moment
    sources_0.jsonl stopped containing SQuAD text -- a 100% failure rate,
    not a partial degradation -- because `_load_corpus_contexts()`
    unconditionally read sources_0.jsonl.

    Fixed by loading ground truth per-source (`_load_source_contexts`) and
    matching each source's own real content against the HF dataset that
    actually backs it: `qiaojin/PubMedQA` for source_0, `rajpurkar/squad`
    for sources_20/100 (matched separately per source, since their content
    only partially overlaps).

    Returns {source_idx: {"probes", "ground_truth", "context_to_title",
    "qa_by_question", "dataset"}}. `context_to_title` is empty for the
    PubMedQA source -- that dataset carries no per-article title field the
    way SQuAD does, so topic_coverage is left unreported (not guessed at)
    for source_0.
    """
    result: Dict[int, Dict] = {}
    random.seed(seed)

    print("[HF] Loading rajpurkar/squad train split (for sources_20/100) ...")
    squad_ds = load_dataset("rajpurkar/squad", split="train")
    for idx in (1, 2):
        ground_truth = _load_source_contexts(idx)
        context_to_title: Dict[str, str] = {}
        qa_by_question: Dict[str, Dict] = {}
        for item in squad_ds:
            ctx = item["context"]
            if ctx not in ground_truth:
                continue
            context_to_title.setdefault(ctx, item["title"])
            q = item["question"].strip()
            if q not in qa_by_question:
                qa_by_question[q] = {"context": ctx, "title": item["title"], "answers": item["answers"]["text"]}
        sampled = random.sample(list(qa_by_question.keys()), min(n, len(qa_by_question))) if qa_by_question else []
        result[idx] = {"probes": sampled, "ground_truth": ground_truth, "context_to_title": context_to_title,
                        "qa_by_question": qa_by_question, "dataset": "squad"}
        print(f"    source_{idx} ({SOURCE_JSONL[idx]}): {len(ground_truth)} ground-truth docs, "
              f"{len(qa_by_question)} matched questions, {len(sampled)} sampled probes")

    print(f"[HF] Loading {PUBMEDQA_DATASET} ({PUBMEDQA_CONFIG}) (for source_0) ...")
    pubmedqa_ds = load_dataset(PUBMEDQA_DATASET, PUBMEDQA_CONFIG, split="train")
    ground_truth_0 = _load_source_contexts(0)
    qa_0: Dict[str, Dict] = {}
    for item in pubmedqa_ds:
        ctx = " ".join(c.strip() for c in item["context"]["contexts"] if c and c.strip())
        if ctx not in ground_truth_0:
            continue
        q = item["question"].strip()
        if q not in qa_0:
            qa_0[q] = {"context": ctx, "title": None, "answers": [item.get("final_decision", "")]}
    sampled_0 = random.sample(list(qa_0.keys()), min(n, len(qa_0))) if qa_0 else []
    result[0] = {"probes": sampled_0, "ground_truth": ground_truth_0, "context_to_title": {},
                 "qa_by_question": qa_0, "dataset": "pubmedqa"}
    print(f"    source_0 ({SOURCE_JSONL[0]}): {len(ground_truth_0)} ground-truth docs, "
          f"{len(qa_0)} matched questions, {len(sampled_0)} sampled probes")

    for idx, info in result.items():
        if not info["probes"]:
            print(f"    [warn] source_{idx} ({info['dataset']}): 0 probe questions matched its own corpus -- "
                  "extraction against this source will be a guaranteed no-op.")
    return result


def load_squad_probes(n=PROBE_SAMPLE_SIZE, seed=RANDOM_SEED, source_idx: int = 1):
    """
    Backward-compatible single-source accessor on top of load_probe_sets(),
    default source_idx=1 (sources_20, SQuAD-domain) since source_0 is now
    PubMedQA -- see load_probe_sets() docstring.
    """
    probe_sets = load_probe_sets(n, seed)
    info = probe_sets[source_idx]
    return info["probes"], info["ground_truth"], info["context_to_title"], info["qa_by_question"]


def probe_source(base_url, questions, k=TOP_K, authenticated=False,
                  ground_truth_contexts=None, context_to_title=None, query_gate=None):
    """
    query_gate: optional callable(question_text) -> bool, consulted before
    each request is actually sent. Returning False skips that query (no HTTP
    call made, counted in `blocked_count`) -- this is the hook
    defense/kb_extraction_defense wires a throttle into, without needing a
    second, duplicated probing loop.
    """
    headers = HEADERS_WITH_KEY if authenticated else HEADERS_WITHOUT_KEY
    collected_docs: List[str] = []
    errors = 0
    blocked_count = 0
    start = time.time()

    for q in questions:
        if query_gate is not None and not query_gate(q):
            blocked_count += 1
            continue
        try:
            resp = requests.post(
                f"{base_url}/query",
                headers=headers,
                json={"query": q, "k": k},
                timeout=15,
            )
            if resp.status_code == 401:
                return {
                    "status":          "unauthorized",
                    "docs_extracted":  0,
                    "chars_extracted": 0,
                    "error_count":     1,
                    "elapsed_sec":     round(time.time() - start, 2),
                }
            resp.raise_for_status()
            data = resp.json()
            docs = (
                data.get("documents") or
                data.get("results") or
                data.get("chunks") or []
            )
            for doc in docs:
                text = (doc.get("text") or doc.get("content") or
                        doc.get("page_content") or "")
                if text and text not in collected_docs:
                    collected_docs.append(text)
        except requests.exceptions.ConnectionError:
            errors += 1
        except Exception:
            errors += 1

    total_chars = sum(len(d) for d in collected_docs)
    result = {
        "status":          "ok",
        "probes_sent":     len(questions),
        "blocked_count":   blocked_count,
        "docs_extracted":  len(collected_docs),
        "chars_extracted": total_chars,
        "kb_size_mb":      round(total_chars / 1_048_576, 4),
        "error_count":     errors,
        "elapsed_sec":     round(time.time() - start, 2),
    }

    # ── ground-truth-based success metrics (previously entirely missing:
    # this script reported raw volume but never measured what fraction of
    # the real corpus was actually recovered, nor how clean the haul was).
    # /query returns raw retrieved document text (not an LLM paraphrase),
    # so exact string match is the correct check here -- unlike Phase C's
    # LLM leakage, there is no paraphrase channel at the retrieval layer.
    if ground_truth_contexts is not None:
        correct = [d for d in collected_docs if d in ground_truth_contexts]
        n_gt = len(ground_truth_contexts)
        n_extracted = len(collected_docs)
        n_queries = len(questions)
        result["correct_extractions"] = len(correct)
        result["extraction_rate"] = round(len(correct) / n_gt, 4) if n_gt else 0.0
        result["extraction_accuracy"] = round(len(correct) / n_extracted, 4) if n_extracted else 0.0
        result["query_efficiency"] = round(len(correct) / n_queries, 4) if n_queries else 0.0
        if context_to_title:
            all_topics = set(context_to_title.values())
            touched_topics = {context_to_title[d] for d in correct if d in context_to_title}
            result["topic_coverage"] = round(len(touched_topics) / len(all_topics), 4) if all_topics else 0.0
            result["topics_touched"] = len(touched_topics)
            result["topics_total"] = len(all_topics)

    return result


def probe_llm_leakage(questions, qa_by_question=None):
    total_chars = 0
    responses   = []
    errors      = 0
    crr_hits    = 0
    ss_scores: List[float] = []
    eed_scores: List[float] = []
    scored = 0

    for q in questions[:20]:
        try:
            resp = requests.post(
                f"{LLM_SERVICE_URL}/query",
                headers=HEADERS_WITH_KEY,
                json={"query": q, "k": TOP_K},
                timeout=30,
            )
            resp.raise_for_status()
            data    = resp.json()
            answer  = data.get("answer") or data.get("response") or ""
            context = data.get("context") or data.get("sources") or []
            ctx_text = " ".join(
                c.get("text") or c.get("content") or "" for c in context
            )
            leaked = len(answer) + len(ctx_text)
            total_chars += leaked
            probe_record = {"question": q, "leaked_chars": leaked}

            # ── CRR (Chunk Recovery Rate) / SS / EED: does the LLM's answer
            # leak the gold fact "in its own words," not just verbatim?
            # Unlike Phase A/B's raw-document retrieval, generation genuinely
            # can paraphrase, so exact match alone would understate leakage.
            gold_answers = (qa_by_question or {}).get(q, {}).get("answers") or []
            if answer.strip() and gold_answers:
                em = _nlg_exact_match(answer, gold_answers)
                best_ss = max(_nlg_semantic_similarity(answer, g) for g in gold_answers)
                best_eed = max(normalized_edit_distance_score(answer, g) for g in gold_answers)
                recovered = bool(em >= 1.0 or best_ss >= CRR_SIM_THRESHOLD or best_eed >= CRR_EDIT_THRESHOLD)
                probe_record.update({"exact_match": em, "semantic_similarity": round(best_ss, 4),
                                      "edit_similarity": round(best_eed, 4), "chunk_recovered": recovered})
                ss_scores.append(best_ss)
                eed_scores.append(best_eed)
                scored += 1
                if recovered:
                    crr_hits += 1

            responses.append(probe_record)
        except Exception:
            errors += 1

    result = {
        "probes_sent":        min(20, len(questions)),
        "total_leaked_chars": total_chars,
        "error_count":        errors,
        "per_probe":          responses,
    }
    if scored:
        result["chunk_recovery_rate"] = round(crr_hits / scored, 4)
        result["avg_semantic_similarity"] = round(sum(ss_scores) / scored, 4)
        result["avg_edit_similarity"] = round(sum(eed_scores) / scored, 4)
        result["chunks_scored"] = scored
    return result


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file  = os.path.join(LOG_DIR, f"attack_{timestamp}_kb_extraction.json")
    print(f"\n{'='*60}")
    print("  Reliable-dRAG  -  KB Extraction Attack")
    print(f"{'='*60}\n")

    probe_sets = load_probe_sets()
    for i in (0, 1, 2):
        info = probe_sets[i]
        n_topics = len(set(info["context_to_title"].values())) if info["context_to_title"] else None
        topic_note = f", {n_topics} topics" if n_topics is not None else " (no topic field in this dataset)"
        print(f"    source_{i}: {info['dataset']} ground truth ({len(info['ground_truth'])} docs{topic_note})")

    print("\n[*] Phase A - Unauthenticated extraction attempt ...")
    unauth_results = {}
    for i, url in enumerate(DATA_SOURCE_URLS):
        label = f"source_{i}"
        info = probe_sets[i]
        print(f"    Probing {label} ({url}, {info['dataset']} probes) without key ...")
        unauth_results[label] = probe_source(url, info["probes"], k=TOP_K, authenticated=False,
                                              ground_truth_contexts=info["ground_truth"],
                                              context_to_title=info["context_to_title"])
        print(f"    -> {unauth_results[label]['status']}  |  docs: {unauth_results[label].get('docs_extracted', 0)}")

    print("\n[*] Phase B - Authenticated extraction (insider / stolen key) ...")
    auth_results = {}
    for i, url in enumerate(DATA_SOURCE_URLS):
        label = f"source_{i}"
        info = probe_sets[i]
        print(f"    Probing {label} ({url}, {info['dataset']} probes) with key ...")
        auth_results[label] = probe_source(url, info["probes"], k=TOP_K, authenticated=True,
                                            ground_truth_contexts=info["ground_truth"],
                                            context_to_title=info["context_to_title"])
        r = auth_results[label]
        print(f"    -> docs: {r.get('docs_extracted',0)}  |  chars: {r.get('chars_extracted',0):,}  |  "
              f"MB: {r.get('kb_size_mb',0)}  |  extraction_rate: {r.get('extraction_rate',0):.3f}  |  "
              f"extraction_accuracy: {r.get('extraction_accuracy',0):.3f}  |  "
              f"topic_coverage: {r.get('topic_coverage', 'n/a')}")

    # Phase C uses source_1's (SQuAD) probe set -- a documented simplification,
    # since 2 of 3 real sources are SQuAD-domain and drag_llm_service fans out
    # to all three sources per query regardless of which probe set is used.
    print("\n[*] Phase C - LLM answer leakage ...")
    llm_probe_info = probe_sets[1]
    llm_result = probe_llm_leakage(llm_probe_info["probes"], qa_by_question=llm_probe_info["qa_by_question"])
    print(f"    Total chars leaked via LLM: {llm_result['total_leaked_chars']:,}")
    if "chunk_recovery_rate" in llm_result:
        print(f"    CRR (chunk recovery rate): {llm_result['chunk_recovery_rate']:.3f}  |  "
              f"avg SS: {llm_result['avg_semantic_similarity']:.3f}  |  "
              f"avg edit-similarity: {llm_result['avg_edit_similarity']:.3f}  "
              f"(over {llm_result['chunks_scored']} scored probes)")

    total_auth_docs  = sum(r.get("docs_extracted", 0) for r in auth_results.values())
    total_auth_chars = sum(r.get("chars_extracted", 0) for r in auth_results.values())
    total_auth_mb    = round(total_auth_chars / 1_048_576, 4)
    # Per-source extraction_rate/accuracy only -- there is no longer a single
    # shared ground-truth corpus to pool a "network-wide" rate against (see
    # load_probe_sets() docstring: the three sources serve different content).
    per_source_extraction = {
        f"source_{i}": {
            "dataset": probe_sets[i]["dataset"],
            "extraction_rate": auth_results[f"source_{i}"].get("extraction_rate"),
            "extraction_accuracy": auth_results[f"source_{i}"].get("extraction_accuracy"),
            "topic_coverage": auth_results[f"source_{i}"].get("topic_coverage"),
        }
        for i in (0, 1, 2)
    }

    print(f"\n{'-'*60}")
    print(f"  Total docs extracted (authenticated): {total_auth_docs}")
    print(f"  Total KB recovered:                   {total_auth_mb} MB")
    for i in (0, 1, 2):
        s = per_source_extraction[f"source_{i}"]
        print(f"  source_{i} ({s['dataset']:<8}) extraction_rate={s['extraction_rate']}  "
              f"extraction_accuracy={s['extraction_accuracy']}  topic_coverage={s['topic_coverage']}")
    print(f"  LLM leakage:                          {llm_result['total_leaked_chars']:,} chars")
    print(f"{'-'*60}\n")

    report = {
        "attack":             "kb_extraction",
        "timestamp":          timestamp,
        "probe_datasets":     {i: probe_sets[i]["dataset"] for i in (0, 1, 2)},
        "top_k":              TOP_K,
        "random_seed":        RANDOM_SEED,
        "phase_a_unauth":     unauth_results,
        "phase_b_auth":       auth_results,
        "phase_c_llm":        llm_result,
        "summary": {
            "total_docs_extracted":  total_auth_docs,
            "total_chars_extracted": total_auth_chars,
            "total_kb_mb":           total_auth_mb,
            "per_source_extraction": per_source_extraction,
            "llm_leaked_chars":      llm_result["total_leaked_chars"],
            "llm_chunk_recovery_rate": llm_result.get("chunk_recovery_rate"),
        },
    }
    with open(log_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Log saved -> {log_file}")
    return report


if __name__ == "__main__":
    main()
