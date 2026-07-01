"""
attack/kb_extraction/run_attack.py
Knowledge-Base Extraction attack for Reliable-dRAG.
Probe questions loaded from HuggingFace rajpurkar/squad (validation split).
"""

import json
import os
import random
import time
from datetime import datetime

import requests
from datasets import load_dataset

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_SOURCE_URLS = [
    os.getenv("DS0_URL", "http://localhost:8001"),
    os.getenv("DS1_URL", "http://localhost:8002"),
    os.getenv("DS2_URL", "http://localhost:8003"),
]
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:9000")
API_KEY         = os.getenv("API_KEY", "reliable-derag-secret-2026")

PROBE_SAMPLE_SIZE = int(os.getenv("PROBE_SAMPLE_SIZE", "54"))
TOP_K             = int(os.getenv("TOP_K", "10"))
RANDOM_SEED       = int(os.getenv("RANDOM_SEED", "42"))

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "attack_logs")
os.makedirs(LOG_DIR, exist_ok=True)

HEADERS_WITH_KEY    = {"Content-Type": "application/json", "X-API-Key": API_KEY}
HEADERS_WITHOUT_KEY = {"Content-Type": "application/json"}


def load_squad_probes(n=PROBE_SAMPLE_SIZE, seed=RANDOM_SEED):
    print(f"[HF] Loading rajpurkar/squad validation split ...")
    ds = load_dataset("rajpurkar/squad", split="validation")
    seen, questions = set(), []
    for item in ds:
        q = item["question"].strip()
        if q not in seen:
            seen.add(q)
            questions.append(q)
    random.seed(seed)
    sampled = random.sample(questions, min(n, len(questions)))
    print(f"[HF] Sampled {len(sampled)} probe questions (seed={seed})")
    return sampled


def probe_source(base_url, questions, k=TOP_K, authenticated=False):
    headers = HEADERS_WITH_KEY if authenticated else HEADERS_WITHOUT_KEY
    collected_docs = []
    errors = 0
    start = time.time()

    for q in questions:
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
    return {
        "status":          "ok",
        "probes_sent":     len(questions),
        "docs_extracted":  len(collected_docs),
        "chars_extracted": total_chars,
        "kb_size_mb":      round(total_chars / 1_048_576, 4),
        "error_count":     errors,
        "elapsed_sec":     round(time.time() - start, 2),
    }


def probe_llm_leakage(questions):
    total_chars = 0
    responses   = []
    errors      = 0
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
            responses.append({"question": q, "leaked_chars": leaked})
        except Exception:
            errors += 1
    return {
        "probes_sent":        min(20, len(questions)),
        "total_leaked_chars": total_chars,
        "error_count":        errors,
        "per_probe":          responses,
    }


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file  = os.path.join(LOG_DIR, f"attack_{timestamp}_kb_extraction.json")
    print(f"\n{'='*60}")
    print("  Reliable-dRAG  -  KB Extraction Attack")
    print(f"{'='*60}\n")

    probes = load_squad_probes()

    print("[*] Phase A - Unauthenticated extraction attempt ...")
    unauth_results = {}
    for i, url in enumerate(DATA_SOURCE_URLS):
        label = f"source_{i}"
        print(f"    Probing {label} ({url}) without key ...")
        unauth_results[label] = probe_source(url, probes, k=TOP_K, authenticated=False)
        print(f"    -> {unauth_results[label]['status']}  |  docs: {unauth_results[label].get('docs_extracted', 0)}")

    print("\n[*] Phase B - Authenticated extraction (insider / stolen key) ...")
    auth_results = {}
    for i, url in enumerate(DATA_SOURCE_URLS):
        label = f"source_{i}"
        print(f"    Probing {label} ({url}) with key ...")
        auth_results[label] = probe_source(url, probes, k=TOP_K, authenticated=True)
        r = auth_results[label]
        print(f"    -> docs: {r.get('docs_extracted',0)}  |  chars: {r.get('chars_extracted',0):,}  |  MB: {r.get('kb_size_mb',0)}")

    print("\n[*] Phase C - LLM answer leakage ...")
    llm_result = probe_llm_leakage(probes)
    print(f"    Total chars leaked via LLM: {llm_result['total_leaked_chars']:,}")

    total_auth_docs  = sum(r.get("docs_extracted", 0) for r in auth_results.values())
    total_auth_chars = sum(r.get("chars_extracted", 0) for r in auth_results.values())
    total_auth_mb    = round(total_auth_chars / 1_048_576, 4)

    print(f"\n{'─'*60}")
    print(f"  Total docs extracted (authenticated): {total_auth_docs}")
    print(f"  Total KB recovered:                   {total_auth_mb} MB")
    print(f"  LLM leakage:                          {llm_result['total_leaked_chars']:,} chars")
    print(f"{'─'*60}\n")

    report = {
        "attack":             "kb_extraction",
        "timestamp":          timestamp,
        "probe_dataset":      "rajpurkar/squad",
        "probe_sample_size":  len(probes),
        "top_k":              TOP_K,
        "random_seed":        RANDOM_SEED,
        "phase_a_unauth":     unauth_results,
        "phase_b_auth":       auth_results,
        "phase_c_llm":        llm_result,
        "summary": {
            "total_docs_extracted":  total_auth_docs,
            "total_chars_extracted": total_auth_chars,
            "total_kb_mb":           total_auth_mb,
            "llm_leaked_chars":      llm_result["total_leaked_chars"],
        },
    }
    with open(log_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Log saved -> {log_file}")
    return report


if __name__ == "__main__":
    main()
