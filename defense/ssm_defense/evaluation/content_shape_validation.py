"""
defense/ssm_defense/evaluation/content_shape_validation.py

Docker-free validation of the two ingestion-time content-shape signals
(GroundingFarmDetector + SentenceLengthDetector) against:
  1. The real attack as implemented (hedge-attribution phrasing).
  2. An adaptive attacker who rewords the connective phrases to plain,
     non-hedging language while keeping the exact same exploit structure
     (same topic anchor, same disconnected-fact-per-sentence shape) --
     simulating an attacker who read content_shape_detector.py and tried
     to evade it.
  3. Real clean corpus documents (false-positive rate), for both signals
     and their OR-combination (either firing rejects the document).

This is a different validation than offline_validation.py (which tests the
disabled relevance-score gating attempt against the real retrieval
pipeline) -- this one is pure text-shape analysis, no embeddings/reranker
needed, so it runs fast.
"""
import json
import math
import os
import random
import re
import sys
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'attack', 'ssm_score'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset  # noqa: E402
import grounding_farming_attack as gfa  # noqa: E402
from content_shape_detector import GroundingFarmDetector  # noqa: E402
from sentence_length_detector import SentenceLengthDetector  # noqa: E402

SEED = 42
FP_SAMPLE_SIZE = 150
TARGET_SOURCE = {"name": "sources_100", "url": "http://localhost:8003"}

# Adaptive-attacker templates: same information structure (one topic anchor
# + one disconnected fact per sentence) as the real attack's
# FRAGMENT_TEMPLATES/SUBTOPIC_TEMPLATES, but with every hedge-attribution
# phrase replaced by a plain, neutral connective.
ADAPTIVE_FRAGMENT_TEMPLATES = [
    "{t} is connected to {f}.",
    "{f} relates to {t}.",
    "{t} includes {f}.",
    "{f} is part of {t}'s history.",
]
ADAPTIVE_SUBTOPIC_TEMPLATES = [
    "{t} involves {k}.",
    "{t} relates to {k}.",
    "{t} includes {k}.",
    "{t} connects to {k}.",
    "{t} covers {k}.",
    "{t} features {k}.",
]
ADAPTIVE_DOMAIN_TEMPLATES = [
    "{t} is a notable subject.",
    "{t} has been covered extensively.",
]


def load_corpus_and_titles():
    print("[HF] Loading rajpurkar/squad train split ...")
    ds = load_dataset("rajpurkar/squad", split="train")
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
    corpus_contexts = per_source["sources_20"] | per_source["sources_100"]

    word_re = re.compile(r"[^\W\d_]+", re.UNICODE)
    stopwords = {
        "what", "which", "when", "where", "does", "did", "was", "were", "that",
        "this", "have", "from", "with", "about", "into", "over", "after",
        "before", "during", "their", "there", "been", "being", "also", "some",
        "many", "much", "most", "other", "those", "these", "would", "could",
        "should", "will", "shall", "than", "then", "them", "such", "only",
        "whom", "whose", "known",
    }

    def keywords(text, exclude=()):
        words = word_re.findall(text.lower())
        ex = {w.lower() for w in exclude}
        return [w for w in words if len(w) > 3 and w not in stopwords and w not in ex]

    title_counts, title_kw_counters, all_answers = {}, {}, []
    for item in ds:
        ans = item["answers"]["text"]
        if ans:
            all_answers.extend(a for a in ans if a and len(a) < 40)
        if item["context"] not in corpus_contexts:
            continue
        title = item.get("title", "").replace("_", " ").strip()
        if title:
            title_counts[title] = title_counts.get(title, 0) + 1
            c = title_kw_counters.setdefault(title, Counter())
            c.update(keywords(item["question"], exclude=title.split()))

    num_titles = len(title_kw_counters)
    doc_freq = Counter()
    for c in title_kw_counters.values():
        doc_freq.update(c.keys())
    title_keywords = {}
    for t, c in title_kw_counters.items():
        scored = {w: tf * math.log((1 + num_titles) / (1 + doc_freq[w])) for w, tf in c.items()}
        title_keywords[t] = [w for w, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:40]]

    return per_source, title_counts, title_keywords, all_answers


def build_filler(title_counts, title_keywords, all_answers):
    attacker = gfa.GroundingFarmingAttack(
        target_source=TARGET_SOURCE, num_titles=10, docs_per_title=4, keywords_per_doc=4,
    )
    return attacker.build_filler_documents(title_counts, all_answers, title_keywords)


def evaluate(label, docs, hedge_detector, length_detector):
    texts = [d["text"] for d in docs]
    hedge_verdicts = hedge_detector.classify_batch(texts)
    length_verdicts = length_detector.classify_batch(texts)
    hedge_flagged = sum(1 for v in hedge_verdicts if v.flagged)
    length_flagged = sum(1 for v in length_verdicts if v.flagged)
    combined_flagged = sum(
        1 for h, l in zip(hedge_verdicts, length_verdicts) if h.flagged or l.flagged
    )
    n = len(docs)
    print(f"{label} (n={n}):")
    print(f"    hedge-attribution detector : {hedge_flagged}/{n} ({hedge_flagged/n:.1%})")
    print(f"    sentence-length detector   : {length_flagged}/{n} ({length_flagged/n:.1%})")
    print(f"    combined (OR)              : {combined_flagged}/{n} ({combined_flagged/n:.1%})")
    return {
        "n": n, "hedge_flagged": hedge_flagged, "length_flagged": length_flagged,
        "combined_flagged": combined_flagged,
    }


def main():
    per_source, title_counts, title_keywords, all_answers = load_corpus_and_titles()

    print("\n[*] Building ORIGINAL (hedge-phrase) filler ...")
    original_filler = build_filler(title_counts, title_keywords, all_answers)

    print("[*] Building ADAPTIVE (reworded, no hedge phrases) filler ...")
    orig_fragment, orig_subtopic, orig_domain = (
        gfa.FRAGMENT_TEMPLATES, gfa.SUBTOPIC_TEMPLATES, gfa.DOMAIN_TEMPLATES
    )
    gfa.FRAGMENT_TEMPLATES = ADAPTIVE_FRAGMENT_TEMPLATES
    gfa.SUBTOPIC_TEMPLATES = ADAPTIVE_SUBTOPIC_TEMPLATES
    gfa.DOMAIN_TEMPLATES = ADAPTIVE_DOMAIN_TEMPLATES
    try:
        adaptive_filler = build_filler(title_counts, title_keywords, all_answers)
    finally:
        gfa.FRAGMENT_TEMPLATES, gfa.SUBTOPIC_TEMPLATES, gfa.DOMAIN_TEMPLATES = (
            orig_fragment, orig_subtopic, orig_domain
        )

    print("\n" + "=" * 70)
    print("  SSM Grounding-Farming Content-Shape Defense -- Validation")
    print("=" * 70)

    results = {}
    results["original_filler"] = evaluate(
        "Original filler (as the attack ships)",
        original_filler, GroundingFarmDetector(), SentenceLengthDetector(),
    )
    results["adaptive_filler"] = evaluate(
        "Adaptive filler (hedge phrases reworded)",
        adaptive_filler, GroundingFarmDetector(), SentenceLengthDetector(),
    )

    print()
    rng = random.Random(SEED)
    # sorted(), not list(): a set's iteration order depends on Python's
    # per-process string-hash randomization, so sampling from list(a_set)
    # draws a DIFFERENT sample every run despite the fixed seed -- confirmed
    # empirically (false-positive rate varied 0.7%/2.7%/3.3% -> 0.0%/4.7%/4.7%
    # across two runs with identical SEED). Sorting first makes the input
    # order deterministic, so the seeded sample is actually reproducible.
    clean_sample = rng.sample(sorted(per_source["sources_20"]), FP_SAMPLE_SIZE)
    clean_docs = [{"text": t} for t in clean_sample]
    results["clean_corpus"] = evaluate(
        "Real clean corpus documents (false-positive check)",
        clean_docs, GroundingFarmDetector(), SentenceLengthDetector(),
    )

    out_path = os.path.join(os.path.dirname(__file__), "content_shape_validation_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved -> {out_path}")


if __name__ == "__main__":
    main()
