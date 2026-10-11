"""
defense/ssm_defense/evaluation/offline_validation.py

Docker/vLLM-free validation of verify_grounding() against realistic
grounding-farming filler, using the REAL retrieval/reranking pipeline
(Reranker + select_top_k_with_defense, the same code drag_llm_service/app/
server.py calls) and REAL SQuAD data -- just without the live Flask/Docker
stack or an actual LLM generating responses.

Approximation used (stated plainly, not hidden): since there is no live LLM
here, each sampled question's own gold short answer stands in for
`response_text`. This is a reasonable proxy for what a genuinely-grounded
answer looks like, and it is also exactly the kind of string
GroundingFarmingAttack's own filler is built to contain (its
`FRAGMENT_TEMPLATES` are seeded from the real SQuAD short-answer
distribution) -- so if anything this is generous to the attack, not
stacked against it.

For each sampled question:
  - one genuinely relevant candidate = the real gold context document
    (from whichever source actually holds it)
  - several genuinely irrelevant real candidates = random other real
    documents from the corpus (noise, as a normal retrieval pool would have)
  - the grounding-farming filler documents for the *target* source (built
    once via GroundingFarmingAttack.build_filler_documents, exactly as
    attack/ssm_score/run_grounding_farming.py does it)

All go through the same Reranker(method="hybrid") + select_top_k_with_defense
pipeline server.py uses (same config.yaml defaults), producing `selected`
with rerank_score/consensus_score attached. We then compare naive grounding
(substring match only) against verify_grounding()'s defended result.

Simplification: candidates' pre-rerank "score" (the original retriever's
own similarity score) is set to a neutral 0.0 for everything here, since
there is no live per-source FAISS index in this offline script and
orig_score_weight is only 0.1 in config.yaml (a minor influence on the
fused score, not the dominant signal).
"""
import json
import math
import os
import random
import re
import sys
from collections import Counter
from typing import Dict, List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'drag_llm_service'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'attack', 'ssm_score'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset  # noqa: E402
from grounding_farming_attack import GroundingFarmingAttack  # noqa: E402
from src.retriever.reranker import Reranker  # noqa: E402
from src.retriever.defense import select_top_k_with_defense  # noqa: E402
from grounding_verifier import verify_grounding  # noqa: E402

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
STOPWORDS = {
    "what", "which", "when", "where", "does", "did", "was", "were", "that",
    "this", "have", "from", "with", "about", "into", "over", "after",
    "before", "during", "their", "there", "been", "being", "also", "some",
    "many", "much", "most", "other", "those", "these", "would", "could",
    "should", "will", "shall", "than", "then", "them", "such", "only",
    "whom", "whose", "known",
}

TARGET_SOURCE = {"name": "sources_100", "url": "http://localhost:8003"}
OTHER_SOURCE_NAME = "sources_20"

SAMPLE_SIZE = 200
RANDOM_SEED = 42
N_CONTEXTS_PER_SOURCE = 3  # matches config.yaml retrieval.n_contexts
TOP_K = 3

DEFENSE_CFG = {
    "dedup_enabled": True,
    "dedup_similarity_threshold": 0.93,
    "dedup_same_source_only": True,
    "consensus_weight": 0.25,
    "consensus_min_sources": 2,
}


def _keyword_tokens(text: str, exclude=()) -> List[str]:
    words = WORD_RE.findall(text.lower())
    exclude_lower = {w.lower() for w in exclude}
    return [w for w in words if len(w) > 3 and w not in STOPWORDS and w not in exclude_lower]


def _load_per_source_contexts() -> Dict[str, set]:
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


def load_data(n=SAMPLE_SIZE, seed=RANDOM_SEED):
    """Like run_grounding_farming.load_squad_eval(), but retains the gold
    context TEXT per sampled row (needed to build a real candidate pool),
    which that function discards."""
    print("[HF] Loading rajpurkar/squad train split ...")
    ds = load_dataset("rajpurkar/squad", split="train")
    per_source = _load_per_source_contexts()
    corpus_contexts = per_source["sources_20"] | per_source["sources_100"]

    seen, rows, all_answers = set(), [], []
    title_counts: Dict[str, int] = {}
    title_keyword_counters: Dict[str, Counter] = {}
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
        source = "sources_20" if item["context"] in per_source["sources_20"] else "sources_100"
        rows.append({
            "question": q,
            "answer": ans_list[0],
            "gold_context": item["context"],
            "gold_source": source,
        })

    if not rows:
        raise RuntimeError("No SQuAD questions matched sources_20/100.")

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
    # sorted(), not list(): a set's iteration order depends on Python's
    # per-process string-hash randomization -- sampling from it would draw a
    # different sample on every run despite the fixed seed (see
    # content_shape_validation.py for the empirical confirmation of this).
    all_contexts = {"sources_20": sorted(per_source["sources_20"]), "sources_100": sorted(per_source["sources_100"])}
    print(f"[HF] Sampled {len(sampled)} matched questions; {len(title_counts)} titles")
    return sampled, all_answers, title_counts, title_keywords, all_contexts


def main():
    sampled, all_answers, title_counts, title_keywords, all_contexts = load_data()

    print("\n[*] Building grounding-farming filler documents for the target source ...")
    attacker = GroundingFarmingAttack(
        target_source=TARGET_SOURCE,
        num_titles=10,
        docs_per_title=4,
        keywords_per_doc=4,
    )
    filler_docs = attacker.build_filler_documents(title_counts, all_answers, title_keywords)
    print(f"    Built {len(filler_docs)} filler documents")

    reranker = Reranker(method="hybrid", hybrid_alpha=0.3, orig_score_weight=0.1, device="cpu")
    rng = random.Random(RANDOM_SEED)

    # Precompute a fixed local "index" per source once (a real per-source
    # FAISS index already narrows to each source's own best n_contexts
    # candidates before the orchestrator ever reranks across sources --
    # simulate that funnel here instead of dumping every filler doc into
    # the final pool at once, which no real query would ever see).
    SOURCE_INDEX_SIZE = 40
    source_index_texts = {
        name: rng.sample(texts, min(SOURCE_INDEX_SIZE, len(texts)))
        for name, texts in all_contexts.items()
    }
    source_index_embs = {
        name: reranker._embed_texts(texts) for name, texts in source_index_texts.items()
    }
    filler_texts = [d["text"] for d in filler_docs]
    filler_embs = reranker._embed_texts(filler_texts)

    def top_n_by_similarity(query_emb, texts, embs, n, ids=None, must_include_idx=None):
        if embs.size == 0:
            return []
        sims = embs @ query_emb
        order = list(np.argsort(-sims))
        chosen = order[:n]
        if must_include_idx is not None and must_include_idx not in chosen:
            chosen = chosen[: n - 1] + [must_include_idx]
        out = []
        for idx in chosen:
            out.append((texts[idx], float(sims[idx]), ids[idx] if ids else None))
        return out

    naive_filler_grounded = 0
    defended_filler_grounded = 0
    filler_selected_total = 0
    filler_scores_grounded: List[tuple] = []

    naive_genuine_grounded = 0
    defended_genuine_grounded = 0
    genuine_selected_total = 0
    genuine_scores_grounded: List[tuple] = []

    for row in sampled:
        question = row["question"]
        answer = row["answer"]
        gold_source = row["gold_source"]
        other_source = "sources_100" if gold_source == "sources_20" else "sources_20"

        q_emb, _ = reranker.embed_query_and_candidates(question, [])

        candidates = []

        # gold_source's own top-n_contexts (must include the true gold doc,
        # same way it would surely rank top-1 in a real per-source index for
        # the question actually asked about it).
        gold_pool_texts = source_index_texts[gold_source] + [row["gold_context"]]
        gold_pool_embs = np.vstack([source_index_embs[gold_source], reranker._embed_texts([row["gold_context"]])])
        gold_idx = len(gold_pool_texts) - 1
        for text, sim, _ in top_n_by_similarity(
            q_emb, gold_pool_texts, gold_pool_embs, N_CONTEXTS_PER_SOURCE, must_include_idx=gold_idx
        ):
            candidates.append({"id": "gold" if text == row["gold_context"] else f"noise_{gold_source}",
                                "text": text, "score": 0.0, "meta": {"source": gold_source}})

        # other_source's own top-n_contexts (ordinary, unfarmed real content).
        for text, sim, _ in top_n_by_similarity(
            q_emb, source_index_texts[other_source], source_index_embs[other_source], N_CONTEXTS_PER_SOURCE
        ):
            candidates.append({"id": f"noise_{other_source}", "text": text, "score": 0.0,
                                "meta": {"source": other_source}})

        # Farmed source's own top-n_contexts picked from among the filler docs
        # (what its internal retriever would actually surface for THIS query).
        ids = [d["id"] for d in filler_docs]
        for text, sim, fid in top_n_by_similarity(
            q_emb, filler_texts, filler_embs, N_CONTEXTS_PER_SOURCE, ids=ids
        ):
            candidates.append({"id": fid, "text": text, "score": 0.0,
                                "meta": {"source": TARGET_SOURCE["name"]}})

        selected = select_top_k_with_defense(
            reranker, question, candidates, top_k=TOP_K,
            reliability_scores=None, reliability_weight=0.0, cfg=DEFENSE_CFG,
        )

        norm_response = answer.lower().strip()
        selected_norm = [c["text"].lower() for c in selected]
        naive_grounded = [norm_response in ctx for ctx in selected_norm]
        defended_grounded = verify_grounding(naive_grounded, selected)

        for is_naive, is_def, cand in zip(naive_grounded, defended_grounded, selected):
            is_filler = cand["id"].startswith("grounding_farm_")
            score_tuple = (cand.get("rerank_score"), cand.get("consensus_score"))
            if is_filler:
                filler_selected_total += 1
                if is_naive:
                    naive_filler_grounded += 1
                    filler_scores_grounded.append(score_tuple)
                if is_def:
                    defended_filler_grounded += 1
            elif cand["id"] == "gold":
                genuine_selected_total += 1
                if is_naive:
                    naive_genuine_grounded += 1
                    genuine_scores_grounded.append(score_tuple)
                if is_def:
                    defended_genuine_grounded += 1

    print("\n" + "=" * 70)
    print("  SSM Grounding-Farming Defense -- Offline Validation")
    print("=" * 70)
    print(f"  Questions sampled: {len(sampled)}")
    print(f"\n  Filler docs selected into top-{TOP_K} (at all): {filler_selected_total}")
    print(f"    Naively 'grounded'   : {naive_filler_grounded}")
    print(f"    Still grounded after defense: {defended_filler_grounded}")
    if naive_filler_grounded:
        reduction = 1 - (defended_filler_grounded / naive_filler_grounded)
        print(f"    False-grounding reduction: {reduction:.1%}")
    print(f"\n  Genuine gold-context doc selected into top-{TOP_K}: {genuine_selected_total}")
    print(f"    Naively 'grounded'   : {naive_genuine_grounded}")
    print(f"    Still grounded after defense: {defended_genuine_grounded}")
    if naive_genuine_grounded:
        cost = 1 - (defended_genuine_grounded / naive_genuine_grounded)
        print(f"    Legitimate-grounding cost (false negatives introduced): {cost:.1%}")

    def _stats(label, pairs):
        rerank_vals = [p[0] for p in pairs if p[0] is not None]
        cons_vals = [p[1] for p in pairs if p[1] is not None]
        def fmt(vals):
            if not vals:
                return "n/a"
            vals = sorted(vals)
            return f"min={vals[0]:.3f} median={vals[len(vals)//2]:.3f} max={vals[-1]:.3f} (n={len(vals)})"
        print(f"  {label}: rerank_score {fmt(rerank_vals)} | consensus_score {fmt(cons_vals)}")

    print("\n  --- Score distributions among naively-grounded candidates (for threshold calibration) ---")
    _stats("Genuine (legitimate) ", genuine_scores_grounded)
    _stats("Filler (farming)     ", filler_scores_grounded)

    out = {
        "questions_sampled": len(sampled),
        "filler_selected_total": filler_selected_total,
        "naive_filler_grounded": naive_filler_grounded,
        "defended_filler_grounded": defended_filler_grounded,
        "genuine_selected_total": genuine_selected_total,
        "naive_genuine_grounded": naive_genuine_grounded,
        "defended_genuine_grounded": defended_genuine_grounded,
    }
    out_path = os.path.join(os.path.dirname(__file__), "offline_validation_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[+] Saved -> {out_path}")


if __name__ == "__main__":
    main()
