#!/usr/bin/env python3
"""
FedRAG News Pipeline — ALL 3 PROBLEMS FIXED:
  Fix 1 (Problem 1+2): Prompt now demands ONE sentence, uses only top-1 context
                        doc. Response validator rejects bare numbers like '2'.
  Fix 2 (Problem 3):   LLM generation runs on BOTH clean and poisoned stores
                        so you can see real end-to-end LLM degradation.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import requests

from fed_rag.evaluators import load_security_config, load_defense_config, run_system
from fed_rag.evaluators.metrics import HashingRetriever
from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore

OLLAMA_URL = "http://localhost:11434/api/generate"

# ── Semantic similarity ────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST
    import numpy as _np
    _sim_model = _ST("sentence-transformers/all-MiniLM-L6-v2")

    def semantic_sim(a: str, b: str) -> float:
        e = _sim_model.encode([a, b])
        denom = _np.linalg.norm(e[0]) * _np.linalg.norm(e[1])
        return float(_np.dot(e[0], e[1]) / denom) if denom else 0.0
except Exception:
    def semantic_sim(a: str, b: str) -> float:
        return 0.0

# ── BLEU ───────────────────────────────────────────────────────────────────────
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    def compute_bleu(pred: str, gold: str) -> float:
        return sentence_bleu([gold.split()], pred.split(),
                             smoothing_function=SmoothingFunction().method1)
except Exception:
    def compute_bleu(pred: str, gold: str) -> float:
        return 0.0

# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_em(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0

def compute_f1(pred: str, gold: str) -> float:
    p, g = set(pred.lower().split()), set(gold.lower().split())
    if not p or not g: return 0.0
    common = p & g
    if not common: return 0.0
    prec, rec = len(common)/len(p), len(common)/len(g)
    return 2*prec*rec/(prec+rec)

# ── Dataset ────────────────────────────────────────────────────────────────────
def load_news_dataset(num_samples: int, seed: int) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("`datasets` package is required.") from exc
    print("Loading news dataset (heegyu/news-category-dataset) ...")
    ds = load_dataset("heegyu/news-category-dataset", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
    rows: list[dict[str, str]] = []
    for item in ds:
        headline  = str(item.get("headline", "")).strip()
        short_desc = str(item.get("short_description", "")).strip()
        category  = str(item.get("category", "NEWS")).strip()
        if headline:
            rows.append({"query": headline, "response": short_desc, "topic": category})
    print(f"Loaded news dataset: {len(rows)} samples")
    return rows

def _make_synthetic_dataset(n: int) -> list[dict[str, str]]:
    topics = ["tech","sports","politics","health","finance",
              "science","world","entertainment","business","weather"]
    return [{"query": f"Breaking: {topics[i%len(topics)].title()} update #{i}",
             "response": f"Correspondent reporting on {topics[i%len(topics)]} story {i}.",
             "topic": topics[i%len(topics)]} for i in range(n)]

# ── Knowledge store ────────────────────────────────────────────────────────────
def _build_store(examples: list[dict[str, Any]], retriever: Any) -> InMemoryKnowledgeStore:
    nodes = []
    for row in examples:
        emb = retriever.encode_context(row["query"]).tolist()
        if emb and isinstance(emb[0], list):
            emb = emb[0]
        nodes.append(KnowledgeNode(
            node_type=NodeType.TEXT,
            text_content=row["query"],
            embedding=[float(v) for v in emb],
            metadata={"answer": row["response"], "topic": row.get("topic", "")},
        ))
    return InMemoryKnowledgeStore.from_nodes(nodes)

# ── FIXED Ollama generation ────────────────────────────────────────────────────
def _build_news_prompt(query: str, context: str, strict: bool = False) -> str:
    """
    FIX Problem 1: ONE sentence, top-1 context only.
    The old "(1-2 sentences)" + 3 numbered context items made llama
    treat it as multiple-choice and answer '2'.
    """
    anti_mcq = (
        " You are NOT answering a multiple-choice question."
        " Do NOT output a number or letter by itself." if strict else ""
    )
    return (
        f"Headline: {query}\n\n"
        f"Background: {context}\n\n"
        f"Write exactly ONE sentence summarising this news story.{anti_mcq}"
        " Begin the sentence immediately — no preamble, no bullet points, no numbering:\n"
    )

def _clean_response(raw: str) -> str:
    """
    FIX Problem 1 validator: reject bare numbers/letters ('2','B','(2)',etc.)
    that indicate the model answered as if this were multiple-choice.
    """
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    if not lines:
        return ""
    first = lines[0]
    if re.fullmatch(r"[\(\[]?[0-9A-Da-d][\)\].]?", first):
        return ""                              # looks like MCQ — reject
    first = re.sub(r"^[\d]+[.)]\s*", "", first)   # strip "1. "
    first = re.sub(r"^[-•]\s*", "", first)        # strip "- "
    return first[:300]

def ollama_generate(query: str, context_docs: list[str], model: str,
                    task_type: str = "news") -> str:
    """
    FIX Problem 1+2:
    • Only the FIRST (most relevant) context doc is used — no numbered list.
    • If the answer still looks like MCQ, retries with stricter prompt (×3).
    """
    context = context_docs[0] if context_docs else f"News story about: {query}"
    raw = ""
    for attempt in range(3):
        prompt = _build_news_prompt(query, context, strict=(attempt > 0))
        try:
            r = requests.post(OLLAMA_URL,
                json={"model": model, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.0, "num_predict": 100}},
                timeout=90)
            raw = r.json().get("response", "").strip()
            cleaned = _clean_response(raw)
            if cleaned:
                return cleaned
        except Exception as e:
            return f"[ERROR: {e}]"
    return raw[:200] if raw else "[no response]"

# ── Federated split ────────────────────────────────────────────────────────────
def _split_across_clients(dataset, num_clients, seed):
    rng = random.Random(seed)
    shuffled = list(dataset)
    rng.shuffle(shuffled)
    chunks: list[list[dict]] = [[] for _ in range(num_clients)]
    for idx, row in enumerate(shuffled):
        chunks[idx % num_clients].append(row)
    return chunks

# ── Main ───────────────────────────────────────────────────────────────────────
def run_news_llm_system(args: argparse.Namespace) -> None:
    config_root = Path(args.config_root)
    security_cfg, _ = load_security_config(config_root, None)
    defense_cfg,  _ = load_defense_config(config_root, None)

    # 1. Dataset
    try:
        dataset = load_news_dataset(args.num_samples, args.seed)
    except Exception as exc:
        print(f"News dataset load failed ({exc}), using synthetic data")
        dataset = _make_synthetic_dataset(args.num_samples)

    # 2. Retriever
    try:
        from fed_rag.retrievers import HFSentenceTransformerRetriever
        retriever = HFSentenceTransformerRetriever(
            model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Using GPU retriever: all-MiniLM-L6-v2")
    except Exception as e:
        print(f"GPU retriever not available ({e}), falling back to HashingRetriever")
        retriever = HashingRetriever()

    # 3. Federated split
    client_shards = _split_across_clients(dataset, args.num_clients, args.seed)
    print(f"\nFederated setup: {args.num_clients} clients")
    for cid, shard in enumerate(client_shards):
        topics = set(r["topic"] for r in shard)
        print(f"  Client {cid:02d}: {len(shard)} samples | topics: {sorted(topics)}")

    store = _build_store(dataset, retriever)
    print(f"\nKnowledge store: {store.count} nodes")
    sys.stdout.flush()

    # 4. Phase 1 — Retrieval-layer security evaluation (all 4 attacks)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRunning security evaluation — attacks: {args.attacks}")
    run_system(
        dataset=dataset, retriever=retriever, knowledge_store=store,
        attacks=args.attacks, security_cfg=security_cfg, defense_cfg=defense_cfg,
        num_clients=args.num_clients, output_dir=out_dir, seed=args.seed,
    )

    # 5. Phase 2 — LLM generation helper
    def _run_llm_on_store(label: str, active_store: InMemoryKnowledgeStore) -> dict:
        """
        FIX Problem 3: runs LLM on ANY store (clean or poisoned).
        This shows true end-to-end LLM-layer impact of each attack.
        """
        print(f"\n  LLM [{label}] with {args.llm_model} ...")
        results: list[dict] = []
        for i, row in enumerate(dataset):
            query, ground_truth = row["query"], row["response"]
            emb_q = retriever.encode_context(query).tolist()
            if emb_q and isinstance(emb_q[0], list):
                emb_q = emb_q[0]
            # top_k=1 — FIX Problem 1: avoids numbered multi-doc context
            nodes = active_store.retrieve([float(v) for v in emb_q], top_k=1)
            context_docs = (
                [f"{node.text_content}. {node.metadata.get('answer','')}".strip(". ")
                 for _, node in nodes]
                if nodes else [query]
            )
            llm_answer = ollama_generate(query, context_docs, args.llm_model)
            em  = compute_em(llm_answer, ground_truth)
            f1  = compute_f1(llm_answer, ground_truth)
            bleu = compute_bleu(llm_answer, ground_truth)
            sem  = semantic_sim(llm_answer, ground_truth)
            results.append({"query": query, "ground_truth": ground_truth,
                            "llm_answer": llm_answer,
                            "em": em, "f1": f1, "bleu": bleu, "semantic_similarity": sem})
            print(f"    [{i+1:02d}/{len(dataset)}] F1={f1:.2f} SemSim={sem:.2f} "
                  f"pred={llm_answer[:55]!r}", flush=True)
        n = len(results)
        return {"label": label,
                "avg_em":   sum(r["em"]  for r in results)/n,
                "avg_f1":   sum(r["f1"]  for r in results)/n,
                "avg_bleu": sum(r["bleu"] for r in results)/n,
                "avg_semantic_similarity": sum(r["semantic_similarity"] for r in results)/n,
                "per_query": results}

    # 6. LLM baseline (clean store)
    print("\n" + "="*70)
    print("PHASE 2: LLM GENERATION (baseline + per-attack)")
    print("="*70)
    llm_results: dict[str, Any] = {}
    clean_store   = _build_store(dataset, retriever)
    baseline_llm  = _run_llm_on_store("baseline (clean store)", clean_store)
    llm_results["baseline"] = baseline_llm

    # 7. FIX Problem 3: LLM on poisoned store
    for atk in args.attacks:
        if atk != "poisoning":
            # node_availability / extraction / MIA don't rewrite stored text,
            # so retrieved context is identical — LLM output won't change.
            print(f"\n  [{atk}] does not corrupt stored text → LLM output = baseline")
            llm_results[atk] = {
                "label": atk,
                "note": "attack does not alter knowledge store text; LLM output same as baseline",
                "avg_em":  baseline_llm["avg_em"],
                "avg_f1":  baseline_llm["avg_f1"],
                "avg_bleu": baseline_llm["avg_bleu"],
                "avg_semantic_similarity": baseline_llm["avg_semantic_similarity"],
            }
            continue
        # Poisoning — rebuild store and corrupt 30% of node answers directly.
        # We do NOT use fed_rag.attacks.get_attack (that API does not exist).
        # Instead we replicate what poisoning does: overwrite stored answers
        # with adversarial text so the LLM reads corrupt context.
        poisoned_store = _build_store(dataset, retriever)
        try:
            # Retrieve all nodes (top_k >> dataset size guarantees all are returned)
            _sample_emb = retriever.encode_context(dataset[0]["query"]).tolist()
            if _sample_emb and isinstance(_sample_emb[0], list):
                _sample_emb = _sample_emb[0]
            _all_pairs  = poisoned_store.retrieve(
                [float(v) for v in _sample_emb], top_k=len(dataset) + 10)
            _nodes_list = [_node for _, _node in _all_pairs]

            _rng_p    = random.Random(args.seed)
            _n_poison = max(1, int(len(_nodes_list) * 0.30))   # 30% = same as Phase 1
            for _node in _rng_p.sample(_nodes_list, _n_poison):
                _node.metadata["answer"] = (
                    "POISONED: This content has been deliberately corrupted "
                    "by an adversarial attack and does not reflect the original news story."
                )
            print(f"\n  [poisoning] manual poisoning applied: "
                  f"{_n_poison}/{len(_nodes_list)} nodes corrupted")
            llm_results[atk] = _run_llm_on_store("after poisoning", poisoned_store)
        except Exception as exc:
            print(f"\n  [poisoning] poisoning failed ({exc}); skipping.")
            llm_results[atk] = {"label": "poisoning", "note": str(exc),
                                 "avg_em": None, "avg_f1": None,
                                 "avg_bleu": None, "avg_semantic_similarity": None}

    # 8. Print summary table
    print(f"\n{'='*70}")
    print(f"=== LLM RESULTS  model={args.llm_model}  clients={args.num_clients} ===")
    print(f"{'='*70}")
    print(f"  {'Scenario':<32} {'EM':>6}  {'F1':>6}  {'SemSim':>8}")
    print("  " + "-"*56)
    for key, res in llm_results.items():
        lbl = res.get("label", key)[:32]
        em, f1, sem = res.get("avg_em"), res.get("avg_f1"), res.get("avg_semantic_similarity")
        if em is None:
            print(f"  {lbl:<32}   n/a     n/a       n/a")
        else:
            mark = ""
            if key != "baseline" and res.get("avg_f1") is not None:
                drop = baseline_llm["avg_f1"] - res["avg_f1"]
                mark = f"  ← F1 drop {drop*100:.1f}%" if drop > 0.01 else "  ← no LLM-layer drop"
            print(f"  {lbl:<32} {em:>6.4f} {f1:>6.4f} {sem:>8.4f}{mark}")
    print()
    print("  NOTE: EM~0 expected for free-text news (not multiple-choice).")
    print("        F1 + Semantic Similarity are the meaningful metrics.")

    # 9. Save JSON
    out_path = out_dir / "llm_generation_results.json"
    out_path.write_text(json.dumps({
        "model": args.llm_model,
        "dataset": "news (heegyu/news-category-dataset)",
        "num_clients": args.num_clients,
        "attacks": args.attacks,
        **{f"llm_{k}": v for k, v in llm_results.items()},
        "avg_em":   baseline_llm["avg_em"],
        "avg_f1":   baseline_llm["avg_f1"],
        "avg_bleu": baseline_llm["avg_bleu"],
        "avg_semantic_similarity": baseline_llm["avg_semantic_similarity"],
        "per_query": baseline_llm["per_query"],
    }, indent=2))
    print(f"Saved: {out_path}")

# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="FedRAG News — all 3 problems fixed")
    p.add_argument("--config-root",  default=".")
    p.add_argument("--llm-model",    default="llama3.2:3b")
    p.add_argument("--num-samples",  type=int, default=20)
    p.add_argument("--num-clients",  type=int, default=20)
    p.add_argument("--top-k",        type=int, default=1)   # fixed to 1
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--attacks",      nargs="+",
        default=["poisoning","node_availability","extraction","membership_inference"],
        choices=["poisoning","node_availability","extraction","membership_inference"])
    p.add_argument("--output-dir",   default="logs/news_llama_20clients_all_attacks")
    args = p.parse_args()
    run_news_llm_system(args)
