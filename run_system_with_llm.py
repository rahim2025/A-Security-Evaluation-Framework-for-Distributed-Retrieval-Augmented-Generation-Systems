#!/usr/bin/env python3
"""System-mode evaluation WITH Ollama LLM generator."""
from __future__ import annotations
import argparse, json, random, requests, sys
from pathlib import Path
from typing import Any

from fed_rag.evaluators import load_security_config, load_defense_config, run_system
from fed_rag.evaluators.centralized import _load_drag_config, _load_drag_dataset
from fed_rag.evaluators.metrics import HashingRetriever
from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore

OLLAMA_URL = "http://localhost:11434/api/generate"

try:
    from sentence_transformers import SentenceTransformer as _ST
    import numpy as _np
    _sim_model = _ST('sentence-transformers/all-MiniLM-L6-v2')
    def semantic_sim(a, b):
        e = _sim_model.encode([a, b])
        return float(_np.dot(e[0], e[1]) / (_np.linalg.norm(e[0]) * _np.linalg.norm(e[1])))
except Exception:
    def semantic_sim(a, b): return 0.0

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    def compute_bleu(pred, gold):
        sf = SmoothingFunction().method1
        return sentence_bleu([gold.split()], pred.split(), smoothing_function=sf)
except Exception:
    def compute_bleu(pred, gold): return 0.0


def ollama_generate(query: str, context_docs: list[str], model: str) -> str:
    context = "\n\n".join(context_docs)
    prompt = (
        f"You are a multiple-choice QA system. Use the context to answer.\n"
        f"Reply with ONLY the index number: 0, 1, 2, or 3. Nothing else.\n\n"
        f"Context (similar Q&A pairs):\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer index (0, 1, 2, or 3):"
    )
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": model, "prompt": prompt,
            "stream": False, "options": {"temperature": 0.0, "num_predict": 30}
        }, timeout=60)
        raw = r.json().get("response", "").strip()
        # Strip verbose explanation — keep only text before first "(" or newline
        import re
        clean = re.split(r'[\n(]', raw)[0].strip().strip("'\"")
        return clean
    except Exception as e:
        return f"[ERROR: {e}]"

def compute_em(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0

def compute_f1(pred: str, gold: str) -> float:
    p_toks = set(pred.lower().split())
    g_toks = set(gold.lower().split())
    if not p_toks or not g_toks:
        return 0.0
    common = p_toks & g_toks
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec  = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)

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

def _make_synthetic_dataset(n: int) -> list[dict[str, str]]:
    topics = ["cryptography","medicine","finance","biology","law",
              "networking","history","physics","privacy","safety"]
    return [{"query": f"Fact {i}: what is the verified answer for {topics[i%len(topics)]}?",
             "response": f"verified-{topics[i%len(topics)]}-answer-{i}",
             "topic": topics[i % len(topics)]} for i in range(n)]

def run_llm_system(args: argparse.Namespace) -> None:
    config_root = Path(args.config_root)
    security_cfg, _ = load_security_config(config_root, None)
    defense_cfg,  _ = load_defense_config(config_root, None)

    rng = random.Random(args.seed)
    try:
        drag_cfg = _load_drag_config(config_root, args.llm, args.dataset)
        dataset  = _load_drag_dataset(config_root, drag_cfg, args.num_samples, args.seed, "huggingface")
        print(f"Loaded MMLU dataset: {len(dataset)} samples")
    except Exception as exc:
        print(f"MMLU load failed ({exc}), using synthetic data")
        dataset = _make_synthetic_dataset(args.num_examples)

    try:
        from fed_rag.retrievers import HFSentenceTransformerRetriever
        retriever = HFSentenceTransformerRetriever(model_name='sentence-transformers/all-MiniLM-L6-v2')
        print('Using GPU retriever: all-MiniLM-L6-v2')
    except Exception as e:
        print(f'GPU retriever not available ({e}), falling back to HashingRetriever')
        retriever = HashingRetriever()
    store     = _build_store(dataset, retriever)

    print(f"Knowledge store: {store.count} nodes")
    print(f"Running system evaluation (attacks: {args.attacks}) ...")
    sys.stdout.flush()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_system(
        dataset=dataset, retriever=retriever, knowledge_store=store,
        attacks=args.attacks, security_cfg=security_cfg, defense_cfg=defense_cfg,
        num_clients=args.num_clients, output_dir=out_dir, seed=args.seed,
    )

    print(f"\nRunning LLM generation with model={args.llm_model} ...")
    results = []
    sample_data = dataset[: args.num_samples or len(dataset)]
    for i, row in enumerate(sample_data):
        query        = row["query"]
        ground_truth = row["response"]
        emb_q = retriever.encode_context(query).tolist()
        if emb_q and isinstance(emb_q[0], list):
            emb_q = emb_q[0]
        nodes = store.retrieve([float(v) for v in emb_q], top_k=args.top_k)
        context_docs = [
            f"Q: {node.text_content}\nA: {node.metadata.get('answer', 'unknown')}"
            for _, node in nodes
        ] if nodes else [query]
        llm_answer   = ollama_generate(query, context_docs, args.llm_model)
        em = compute_em(llm_answer, ground_truth)
        f1 = compute_f1(llm_answer, ground_truth)
        bleu    = compute_bleu(llm_answer, ground_truth)
        sem_sim = semantic_sim(llm_answer, ground_truth)
        results.append({"query": query, "ground_truth": ground_truth,
                        "llm_answer": llm_answer, "em": em, "f1": f1,
                        "bleu": bleu, "semantic_similarity": sem_sim})
        print(f"  [{i+1}/{len(sample_data)}] EM={em:.0f}  F1={f1:.2f}  pred={llm_answer!r}", flush=True)

    avg_em   = sum(r["em"]  for r in results) / len(results)
    avg_f1   = sum(r["f1"]  for r in results) / len(results)
    avg_bleu = sum(r["bleu"] for r in results) / len(results)
    avg_sem  = sum(r["semantic_similarity"] for r in results) / len(results)
    summary = {"model": args.llm_model, "attacks": args.attacks,
               "num_queries": len(results),
               "avg_em": avg_em, "avg_f1": avg_f1,
               "avg_bleu": avg_bleu, "avg_semantic_similarity": avg_sem,
               "per_query": results}
    out_path = out_dir / "llm_generation_results.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== LLM RESULTS (model={args.llm_model}) ===")
    print(f"  EM            : {avg_em:.4f}  ({avg_em*100:.1f}%)")
    print(f"  F1            : {avg_f1:.4f}  ({avg_f1*100:.1f}%)")
    print(f"  BLEU          : {avg_bleu:.4f}")
    print(f"  Semantic Sim  : {avg_sem:.4f}")
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config-root",  default=".")
    p.add_argument("--llm",          default="llama32_3b",
                   choices=["llama32_3b","gemma2_2b","qwen25_3b"])
    p.add_argument("--llm-model",    default="mistral",
                   help="Ollama model tag, e.g. mistral / llama3.2:3b / gemma2:2b")
    p.add_argument("--dataset",      default="mmlu")
    p.add_argument("--num-samples",  type=int, default=20)
    p.add_argument("--num-examples", type=int, default=240)
    p.add_argument("--top-k",        type=int, default=3)
    p.add_argument("--num-clients",  type=int, default=10)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--attacks",      nargs="+",
                   default=["poisoning","node_availability"])
    p.add_argument("--output-dir",   default="logs/system_llm")
    args = p.parse_args()
    run_llm_system(args)
