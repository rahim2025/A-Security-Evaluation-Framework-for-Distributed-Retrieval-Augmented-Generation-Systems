#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, sys, requests
from pathlib import Path
from typing import Any
from fed_rag.evaluators import load_security_config, load_defense_config, run_system
from fed_rag.evaluators.metrics import HashingRetriever
from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore

OLLAMA_URL = "http://localhost:11434/api/generate"

try:
    from sentence_transformers import SentenceTransformer as _ST
    import numpy as _np
    _sim_model = _ST("sentence-transformers/all-MiniLM-L6-v2")
    def semantic_sim(a, b):
        e = _sim_model.encode([a, b])
        d = _np.linalg.norm(e[0]) * _np.linalg.norm(e[1])
        return float(_np.dot(e[0], e[1]) / d) if d else 0.0
except Exception:
    def semantic_sim(a, b): return 0.0

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    def compute_bleu(pred, gold):
        return sentence_bleu([gold.split()], pred.split(), smoothing_function=SmoothingFunction().method1)
except Exception:
    def compute_bleu(pred, gold): return 0.0

def compute_em(pred, gold):
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0

def compute_f1(pred, gold):
    p = set(pred.lower().split()); g = set(gold.lower().split())
    if not p or not g: return 0.0
    c = p & g
    if not c: return 0.0
    return 2 * (len(c)/len(p)) * (len(c)/len(g)) / (len(c)/len(p) + len(c)/len(g))

def load_news_dataset(num_samples, seed):
    from datasets import load_dataset
    print("Loading news dataset (heegyu/news-category-dataset) ...")
    ds = load_dataset("heegyu/news-category-dataset", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
    rows = []
    for item in ds:
        h = str(item.get("headline", "")).strip()
        if not h: continue
        rows.append({"query": h,
                     "response": str(item.get("short_description", "")).strip(),
                     "topic":    str(item.get("category", "NEWS")).strip()})
    print(f"Loaded news dataset: {len(rows)} samples")
    return rows

def _synthetic(n):
    topics = ["tech","sports","politics","health","finance","science","world","entertainment","business","weather"]
    return [{"query": f"Breaking: {topics[i%len(topics)].title()} update #{i}",
             "response": f"Story about {topics[i%len(topics)]} number {i}.",
             "topic": topics[i%len(topics)]} for i in range(n)]

def _build_store(examples, retriever):
    nodes = []
    for row in examples:
        emb = retriever.encode_context(row["query"]).tolist()
        if emb and isinstance(emb[0], list): emb = emb[0]
        nodes.append(KnowledgeNode(node_type=NodeType.TEXT, text_content=row["query"],
            embedding=[float(v) for v in emb],
            metadata={"answer": row["response"], "topic": row.get("topic","")}))
    return InMemoryKnowledgeStore.from_nodes(nodes)

def _split_clients(dataset, num_clients, seed):
    rng = random.Random(seed); data = list(dataset); rng.shuffle(data)
    chunks = [[] for _ in range(num_clients)]
    for i, row in enumerate(data): chunks[i % num_clients].append(row)
    return chunks

def ollama_generate(query, context_docs, model):
    context = context_docs[0] if context_docs else query
    prompt = (
        f"News headline: {query}\n\n"
        f"Related context: {context}\n\n"
        "Write ONE sentence (no lists, no preamble, no bullet points) "
        "summarising what this news story is about. "
        "Start directly with the summary sentence:"
    )
    try:
        r = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt,
            "stream": False, "options": {"temperature": 0.0, "num_predict": 80}}, timeout=90)
        raw = r.json().get("response", "").strip()
        return raw.split("\n")[0].strip()[:300]
    except Exception as e:
        return f"[ERROR: {e}]"

def run_news_llm_system(args):
    config_root = Path(args.config_root)
    security_cfg, _ = load_security_config(config_root, None)
    defense_cfg,  _ = load_defense_config(config_root, None)

    try:
        dataset = load_news_dataset(args.num_samples, args.seed)
    except Exception as e:
        print(f"News load failed ({e}), using synthetic data")
        dataset = _synthetic(args.num_samples)

    try:
        from fed_rag.retrievers import HFSentenceTransformerRetriever
        retriever = HFSentenceTransformerRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Using GPU retriever: all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Falling back to HashingRetriever ({e})")
        retriever = HashingRetriever()

    shards = _split_clients(dataset, args.num_clients, args.seed)
    print(f"\nFederated setup: {args.num_clients} clients")
    for cid, shard in enumerate(shards):
        print(f"  Client {cid:02d}: {len(shard)} samples | topics: {sorted(set(r['topic'] for r in shard))}")

    store = _build_store(dataset, retriever)
    print(f"\nKnowledge store: {store.count} nodes"); sys.stdout.flush()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning security evaluation — attacks: {args.attacks}")
    run_system(dataset=dataset, retriever=retriever, knowledge_store=store,
               attacks=args.attacks, security_cfg=security_cfg, defense_cfg=defense_cfg,
               num_clients=args.num_clients, output_dir=out_dir, seed=args.seed)

    print(f"\nRunning LLM generation with {args.llm_model} ...")
    results = []
    for i, row in enumerate(dataset):
        query, gt = row["query"], row["response"]
        emb_q = retriever.encode_context(query).tolist()
        if emb_q and isinstance(emb_q[0], list): emb_q = emb_q[0]
        nodes = store.retrieve([float(v) for v in emb_q], top_k=args.top_k)
        ctx = ([f"Headline: {n.text_content}\nSummary: {n.metadata.get('answer','')}" for _,n in nodes]
               if nodes else [f"Headline: {query}"])
        ans  = ollama_generate(query, ctx, args.llm_model)
        em   = compute_em(ans, gt)
        f1   = compute_f1(ans, gt)
        bleu = compute_bleu(ans, gt)
        sem  = semantic_sim(ans, gt)
        results.append({"query": query, "ground_truth": gt, "llm_answer": ans,
                        "em": em, "f1": f1, "bleu": bleu, "semantic_similarity": sem})
        print(f"  [{i+1:02d}/{len(dataset)}] F1={f1:.2f} SemSim={sem:.2f} pred={ans[:60]!r}", flush=True)

    n = len(results)
    avg = {k: sum(r[k] for r in results)/n for k in ["em","f1","bleu","semantic_similarity"]}
    summary = {"model": args.llm_model, "dataset": "news (heegyu/news-category-dataset)",
               "num_clients": args.num_clients, "attacks": args.attacks,
               "num_queries": n, "avg_em": avg["em"], "avg_f1": avg["f1"],
               "avg_bleu": avg["bleu"], "avg_semantic_similarity": avg["semantic_similarity"],
               "per_query": results}
    out_path = out_dir / "llm_generation_results.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*55}")
    print(f"=== RESULTS  model={args.llm_model}  clients={args.num_clients} ===")
    print(f"{'='*55}")
    print(f"  EM            : {avg['em']:.4f}  ({avg['em']*100:.1f}%)")
    print(f"  F1            : {avg['f1']:.4f}  ({avg['f1']*100:.1f}%)")
    print(f"  BLEU          : {avg['bleu']:.4f}")
    print(f"  Semantic Sim  : {avg['semantic_similarity']:.4f}  ({avg['semantic_similarity']*100:.1f}%)")
    print(f"\nSaved: {out_path}")
    print("NOTE: EM~0 is expected for news free-text. F1 + Semantic Sim are the key metrics.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config-root",  default=".")
    p.add_argument("--llm-model",    default="llama3.2:3b")
    p.add_argument("--num-samples",  type=int, default=20)
    p.add_argument("--num-clients",  type=int, default=20)
    p.add_argument("--top-k",        type=int, default=3)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--attacks",      nargs="+",
                   default=["poisoning","node_availability","extraction","membership_inference"],
                   choices=["poisoning","node_availability","extraction","membership_inference"])
    p.add_argument("--output-dir",   default="logs/news_llama_20clients_all_attacks")
    args = p.parse_args()
    run_news_llm_system(args)
