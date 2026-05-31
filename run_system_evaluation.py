#!/usr/bin/env python3
"""System-level security evaluation runner for FedRAG.

Mirrors DRAG's ``simulator.py`` flow:

    baseline evaluation → apply attack(s) → post-attack evaluation
    → compute degradation → write matrix + JSON + terminal summary

Usage::

    # Centralized mode (single knowledge store, real retriever)
    python run_system_evaluation.py \
        --mode centralized \
        --dataset mmlu \
        --llm llama32_3b \
        --attacks poisoning node_availability \
        --output-dir logs/system_evaluation

    # Federated mode (multi-client simulation)
    python run_system_evaluation.py \
        --mode federated \
        --num-clients 10 \
        --attacks poisoning node_availability extraction \
        --output-dir logs/system_evaluation

Both modes produce the **same evaluation matrix** so that DRAG and FedRAG
results can be compared directly in a paper.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from fed_rag.evaluators import (
    load_defense_config,
    load_security_config,
    run_system,
)
from fed_rag.evaluators.centralized import (
    _load_drag_config,
    _load_drag_dataset,
)
from fed_rag.evaluators.metrics import HashingRetriever
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType

# ------------------------------------------------------------------
# Optional real retriever
# ------------------------------------------------------------------

try:
    from fed_rag.retrievers.huggingface import HFSentenceTransformerRetriever
    _REAL_RETRIEVER_AVAILABLE = True
except Exception:
    _REAL_RETRIEVER_AVAILABLE = False


DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedRAG system-level security evaluation runner."
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=DEFAULT_CONFIG_ROOT,
        help="Repo root containing config/ files.",
    )
    parser.add_argument(
        "--mode",
        choices=["centralized", "federated"],
        default="centralized",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--security-config",
        type=Path,
        default=None,
        help="Path to security YAML.",
    )
    parser.add_argument(
        "--defense-config",
        type=Path,
        default=None,
        help="Path to defence YAML.",
    )

    # Dataset / model (centralized mode)
    parser.add_argument(
        "--llm",
        choices=["llama32_3b", "gemma2_2b", "qwen25_3b"],
        default="llama32_3b",
    )
    parser.add_argument(
        "--dataset",
        choices=["mmlu", "medical", "news"],
        default="mmlu",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset-source",
        choices=["huggingface", "drag-log"],
        default="huggingface",
    )

    # Federated-mode args
    parser.add_argument(
        "--num-clients", type=int, default=10, help="Federated only."
    )
    parser.add_argument(
        "--num-examples", type=int, default=240, help="Federated only."
    )
    parser.add_argument(
        "--dataset-jsonl", type=Path, default=None, help="Federated only."
    )

    # Attack selection
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["poisoning", "node_availability"],
        choices=["poisoning", "node_availability", "extraction", "membership_inference"],
        help="Which attacks to evaluate.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/system_evaluation"),
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    security_cfg, security_path = load_security_config(
        args.config_root, args.security_config
    )
    defense_cfg, defense_path = load_defense_config(
        args.config_root, args.defense_config
    )
    print(f"Loaded security config: {security_path}")
    print(f"Loaded defence config:  {defense_path}")
    print(f"Mode: {args.mode}")
    print(f"Attacks: {args.attacks}")
    sys.stdout.flush()

    rng = random.Random(args.seed)

    # ================================================================
    # Build dataset + knowledge store
    # ================================================================
    if args.mode == "centralized":
        drag_config = _load_drag_config(
            args.config_root, args.llm, args.dataset
        )
        dataset = _load_drag_dataset(
            args.config_root,
            drag_config,
            args.num_samples,
            args.seed,
            args.dataset_source,
        )
        retriever = (
            HFSentenceTransformerRetriever(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            if _REAL_RETRIEVER_AVAILABLE
            else HashingRetriever()
        )
        store = _build_store(dataset, retriever)
    else:
        # Federated mode — synthetic IID split
        dataset = (
            _load_jsonl_dataset(args.dataset_jsonl)
            if args.dataset_jsonl
            else _make_synthetic_dataset(args.num_examples)
        )
        clients = _iid_split(dataset, args.num_clients, rng)
        # Flatten with client_id
        dataset = []
        for cid, client in enumerate(clients):
            for row in client:
                row["client_id"] = cid
                dataset.append(row)
        retriever = HashingRetriever()
        store = _build_store(dataset, retriever)

    print(f"Dataset size: {len(dataset)}")
    print(f"Knowledge store nodes: {store.count}")
    sys.stdout.flush()

    # ================================================================
    # Run system-level evaluation
    # ================================================================
    run_system(
        dataset=dataset,
        retriever=retriever,
        knowledge_store=store,
        attacks=args.attacks,
        security_cfg=security_cfg,
        defense_cfg=defense_cfg,
        num_clients=args.num_clients if args.mode == "federated" else 1,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print("\nDone.")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_store(
    examples: list[dict[str, Any]], retriever: Any
) -> InMemoryKnowledgeStore:
    nodes = []
    for row in examples:
        emb = retriever.encode_context(row["query"]).tolist()
        nodes.append(
            KnowledgeNode(
                node_type=NodeType.TEXT,
                text_content=row["query"],
                embedding=[float(v) for v in emb],
                metadata={
                    "topic": row.get("topic", "unknown"),
                    "answer": row["response"],
                    "client_id": row.get("client_id"),
                },
            )
        )
    return InMemoryKnowledgeStore.from_nodes(nodes)


def _make_synthetic_dataset(num_examples: int) -> list[dict[str, str]]:
    topics = [
        "cryptography",
        "medicine",
        "finance",
        "biology",
        "law",
        "networking",
        "history",
        "physics",
        "privacy",
        "safety",
        "databases",
        "governance",
    ]
    rows = []
    for idx in range(num_examples):
        topic = topics[idx % len(topics)]
        rows.append(
            {
                "query": (
                    f"Fact {idx}: what is the verified answer for {topic}?"
                ),
                "response": f"verified-{topic}-answer-{idx}",
                "topic": topic,
            }
        )
    return rows


def _iid_split(
    dataset: list[dict[str, str]],
    num_clients: int,
    rng: random.Random,
) -> list[list[dict[str, str]]]:
    shuffled = [dict(row) for row in dataset]
    rng.shuffle(shuffled)
    base, remainder = divmod(len(shuffled), num_clients)
    clients = []
    offset = 0
    for client_id in range(num_clients):
        size = base + (1 if client_id < remainder else 0)
        clients.append(shuffled[offset : offset + size])
        offset += size
    return clients


def _load_jsonl_dataset(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(
                {
                    "query": str(row["query"]),
                    "response": str(row["response"]),
                    "topic": str(row.get("topic", "unknown")),
                }
            )
    return rows


import json
import sys

if __name__ == "__main__":
    main()
