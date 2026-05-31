#!/usr/bin/env python3
"""Unified security-evaluation runner for FedRAG.

Reads ``config/security.yaml`` (attack toggles) and ``config/defense.yaml``
(defence toggles) and dispatches to either a **centralized**, **federated**,
or **system-level** evaluator.

- *Centralized* evaluates a single knowledge store with real models.
- *Federated* simulates multiple IID clients with synthetic data.
- *System-level* mirrors DRAG's ``simulator.py`` flow:
  baseline → attack → post-attack → degradation report.

Usage::

    # Centralized (default) — uses real sentence-transformer + dataset
    python run_security_evaluation.py --mode centralized \
        --dataset mmlu --llm llama32_3b --seed 0

    # Federated — simulates multiple clients
    python run_security_evaluation.py --mode federated \
        --num-clients 10 --seed 7

    # System-level — baseline vs post-attack with DRAG-compatible metrics
    python run_security_evaluation.py --mode system \
        --num-clients 10 --attacks poisoning node_availability

    # Enable node-availability attack (ported from DRAG)
    python run_security_evaluation.py --mode federated \
        --enable-node-availability --node-attack-type byzantine
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from fed_rag.evaluators import (
    load_defense_config,
    load_security_config,
    run_centralized,
    run_federated,
    run_system,
)
from fed_rag.evaluators.centralized import (
    _load_drag_config,
    _load_drag_dataset,
)
from fed_rag.evaluators.metrics import HashingRetriever
from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore

try:
    from fed_rag.evaluators.metrics import HashingRetriever
    _REAL_RETRIEVER_AVAILABLE = True
except Exception:
    _REAL_RETRIEVER_AVAILABLE = False

DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedRAG unified security evaluation runner."
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=DEFAULT_CONFIG_ROOT,
        help="Repo root containing config/ files.",
    )
    parser.add_argument(
        "--mode",
        choices=["centralized", "federated", "system"],
        default="centralized",
        help="Evaluation mode (default: centralized).",
    )
    parser.add_argument(
        "--security-config",
        type=Path,
        default=None,
        help="Path to security YAML (default: config/security.yaml).",
    )
    parser.add_argument(
        "--defense-config",
        type=Path,
        default=None,
        help="Path to defence YAML (default: config/defense.yaml).",
    )

    # Centralized / system mode args
    parser.add_argument(
        "--llm",
        choices=["llama32_3b", "gemma2_2b", "qwen25_3b"],
        default="llama32_3b",
        help="LLM config name under config/llm/ (centralized / system only).",
    )
    parser.add_argument(
        "--dataset",
        choices=["mmlu", "medical", "news"],
        default="mmlu",
        help="Dataset config name under config/data/ (centralized / system only).",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--membership-threshold", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/security_evaluation"),
    )
    parser.add_argument(
        "--use-ollama-generation",
        action="store_true",
        help="Use Ollama LLM for QA generation (centralized only).",
    )
    parser.add_argument(
        "--max-generation-examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dataset-source",
        choices=["huggingface", "drag-log"],
        default="huggingface",
    )

    # Federated / system mode args
    parser.add_argument(
        "--num-clients", type=int, default=10, help="Federated / system only."
    )
    parser.add_argument(
        "--num-examples", type=int, default=240, help="Federated / system only."
    )
    parser.add_argument(
        "--malicious-clients", type=int, default=2, help="Federated only."
    )
    parser.add_argument(
        "--poisoning-ratio", type=float, default=0.35, help="Federated only."
    )
    parser.add_argument(
        "--extraction-top-k", type=int, default=2, help="Federated only."
    )
    parser.add_argument(
        "--rate-limit", type=int, default=18, help="Federated only."
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL (federated / system only).",
    )

    # System-mode args
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["poisoning", "node_availability"],
        choices=["poisoning", "node_availability", "extraction", "membership_inference"],
        help="Attacks to evaluate in system mode.",
    )

    # Node availability (federated / system mode)
    parser.add_argument(
        "--enable-node-availability",
        action="store_true",
        help="Enable node-availability attack evaluation.",
    )
    parser.add_argument(
        "--node-attack-type",
        choices=["node_removal", "byzantine", "partition", "ddos", "sybil"],
        default="node_removal",
    )
    parser.add_argument(
        "--node-attack-ratio", type=float, default=0.3
    )
    return parser.parse_args()


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

    if args.mode == "centralized":
        run_centralized(
            config_root=args.config_root,
            llm_name=args.llm,
            dataset_name=args.dataset,
            num_samples=args.num_samples,
            seed=args.seed,
            membership_threshold=args.membership_threshold,
            top_k=args.top_k,
            output_dir=args.output_dir,
            use_ollama=args.use_ollama_generation,
            max_generation_examples=args.max_generation_examples,
            dataset_source=args.dataset_source,
            security_cfg=security_cfg,
        )
    elif args.mode == "federated":
        run_federated(
            num_clients=args.num_clients,
            num_examples=args.num_examples,
            seed=args.seed,
            malicious_clients=args.malicious_clients,
            poisoning_ratio=args.poisoning_ratio,
            membership_threshold=args.membership_threshold,
            extraction_top_k=args.extraction_top_k,
            rate_limit=args.rate_limit,
            output_dir=args.output_dir,
            dataset_jsonl=args.dataset_jsonl,
            enable_node_availability=args.enable_node_availability,
            node_attack_type=args.node_attack_type,
            node_attack_ratio=args.node_attack_ratio,
            security_cfg=security_cfg,
            defense_cfg=defense_cfg,
        )
    else:
        # system mode
        output_dir = args.output_dir
        if output_dir == Path("logs/security_evaluation"):
            output_dir = Path("logs/system_evaluation")

        rng = random.Random(args.seed)

        if args.dataset_jsonl:
            dataset = _load_jsonl_dataset(args.dataset_jsonl)
        elif args.mode == "system":
            # Build dataset from DRAG config or synthetic
            try:
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
            except Exception:
                dataset = _make_synthetic_dataset(args.num_examples)
        else:
            dataset = _make_synthetic_dataset(args.num_examples)

        # In system mode, if num_clients > 1, simulate IID split
        if args.num_clients > 1:
            clients = _iid_split(dataset, args.num_clients, rng)
            dataset = []
            for cid, client in enumerate(clients):
                for row in client:
                    row = dict(row)
                    row["client_id"] = cid
                    dataset.append(row)

        retriever = (
            HashingRetriever()
            if _REAL_RETRIEVER_AVAILABLE
            else HashingRetriever()
        )
        store = _build_store(dataset, retriever)

        print(f"Dataset size: {len(dataset)}")
        print(f"Knowledge store nodes: {store.count}")
        import sys
        sys.stdout.flush()

        run_system(
            dataset=dataset,
            retriever=retriever,
            knowledge_store=store,
            attacks=args.attacks,
            security_cfg=security_cfg,
            defense_cfg=defense_cfg,
            num_clients=args.num_clients,
            output_dir=output_dir,
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
        # Handle both 1-D (HashingRetriever) and 2-D (HF retriever) embeddings
        if emb and isinstance(emb[0], list):
            emb = emb[0]
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


if __name__ == "__main__":
    main()
