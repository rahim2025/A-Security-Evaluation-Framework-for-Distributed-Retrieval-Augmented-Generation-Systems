#!/usr/bin/env python3
"""Unified security-evaluation runner for FedRAG.

Reads ``config/security.yaml`` (attack toggles) and ``config/defense.yaml``
(defence toggles) and dispatches to either a **centralized** or
**federated** evaluator.

Centralized mode evaluates a single knowledge store with real models.
Federated mode simulates multiple IID clients with synthetic data.

Usage::

    # Centralized (default) — uses real sentence-transformer + dataset
    python run_security_evaluation.py --mode centralized \
        --dataset mmlu --llm llama32_3b --seed 0

    # Federated — simulates multiple clients
    python run_security_evaluation.py --mode federated \
        --num-clients 10 --seed 7

    # Enable node-availability attack (ported from DRAG)
    python run_security_evaluation.py --mode federated \
        --enable-node-availability --node-attack-type byzantine
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fed_rag.evaluators import (
    load_defense_config,
    load_security_config,
    run_centralized,
    run_federated,
)

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
        choices=["centralized", "federated"],
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

    # Centralized-mode args
    parser.add_argument(
        "--llm",
        choices=["llama32_3b", "gemma2_2b", "qwen25_3b"],
        default="llama32_3b",
        help="LLM config name under config/llm/ (centralized only).",
    )
    parser.add_argument(
        "--dataset",
        choices=["mmlu", "medical", "news"],
        default="mmlu",
        help="Dataset config name under config/data/ (centralized only).",
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

    # Federated-mode args
    parser.add_argument(
        "--num-clients", type=int, default=10, help="Federated only."
    )
    parser.add_argument(
        "--num-examples", type=int, default=240, help="Federated only."
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
        help="Optional JSONL (federated only).",
    )

    # Node availability (federated mode, or centralized with a store)
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
    else:
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


if __name__ == "__main__":
    main()
