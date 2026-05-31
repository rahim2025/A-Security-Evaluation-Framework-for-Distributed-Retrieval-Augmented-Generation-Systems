#!/usr/bin/env python3
"""CLI wrapper for the training-based federated security evaluator.

Thin wrapper around ``fed_rag.evaluators.run_training_federated``.  All
logic lives in the evaluator module; this script only parses arguments and
invokes the runner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fed_rag.evaluators import (
    load_mmlu_dataset,
    make_synthetic_dataset,
    run_training_federated,
)
from fed_rag.evaluators.config import DefenseConfig, SecurityConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated training evaluation with client-level attacks."
    )
    parser.add_argument(
        "--num-clients", type=int, default=5,
        help="Number of simulated clients.",
    )
    parser.add_argument(
        "--num-rounds", type=int, default=3,
        help="Number of federated aggregation rounds.",
    )
    parser.add_argument(
        "--local-epochs", type=int, default=1,
        help="Local training epochs per client per round.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for local training.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base SentenceTransformer model name.",
    )

    # Attack toggles
    parser.add_argument(
        "--attack",
        choices=[
            "none",
            "data_poisoning",
            "gradient_flip",
            "gradient_noise",
            "gradient_scale",
        ],
        default="none",
        help="Training-time attack applied on malicious clients.",
    )
    parser.add_argument(
        "--malicious-clients", type=int, default=1,
        help="Number of malicious clients.",
    )
    parser.add_argument(
        "--poisoning-ratio", type=float, default=0.35,
        help="Data-poisoning ratio per malicious client.",
    )
    parser.add_argument(
        "--noise-std", type=float, default=0.5,
        help="Std-dev for gradient_noise attack.",
    )
    parser.add_argument(
        "--scale", type=float, default=2.0,
        help="Multiplier for gradient_scale attack.",
    )

    # Data
    parser.add_argument(
        "--dataset",
        choices=["mmlu", "synthetic"],
        default="synthetic",
        help="Dataset to use.",
    )
    parser.add_argument(
        "--num-examples", type=int, default=120,
        help="Number of examples to load (synthetic or MMLU).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/federated_training_evaluation"),
        help="Directory for output files.",
    )

    # Defense
    parser.add_argument(
        "--enable-defense",
        action="store_true",
        help="Enable post-training defense evaluation.",
    )
    parser.add_argument(
        "--defense-quarantine-threshold",
        type=float,
        default=0.8,
        help="Quarantine threshold for client data poisoning defense.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load dataset
    if args.dataset == "mmlu":
        dataset = load_mmlu_dataset(args.num_examples)
    else:
        dataset = make_synthetic_dataset(args.num_examples)

    # Build configs
    security_cfg = SecurityConfig(
        enable_attack=args.attack != "none",
        poisoning_ratio=args.poisoning_ratio,
        poison_type="answer_swap",
    )
    defense_cfg = DefenseConfig(
        enabled=args.enable_defense,
        client_data_poisoning_defense_enabled=args.enable_defense,
        client_data_poisoning_defense_quarantine_threshold=args.defense_quarantine_threshold,
    )

    run_training_federated(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        model_name=args.model_name,
        dataset=dataset,
        output_dir=args.output_dir,
        security_cfg=security_cfg,
        defense_cfg=defense_cfg,
        attack=args.attack,
        malicious_clients=args.malicious_clients,
        poisoning_ratio=args.poisoning_ratio,
        noise_std=args.noise_std,
        scale=args.scale,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
