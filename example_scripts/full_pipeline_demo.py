#!/usr/bin/env python3
"""
full_pipeline_demo.py
=====================
Demonstrates the full FedRAG Security Evaluation Framework API.

No ML dependencies required — uses hash-based embeddings from FedRAGSimulator.

Run from the repository root:
    python example_scripts/full_pipeline_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from any directory
repo_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from security_framework import FedRAGSimulator, SecurityPipeline, SecurityReport


def demo_full_pipeline() -> None:
    print("\n" + "=" * 60)
    print("  Demo 1: Full pipeline — all attacks + all defenses")
    print("=" * 60)

    sim = FedRAGSimulator(num_clients=10, num_examples=200, seed=0)
    pipe = SecurityPipeline(sim, defense_enabled=True)
    report = pipe.run_all()

    report.print_summary()
    report.save("results/demo_full")
    print("  Saved to results/demo_full/\n")


def demo_no_defense() -> None:
    print("\n" + "=" * 60)
    print("  Demo 2: Attacks only — no defenses (raw effectiveness)")
    print("=" * 60)

    sim = FedRAGSimulator(num_clients=8, num_examples=160, seed=1)
    pipe = SecurityPipeline(sim, defense_enabled=False, poison_type="answer_swap")
    report = pipe.run_all()

    report.print_summary()
    report.save("results/demo_no_defense")
    print("  Saved to results/demo_no_defense/\n")


def demo_single_attack() -> None:
    print("\n" + "=" * 60)
    print("  Demo 3: Run a single attack programmatically")
    print("=" * 60)

    sim = FedRAGSimulator(num_clients=6, num_examples=120, seed=2)
    pipe = SecurityPipeline(sim, membership_threshold=0.75)

    result = pipe.run_attack("membership_inference")
    print(json.dumps(result, indent=2))
    print()


def demo_custom_dataset() -> None:
    """Show how to plug in your own JSONL dataset."""
    print("\n" + "=" * 60)
    print("  Demo 4: Custom synthetic JSONL dataset")
    print("=" * 60)

    # Write a tiny JSONL dataset to /tmp
    import tempfile, os

    rows = [
        {"query": f"What is concept {i}?", "response": f"answer-{i}", "topic": f"topic-{i % 5}"}
        for i in range(60)
    ]
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for row in rows:
        tmp.write(json.dumps(row) + "\n")
    tmp.close()

    try:
        sim = FedRAGSimulator(num_clients=5, dataset_jsonl=tmp.name, seed=3)
        pipe = SecurityPipeline(sim, defense_enabled=True)
        report = pipe.run_all()
        report.print_summary()
        report.save("results/demo_custom_dataset")
        print("  Saved to results/demo_custom_dataset/\n")
    finally:
        os.unlink(tmp.name)


def demo_node_attack_variants() -> None:
    print("\n" + "=" * 60)
    print("  Demo 5: Node availability — all attack types")
    print("=" * 60)

    for attack_type in ["node_removal", "byzantine", "partition", "ddos", "sybil"]:
        sim = FedRAGSimulator(num_clients=10, num_examples=100, seed=0)
        pipe = SecurityPipeline(
            sim,
            node_attack_type=attack_type,
            node_attack_ratio=0.3,
            defense_enabled=True,
        )
        result = pipe.run_attack("node_availability")
        a = result["attacked"]
        b = result.get("baseline", {})
        d = result.get("defended") or {}
        print(
            f"  {attack_type:15s} | "
            f"baseline F1={b.get('f1', 0):.3f} | "
            f"attacked F1={a.get('f1', 0):.3f} | "
            f"defended F1={d.get('f1', 0):.3f}"
        )
    print()


if __name__ == "__main__":
    demo_full_pipeline()
    demo_no_defense()
    demo_single_attack()
    demo_custom_dataset()
    demo_node_attack_variants()

    print("All demos complete. Check results/ for output files.\n")
