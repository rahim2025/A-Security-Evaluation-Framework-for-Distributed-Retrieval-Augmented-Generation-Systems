"""System-level security evaluation for FedRAG.

Mirrors DRAG's ``simulator.py`` flow:

    baseline evaluation → apply attack(s) → post-attack evaluation
    → compute degradation → write matrix + JSON + terminal summary

Produces architecture-agnostic answer-quality metrics (EM, precision,
recall, F1, BLEU, ROUGE, semantic similarity, edit distance, n-gram
overlap, query-failure rate) so that DRAG and FedRAG can be compared
in a unified evaluation matrix.

**Scope:** This evaluator operates on a single pooled knowledge store
and measures how answer quality degrades when the store is corrupted or
clients are dropped.  It does NOT model federated learning (no local
client training, no FedAvg, no Flower rounds).
"""

from __future__ import annotations

import csv
import json
import resource
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fed_rag.attacks import (
    DataPoisoningAttack,
    KnowledgeExtractionAttack,
    MembershipInferenceAttack,
    NodeAvailabilityAttack,
)
from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType
from fed_rag.evaluators.config import DefenseConfig, SecurityConfig
from fed_rag.evaluators.metrics import (
    HashingRetriever,
    evaluate_system_rag_answers,
    semantic_similarity,
    token_precision,
    token_recall,
    token_f1,
    simple_bleu,
    jaccard,
    exact_match,
    rouge_scores,
    edit_distance_metrics,
    ngram_overlap,
    compute_all_metrics,
)
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.utils.evaluation_matrix import (
    write_evaluation_matrix,
    write_json_results,
)


# ------------------------------------------------------------------
# Dataclasses for system-level test case tracking
# ------------------------------------------------------------------


@dataclass
class SystemTestcase:
    """Single query result mirroring DRAG's ``Testcase``."""

    question: str
    expected_output: str
    actual_output: str
    relevant_knowledge: str = ""
    relevant_score: float = 0.0
    num_hops: int = 0
    num_messages: int = 0
    is_query_hit: bool = False


@dataclass
class SystemEvaluationResult:
    """Aggregated system evaluation result."""

    phase: str  # "baseline" | "post_attack" | "defended"
    metrics: dict[str, float] = field(default_factory=dict)
    testcases: list[SystemTestcase] = field(default_factory=list)


# ------------------------------------------------------------------
# Public runner
# ------------------------------------------------------------------


def run(
    *,
    dataset: list[dict[str, Any]],
    retriever: Any,
    knowledge_store: InMemoryKnowledgeStore,
    attacks: list[str],
    security_cfg: SecurityConfig,
    defense_cfg: DefenseConfig | None = None,
    num_clients: int = 1,
    output_dir: Path,
    seed: int = 0,
) -> dict[str, Any]:
    """Run full system-level evaluation.

    Args:
        dataset: list of {"query", "response", "topic", "client_id"?} dicts
        retriever: a ``BaseRetriever`` instance
        knowledge_store: the clean (un-poisoned) knowledge store
        attacks: which attacks to evaluate, e.g.
                 ["poisoning", "node_availability"]
        security_cfg: parsed ``config/security.yaml``
        defense_cfg: parsed ``config/defense.yaml`` (optional)
        num_clients: number of simulated clients (for federated mode)
        output_dir: where to write logs
        seed: random seed

    Returns:
        dict with raw results for programmatic access.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    defense_cfg = defense_cfg or DefenseConfig()

    # ================================================================
    # 1. BASELINE EVALUATION
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: BASELINE SYSTEM EVALUATION")
    print("=" * 70)
    sys.stdout.flush()

    baseline_result = _evaluate_system(
        dataset=dataset,
        retriever=retriever,
        knowledge_store=knowledge_store,
        phase="baseline",
    )
    baseline_metrics = baseline_result.metrics
    _print_metrics(baseline_metrics, label="BASELINE")

    # ================================================================
    # 2. ATTACK(S) + POST-ATTACK EVALUATION
    # ================================================================
    all_records: list[dict[str, Any]] = []
    attack_results: dict[str, Any] = {}

    for attack_name in attacks:
        print("\n" + "=" * 70)
        print(f"ATTACK: {attack_name.upper()}")
        print("=" * 70)
        sys.stdout.flush()

        # -- run attack --
        attacked_store, attack_meta = _run_attack(
            attack_name=attack_name,
            dataset=dataset,
            knowledge_store=knowledge_store,
            retriever=retriever,
            security_cfg=security_cfg,
            num_clients=num_clients,
            seed=seed,
        )
        attack_results[attack_name] = attack_meta

        # -- post-attack evaluation --
        print(f"\n--- Post-Attack Evaluation ({attack_name}) ---")
        sys.stdout.flush()
        post_result = _evaluate_system(
            dataset=dataset,
            retriever=retriever,
            knowledge_store=attacked_store,
            phase="post_attack",
            attack_meta=attack_meta,
        )
        post_metrics = post_result.metrics
        _print_metrics(post_metrics, label="POST-ATTACK")

        # -- defended evaluation (if applicable) --
        defended_metrics: dict[str, float] = {}
        if defense_cfg.enabled:
            print(f"\n--- Defended Evaluation ({attack_name}) ---")
            sys.stdout.flush()
            defended_store = _apply_defense(
                attack_name=attack_name,
                attacked_store=attacked_store,
                clean_store=knowledge_store,
                dataset=dataset,
                defense_cfg=defense_cfg,
                attack_meta=attack_meta,
            )
            if defended_store is not None:
                defended_result = _evaluate_system(
                    dataset=dataset,
                    retriever=retriever,
                    knowledge_store=defended_store,
                    phase="defended",
                    attack_meta=attack_meta,
                )
                defended_metrics = defended_result.metrics
                _print_metrics(defended_metrics, label="DEFENDED")
            else:
                defended_metrics = dict(post_metrics)

        # -- compute degradation --
        comparison = _compute_comparison(
            baseline=baseline_metrics,
            post_attack=post_metrics,
            defended=defended_metrics,
        )
        _print_comparison(comparison)

        # -- build matrix record --
        record = _build_matrix_record(
            attack=attack_name,
            baseline=baseline_metrics,
            post_attack=post_metrics,
            defended=defended_metrics,
            comparison=comparison,
            attack_meta=attack_meta,
            security_cfg=security_cfg,
            defense_cfg=defense_cfg,
        )
        all_records.append(record)

    # ================================================================
    # 3. WRITE OUTPUTS
    # ================================================================
    raw_results = {
        "baseline": baseline_result.metrics,
        "attacks": attack_results,
        "records": all_records,
        "seed": seed,
        "num_clients": num_clients,
        "dataset_size": len(dataset),
    }

    write_json_results(raw_results, output_dir / "system_evaluation.json")
    write_evaluation_matrix(
        all_records,
        markdown_path=output_dir / "SYSTEM_EVALUATION_MATRIX.md",
        csv_path=output_dir / "system_evaluation.csv",
    )

    # -- terminal summary --
    _print_terminal_summary(
        baseline=baseline_metrics,
        records=all_records,
        output_dir=output_dir,
    )

    return raw_results


# ------------------------------------------------------------------
# System evaluation helper
# ------------------------------------------------------------------


def _evaluate_system(
    *,
    dataset: list[dict[str, Any]],
    retriever: Any,
    knowledge_store: InMemoryKnowledgeStore,
    phase: str,
    attack_meta: dict[str, Any] | None = None,
) -> SystemEvaluationResult:
    """Evaluate the full system on *dataset* using *knowledge_store*."""
    dropped: set[int] = set()
    byzantine: set[int] = set()
    if attack_meta:
        dropped = set(attack_meta.get("dropped_clients", []))
        byzantine = set(attack_meta.get("byzantine_clients", []))

    result = SystemEvaluationResult(phase=phase)
    for row in dataset:
        query = row["query"]
        expected = row["response"]
        client_id = row.get("client_id")

        # Simulate query through the system
        if dropped and client_id in dropped:
            actual = "QUERY_FAILED_NODE_UNAVAILABLE"
            knowledge_text = ""
            score = 0.0
        else:
            retrieved = knowledge_store.retrieve(
                query_emb=retriever.encode_query(query).tolist(),
                top_k=1,
            )
            if retrieved:
                _score, node = retrieved[0]
                actual = str(node.metadata.get("answer", ""))
                knowledge_text = node.text_content
                score = float(_score) if isinstance(_score, (int, float)) else 0.0
            else:
                actual = ""
                knowledge_text = ""
                score = 0.0

        if byzantine and client_id in byzantine:
            actual = "INCORRECT_BYZANTINE_RESPONSE"
            score = 0.0

        result.testcases.append(
            SystemTestcase(
                question=query,
                expected_output=expected,
                actual_output=actual,
                relevant_knowledge=knowledge_text,
                relevant_score=score,
            )
        )

    # Aggregate metrics exactly like DRAG's QAEvaluator
    metrics: dict[str, list[float]] = {
        "exact_match": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "bleu": [],
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "semantic_similarity": [],
        "jaccard": [],
        "edit_distance": [],
        "normalized_edit_distance": [],
        "bigram_overlap": [],
        "trigram_overlap": [],
    }
    failed_queries = 0

    for tc in result.testcases:
        if tc.actual_output == "QUERY_FAILED_NODE_UNAVAILABLE":
            failed_queries += 1
        allm = compute_all_metrics(tc.actual_output, tc.expected_output)
        for k in metrics:
            if k in allm:
                metrics[k].append(allm[k])

    result.metrics = {
        k: sum(v) / max(1, len(v)) for k, v in metrics.items()
    }
    result.metrics["query_failure_rate"] = (
        failed_queries / len(dataset) if dataset else 0.0
    )
    result.metrics["successful_queries"] = len(dataset) - failed_queries
    result.metrics["failed_queries"] = failed_queries
    return result


# ------------------------------------------------------------------
# Attack dispatchers
# ------------------------------------------------------------------


def _run_attack(
    *,
    attack_name: str,
    dataset: list[dict[str, Any]],
    knowledge_store: InMemoryKnowledgeStore,
    retriever: Any,
    security_cfg: SecurityConfig,
    num_clients: int,
    seed: int,
) -> tuple[InMemoryKnowledgeStore, dict[str, Any]]:
    """Execute *attack_name* and return the attacked store + metadata."""
    if attack_name == "poisoning":
        return _run_poisoning_attack(
            dataset=dataset,
            knowledge_store=knowledge_store,
            retriever=retriever,
            security_cfg=security_cfg,
            seed=seed,
        )
    if attack_name == "node_availability":
        return _run_node_availability_attack(
            dataset=dataset,
            knowledge_store=knowledge_store,
            num_clients=num_clients,
            security_cfg=security_cfg,
            seed=seed,
        )
    if attack_name == "extraction":
        return _run_extraction_attack(
            dataset=dataset,
            knowledge_store=knowledge_store,
            retriever=retriever,
            security_cfg=security_cfg,
            seed=seed,
        )
    if attack_name == "membership_inference":
        return _run_membership_inference_attack(
            dataset=dataset,
            knowledge_store=knowledge_store,
            retriever=retriever,
            security_cfg=security_cfg,
            seed=seed,
        )
    raise ValueError(f"Unknown attack: {attack_name}")


def _run_poisoning_attack(
    *,
    dataset: list[dict[str, Any]],
    knowledge_store: InMemoryKnowledgeStore,
    retriever: Any,
    security_cfg: SecurityConfig,
    seed: int,
) -> tuple[InMemoryKnowledgeStore, dict[str, Any]]:
    start = time.perf_counter()
    attack = DataPoisoningAttack(
        poisoning_ratio=security_cfg.poisoning_ratio,
        poison_type=security_cfg.poison_type,
        mode="replace",
        amplification_factor=security_cfg.amplification_factor,
        question_variants=security_cfg.question_variants,
        seed=seed,
    )
    poisoned = attack.execute(dataset)
    runtime = time.perf_counter() - start
    poisoned_store = _build_store(poisoned.poisoned_examples, retriever)
    return poisoned_store, {
        "attack": "poisoning",
        "runtime": runtime,
        "num_poisoned": poisoned.num_poisoned,
        "dropped_clients": [],
        "byzantine_clients": [],
    }


def _run_node_availability_attack(
    *,
    dataset: list[dict[str, Any]],
    knowledge_store: InMemoryKnowledgeStore,
    num_clients: int,
    security_cfg: SecurityConfig,
    seed: int,
) -> tuple[InMemoryKnowledgeStore, dict[str, Any]]:
    nodes = [
        {"client_id": i, "data_size": 1} for i in range(num_clients)
    ]
    attack = NodeAvailabilityAttack(
        attack_type=security_cfg.node_attack_type,
        attack_ratio=security_cfg.node_attack_ratio,
        seed=seed,
    )
    start = time.perf_counter()
    result = attack.execute(nodes, strategy="random")
    runtime = time.perf_counter() - start

    dropped: set[int] = set()
    byzantine: set[int] = set()
    if security_cfg.node_attack_type in ("node_removal", "ddos"):
        dropped = set(result.affected_nodes)
    elif security_cfg.node_attack_type == "byzantine":
        byzantine = set(result.affected_nodes)
    elif security_cfg.node_attack_type == "partition":
        if result.iterations:
            dropped = set(result.iterations[0].get("partition_2", []))

    return knowledge_store, {
        "attack": "node_availability",
        "runtime": runtime,
        "node_attack_type": security_cfg.node_attack_type,
        "node_attack_ratio": security_cfg.node_attack_ratio,
        "affected_nodes": result.affected_nodes,
        "availability_before": result.availability_before,
        "availability_after": result.availability_after,
        "dropped_clients": sorted(dropped),
        "byzantine_clients": sorted(byzantine),
    }


def _run_extraction_attack(
    *,
    dataset: list[dict[str, Any]],
    knowledge_store: InMemoryKnowledgeStore,
    retriever: Any,
    security_cfg: SecurityConfig,
    seed: int,
) -> tuple[InMemoryKnowledgeStore, dict[str, Any]]:
    topics = sorted({row["topic"] for row in dataset})
    extraction = KnowledgeExtractionAttack(top_k=3)
    queries = [row["query"] for row in dataset] + extraction.generate_queries(
        topics
    )
    start = time.perf_counter()
    result = extraction.execute(
        retriever=retriever,
        knowledge_store=knowledge_store,
        queries=queries,
        total_nodes=knowledge_store.count,
    )
    runtime = time.perf_counter() - start
    # Extraction doesn't modify the store
    return knowledge_store, {
        "attack": "extraction",
        "runtime": runtime,
        "recovery_ratio": result.recovery_ratio,
        "total_queries": result.total_queries,
        "kb_recovery_pct": result.recovery_ratio * 100.0,
        "dropped_clients": [],
        "byzantine_clients": [],
    }


def _run_membership_inference_attack(
    *,
    dataset: list[dict[str, Any]],
    knowledge_store: InMemoryKnowledgeStore,
    retriever: Any,
    security_cfg: SecurityConfig,
    seed: int,
) -> tuple[InMemoryKnowledgeStore, dict[str, Any]]:
    member_queries = [
        row["query"] for row in dataset[: max(1, len(dataset) // 2)]
    ]
    non_member_queries = [
        f"Out-of-distribution probe {idx}: unrelated private record"
        for idx in range(max(1, len(member_queries)))
    ]
    start = time.perf_counter()
    mia = MembershipInferenceAttack(
        threshold=security_cfg.membership_threshold, top_k=1
    ).execute(
        retriever=retriever,
        knowledge_store=knowledge_store,
        member_queries=member_queries,
        non_member_queries=non_member_queries,
    )
    runtime = time.perf_counter() - start
    return knowledge_store, {
        "attack": "membership_inference",
        "runtime": runtime,
        "attack_accuracy": mia.attack_accuracy,
        "num_members_tested": mia.num_members_tested,
        "num_non_members_tested": mia.num_non_members_tested,
        "dropped_clients": [],
        "byzantine_clients": [],
    }


# ------------------------------------------------------------------
# Defense helper
# ------------------------------------------------------------------


def _apply_defense(
    *,
    attack_name: str,
    attacked_store: InMemoryKnowledgeStore,
    clean_store: InMemoryKnowledgeStore,
    dataset: list[dict[str, Any]],
    defense_cfg: DefenseConfig,
    attack_meta: dict[str, Any],
) -> InMemoryKnowledgeStore | None:
    """Apply the best defence for *attack_name* and return the defended store."""
    # For now, defence mainly affects node-availability (cross-peer validation)
    # and extraction (rate limiting / anomaly detection).  Poisoning defence
    # would require rebuilding the store from sanitised client data, which is
    # handled in the federated evaluator.  Here we return ``None`` to signal
    # that no store-level defence was applied.
    if attack_name == "node_availability" and defense_cfg.cross_peer_validation_enabled:
        # Cross-peer validation doesn't rebuild the store; it validates answers
        # at query time.  Return ``None`` so the runner falls back to post-attack.
        return None
    return None


# ------------------------------------------------------------------
# Store builder
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


# ------------------------------------------------------------------
# Comparison / degradation
# ------------------------------------------------------------------


def _compute_comparison(
    *,
    baseline: dict[str, float],
    post_attack: dict[str, float],
    defended: dict[str, float],
) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    for metric in [
        "exact_match",
        "precision",
        "recall",
        "f1",
        "bleu",
        "rouge1",
        "rouge2",
        "rougeL",
        "semantic_similarity",
        "jaccard",
        "query_failure_rate",
    ]:
        b = baseline.get(metric, 0.0)
        p = post_attack.get(metric, 0.0)
        d = defended.get(metric, 0.0)

        if metric == "query_failure_rate":
            degradation = p - b
        else:
            degradation = b - p
        degradation_pct = (
            (degradation / (b + 1e-4)) * 100 if b != 0 else 0.0
        )

        comparison[metric] = {
            "baseline": b,
            "post_attack": p,
            "defended": d,
            "degradation": degradation,
            "degradation_pct": degradation_pct,
        }
    return comparison


# ------------------------------------------------------------------
# Matrix record builder
# ------------------------------------------------------------------


def _build_matrix_record(
    *,
    attack: str,
    baseline: dict[str, float],
    post_attack: dict[str, float],
    defended: dict[str, float],
    comparison: dict[str, dict[str, float]],
    attack_meta: dict[str, Any],
    security_cfg: SecurityConfig,
    defense_cfg: DefenseConfig,
) -> dict[str, Any]:
    comp = comparison
    return {
        "Attack": attack,
        "System": "FedRAG",
        "Dataset": "synthetic-client-rag",
        "Model": "system-eval",
        "Baseline BLEU": baseline.get("bleu"),
        "Post-Attack BLEU": post_attack.get("bleu"),
        "Defended BLEU": defended.get("bleu"),
        "Delta BLEU": post_attack.get("bleu", 0.0) - baseline.get("bleu", 0.0),
        "Baseline EM": baseline.get("exact_match"),
        "Post-Attack EM": post_attack.get("exact_match"),
        "Defended EM": defended.get("exact_match"),
        "Delta EM": post_attack.get("exact_match", 0.0) - baseline.get("exact_match", 0.0),
        "Baseline F1": baseline.get("f1"),
        "Post-Attack F1": post_attack.get("f1"),
        "Defended F1": defended.get("f1"),
        "Delta F1": post_attack.get("f1", 0.0) - baseline.get("f1", 0.0),
        "Baseline Precision": baseline.get("precision"),
        "Post-Attack Precision": post_attack.get("precision"),
        "Defended Precision": defended.get("precision"),
        "Delta Precision": post_attack.get("precision", 0.0) - baseline.get("precision", 0.0),
        "Baseline Recall": baseline.get("recall"),
        "Post-Attack Recall": post_attack.get("recall"),
        "Defended Recall": defended.get("recall"),
        "Delta Recall": post_attack.get("recall", 0.0) - baseline.get("recall", 0.0),
        "Baseline Rouge1": baseline.get("rouge1"),
        "Post-Attack Rouge1": post_attack.get("rouge1"),
        "Defended Rouge1": defended.get("rouge1"),
        "Delta Rouge1": post_attack.get("rouge1", 0.0) - baseline.get("rouge1", 0.0),
        "Baseline Rouge2": baseline.get("rouge2"),
        "Post-Attack Rouge2": post_attack.get("rouge2"),
        "Defended Rouge2": defended.get("rouge2"),
        "Delta Rouge2": post_attack.get("rouge2", 0.0) - baseline.get("rouge2", 0.0),
        "Baseline RougeL": baseline.get("rougeL"),
        "Post-Attack RougeL": post_attack.get("rougeL"),
        "Defended RougeL": defended.get("rougeL"),
        "Delta RougeL": post_attack.get("rougeL", 0.0) - baseline.get("rougeL", 0.0),
        "Baseline Semantic Sim": baseline.get("semantic_similarity"),
        "Post-Attack Semantic Sim": post_attack.get("semantic_similarity"),
        "Defended Semantic Sim": defended.get("semantic_similarity"),
        "Delta Semantic Sim": post_attack.get("semantic_similarity", 0.0) - baseline.get("semantic_similarity", 0.0),
        "Query Failure Rate Baseline": baseline.get("query_failure_rate"),
        "Query Failure Rate Post-Attack": post_attack.get("query_failure_rate"),
        "Query Failure Rate Defended": defended.get("query_failure_rate"),
        "Availability %": attack_meta.get("availability_before", 1.0) * 100.0,
        "Post-Attack Availability %": attack_meta.get("availability_after", 1.0) * 100.0,
        "Defended Availability %": attack_meta.get("availability_after", 1.0) * 100.0,
        "Membership Acc": attack_meta.get("attack_accuracy"),
        "KB Recovery %": attack_meta.get("kb_recovery_pct"),
        "Runtime (s)": attack_meta.get("runtime", 0.0),
        "Memory Overhead MB": _current_memory_mb(),
        "Notes": (
            f"attack={attack}; "
            f"node_type={attack_meta.get('node_attack_type', 'N/A')}; "
            f"affected={attack_meta.get('affected_nodes', [])}; "
            f"defense_enabled={defense_cfg.enabled}"
        ),
    }


# ------------------------------------------------------------------
# Terminal printing (DRAG-style)
# ------------------------------------------------------------------


def _print_metrics(metrics: dict[str, float], label: str) -> None:
    print(f"\n  {label} METRICS")
    print(f"  {'─' * 50}")
    for k in [
        "exact_match",
        "precision",
        "recall",
        "f1",
        "bleu",
        "rouge1",
        "rougeL",
        "semantic_similarity",
        "query_failure_rate",
    ]:
        v = metrics.get(k, 0.0)
        print(f"  {k:<25} {v:>10.4f}")
    print(f"  Queries: {metrics.get('successful_queries', 0)} successful, "
          f"{metrics.get('failed_queries', 0)} failed")
    sys.stdout.flush()


def _print_comparison(comparison: dict[str, dict[str, float]]) -> None:
    print(f"\n  {'METRIC':<25} {'BASELINE':>10} {'POST-ATTACK':>12} {'DEGRADATION':>14} {'IMPACT %':>10}")
    print(f"  {'─' * 25} {'─' * 10} {'─' * 12} {'─' * 14} {'─' * 10}")
    for metric in [
        "exact_match",
        "precision",
        "recall",
        "f1",
        "bleu",
        "rouge1",
        "rougeL",
        "semantic_similarity",
    ]:
        c = comparison[metric]
        bar_len = int(abs(c["degradation_pct"]) / 5)
        bar = "█" * min(bar_len, 15)
        print(
            f"  {metric.upper():<25} {c['baseline']:>10.4f} "
            f"{c['post_attack']:>12.4f} {c['degradation']:>+14.4f} "
            f"{c['degradation_pct']:>9.2f}%  {bar}"
        )
    qfr = comparison.get("query_failure_rate", {})
    print(f"\n  {'QUERY_FAILURE_RATE':<25} {qfr.get('baseline', 0.0):>10.4f} "
          f"{qfr.get('post_attack', 0.0):>12.4f} "
          f"{qfr.get('degradation', 0.0):>+14.4f}")
    print("=" * 70)
    sys.stdout.flush()


def _print_terminal_summary(
    *,
    baseline: dict[str, float],
    records: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    print("\n" + "=" * 70)
    print("  SYSTEM EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Baseline EM:  {baseline.get('exact_match', 0.0):.4f}")
    print(f"  Baseline F1:  {baseline.get('f1', 0.0):.4f}")
    print(f"  Baseline BLEU: {baseline.get('bleu', 0.0):.4f}")
    print(f"\n  Records generated: {len(records)}")
    for r in records:
        print(f"    - {r['Attack']}: EM={r.get('Post-Attack EM', 'N/A')}")
    print(f"\n  Outputs written to: {output_dir}")
    print("=" * 70 + "\n")
    sys.stdout.flush()


def _current_memory_mb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return max_rss / (1024.0 * 1024.0)
    return max_rss / 1024.0
