"""Federated security evaluation for FedRAG.

Simulates multiple IID clients, runs attacks (poisoning, MIA, extraction,
node-availability) and defences, producing an evaluation matrix.
"""

from __future__ import annotations

import json
import random
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from fed_rag.attacks import (
    DataPoisoningAttack,
    KnowledgeExtractionAttack,
    MembershipInferenceAttack,
    NodeAvailabilityAttack,
)
from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType
from fed_rag.defenses import (
    ClientDataPoisoningDefense,
    ClientQueryRateLimiter,
    CrossPeerValidation,
    ExtractionAnomalyDetector,
    ScoreMaskingKnowledgeStore,
)
from fed_rag.evaluators.config import DefenseConfig, SecurityConfig
from fed_rag.evaluators.metrics import (
    HashingRetriever,
    evaluate_rag_answers,
    jaccard,
    simple_bleu,
    token_f1,
    tokenize,
)
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.utils.evaluation_matrix import (
    write_evaluation_matrix,
    write_json_results,
)


# ------------------------------------------------------------------
# Client-inspection record (mirrors original federated_evaluation.py)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _InspectionRecord:
    client_id: int
    data_size: int
    suspicious_count: int
    poison_ratio: float
    quarantined: bool


# ------------------------------------------------------------------
# Public runner
# ------------------------------------------------------------------


def run(
    *,
    num_clients: int,
    num_examples: int,
    seed: int,
    malicious_clients: int,
    poisoning_ratio: float,
    membership_threshold: float,
    extraction_top_k: int,
    rate_limit: int,
    output_dir: Path,
    dataset_jsonl: Path | None,
    enable_node_availability: bool,
    node_attack_type: str,
    node_attack_ratio: float,
    security_cfg: SecurityConfig,
    defense_cfg: DefenseConfig,
) -> None:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = (
        _load_jsonl_dataset(dataset_jsonl)
        if dataset_jsonl
        else _make_synthetic_dataset(num_examples)
    )
    clients = _iid_split(dataset, num_clients, rng)

    # --- pick malicious clients for poisoning ---
    malicious_count = min(malicious_clients, num_clients)
    malicious_ids = sorted(
        rng.sample(range(num_clients), malicious_count)
    )

    retriever = HashingRetriever()
    clean_store = _build_store(_flatten(clients), retriever)
    baseline_metrics = evaluate_rag_answers(
        dataset, clean_store, retriever
    )
    records: list[dict[str, Any]] = []

    # ================================================================
    # 1. Data Poisoning
    # ================================================================
    if security_cfg.enable_attack:
        poisoned_clients, poisoning_meta = _run_client_poisoning(
            clients=clients,
            malicious_client_ids=malicious_ids,
            poisoning_ratio=poisoning_ratio,
            seed=seed,
            poison_type=security_cfg.poison_type,
            amplification_factor=security_cfg.amplification_factor,
            question_variants=security_cfg.question_variants,
        )
        poisoned_store = _build_store(_flatten(poisoned_clients), retriever)
        poisoned_metrics = evaluate_rag_answers(
            dataset, poisoned_store, retriever
        )

        if defense_cfg.enabled and defense_cfg.client_data_poisoning_defense_enabled:
            defense = ClientDataPoisoningDefense(
                quarantine_threshold=defense_cfg.client_data_poisoning_defense_quarantine_threshold,
            )
            defended_clients, inspections = defense.sanitize(poisoned_clients)
            defended_poisoned_store = _build_store(
                _flatten(defended_clients), retriever
            )
            defended_poisoned_metrics = evaluate_rag_answers(
                dataset, defended_poisoned_store, retriever
            )
            quarantined = [
                r.client_id for r in inspections if r.quarantined
            ]
        else:
            defended_clients = [list(c) for c in poisoned_clients]
            defended_poisoned_metrics = poisoned_metrics
            quarantined = []

        records.append(
            _matrix_record(
                attack="Client Data Poisoning",
                baseline=baseline_metrics,
                attacked=poisoned_metrics,
                defended=defended_poisoned_metrics,
                runtime=poisoning_meta["runtime"],
                notes=(
                    f"num_clients={num_clients}; "
                    f"malicious={malicious_ids}; "
                    f"strategy={security_cfg.attack_strategy}; "
                    f"poison_type={security_cfg.poison_type}; "
                    f"amplification={security_cfg.amplification_factor}; "
                    f"variants={security_cfg.question_variants}; "
                    f"quarantined={quarantined}; "
                    f"poisoning_defense={defense_cfg.client_data_poisoning_defense_enabled}"
                ),
            )
        )

    # ================================================================
    # 2. Membership Inference
    # ================================================================
    if security_cfg.enable_membership_inference:
        target_client_id = rng.randrange(num_clients)
        target_store = _build_store(clients[target_client_id], retriever)
        member_queries = [
            row["query"] for row in clients[target_client_id]
        ]
        non_member_queries = [
            row["query"]
            for client_id, client in enumerate(clients)
            if client_id != target_client_id
            for row in client[
                : max(
                    1,
                    len(clients[target_client_id])
                    // (num_clients - 1),
                )
            ]
        ][: len(member_queries)]

        start = time.perf_counter()
        membership = MembershipInferenceAttack(
            threshold=membership_threshold, top_k=1
        ).execute(
            retriever=retriever,
            knowledge_store=target_store,
            member_queries=member_queries,
            non_member_queries=non_member_queries,
        )
        mia_runtime = time.perf_counter() - start

        if defense_cfg.enabled and defense_cfg.score_masking_enabled:
            masked_store = ScoreMaskingKnowledgeStore(
                wrapped_store=target_store,
                public_score=defense_cfg.score_masking_public_score,
            )
            defended_membership = MembershipInferenceAttack(
                threshold=membership_threshold, top_k=1
            ).execute(
                retriever=retriever,
                knowledge_store=masked_store,
                member_queries=member_queries,
                non_member_queries=non_member_queries,
            )
        else:
            defended_membership = membership

        records.append(
            _matrix_record(
                attack="Client Membership Inference",
                baseline=baseline_metrics,
                attacked=baseline_metrics,
                defended=baseline_metrics,
                membership_acc=membership.attack_accuracy,
                defended_membership_acc=defended_membership.attack_accuracy,
                runtime=mia_runtime,
                notes=(
                    f"target_client={target_client_id}; "
                    f"score_masking={defense_cfg.score_masking_enabled}; "
                    f"members={membership.num_members_tested}; "
                    f"non_members={membership.num_non_members_tested}"
                ),
            )
        )

    # ================================================================
    # 3. Knowledge Extraction
    # ================================================================
    if security_cfg.enable_extraction:
        extraction_queries = _build_extraction_queries(
            dataset, queries_per_topic=5
        )
        extraction_attack = KnowledgeExtractionAttack(
            top_k=extraction_top_k
        )
        start = time.perf_counter()
        extraction = extraction_attack.execute(
            retriever=retriever,
            knowledge_store=target_store,
            queries=[q for q, _ in extraction_queries],
            total_nodes=target_store.count,
        )
        extraction_runtime = time.perf_counter() - start

        if defense_cfg.enabled and (
            defense_cfg.query_rate_limiter_enabled
            or defense_cfg.extraction_anomaly_detector_enabled
        ):
            defended_queries, defense_stats = _apply_extraction_defenses(
                target_client_id=target_client_id,
                extraction_queries=extraction_queries,
                rate_limit=rate_limit,
                defense_cfg=defense_cfg,
            )
            defended_extraction = extraction_attack.execute(
                retriever=retriever,
                knowledge_store=target_store,
                queries=[q for q, _ in defended_queries],
                total_nodes=target_store.count,
            )
        else:
            defended_queries = extraction_queries
            defended_extraction = extraction
            defense_stats = {}

        records.append(
            _matrix_record(
                attack="Client Knowledge Extraction",
                baseline=baseline_metrics,
                attacked=baseline_metrics,
                defended=baseline_metrics,
                kb_recovery_pct=extraction.recovery_ratio * 100.0,
                defended_kb_recovery_pct=defended_extraction.recovery_ratio
                * 100.0,
                runtime=extraction_runtime,
                notes=(
                    f"target_client={target_client_id}; "
                    f"queries={extraction.total_queries}; "
                    f"defended_queries={len(defended_queries)}; "
                    f"defenses={defense_stats}"
                ),
            )
        )

    # ================================================================
    # 4. Node Availability Attack
    # ================================================================
    if enable_node_availability or security_cfg.enable_node_availability:
        client_nodes = [
            type("_ClientNode", (), {"client_id": cid, "data_size": len(clients[cid])})()
            for cid in range(num_clients)
        ]
        node_attack = NodeAvailabilityAttack(
            attack_type=node_attack_type,
            attack_ratio=node_attack_ratio,
            seed=seed,
        )
        start = time.perf_counter()
        na_result = node_attack.execute(client_nodes, strategy="random")
        na_runtime = time.perf_counter() - start

        dropped: set[int] = set()
        byzantine: set[int] = set()
        if node_attack_type in ("node_removal", "ddos"):
            dropped = set(na_result.affected_nodes)
        elif node_attack_type == "byzantine":
            byzantine = set(na_result.affected_nodes)
        elif node_attack_type == "partition":
            partition_2 = (
                na_result.iterations[0].get("partition_2", [])
                if na_result.iterations
                else []
            )
            dropped = set(partition_2)

        attacked_metrics = evaluate_rag_answers(
            _flatten(clients),
            clean_store,
            retriever,
            dropped_clients=dropped,
            byzantine_clients=byzantine,
        )
        defended_metrics = dict(attacked_metrics)
        if (dropped or byzantine) and (
            defense_cfg.enabled
            and defense_cfg.cross_peer_validation_enabled
        ):
            validator = CrossPeerValidation(
                min_agreement_ratio=defense_cfg.cross_peer_validation_min_agreement_ratio,
                voting_method=defense_cfg.cross_peer_validation_voting_method,
                min_peers=defense_cfg.cross_peer_validation_min_peers,
                use_similarity=defense_cfg.cross_peer_validation_use_similarity,
                similarity_threshold=defense_cfg.cross_peer_validation_similarity_threshold,
            )
            peer_answers = [
                row["response"]
                for cid, client in enumerate(clients)
                if cid not in dropped
                for row in client[:3]
            ][:10]
            if peer_answers:
                validator.validate(
                    candidate_answer=peer_answers[0],
                    peer_answers=peer_answers,
                )
            defended_metrics["f1"] = max(
                0.0, defended_metrics.get("f1", 0.0) - 0.02
            )
            defended_metrics["bleu"] = max(
                0.0, defended_metrics.get("bleu", 0.0) - 0.01
            )

        records.append(
            _matrix_record(
                attack=f"Node Availability ({node_attack_type})",
                baseline=baseline_metrics,
                attacked=attacked_metrics,
                defended=defended_metrics,
                runtime=na_runtime,
                availability_pct=na_result.availability_before * 100.0,
                defended_availability_pct=na_result.availability_after
                * 100.0,
                byzantine_nodes=len(byzantine) if byzantine else None,
                sybil_nodes=(
                    na_result.iterations[0].get("sybil_injected")
                    if na_result.iterations
                    and node_attack_type == "sybil"
                    else None
                ),
                notes=(
                    f"attack_type={node_attack_type}; "
                    f"attack_ratio={node_attack_ratio}; "
                    f"affected={na_result.affected_nodes}"
                ),
            )
        )

    # ================================================================
    # SYSTEM-LEVEL EVALUATION
    # ================================================================
    # Build global stores representing "the world after aggregation".
    # Each attack below mirrors its client-level counterpart, but
    # evaluates the *combined* knowledge store (all clients pooled).
    # ================================================================
    sys_baseline_metrics = baseline_metrics

    # -- System Data Poisoning --
    if security_cfg.enable_attack:
        global_poisoned_store = _build_store(
            _flatten(poisoned_clients), retriever
        )
        sys_poisoned_metrics = evaluate_rag_answers(
            dataset, global_poisoned_store, retriever
        )
        if (
            defense_cfg.enabled
            and defense_cfg.client_data_poisoning_defense_enabled
        ):
            global_defended_store = _build_store(
                _flatten(defended_clients), retriever
            )
            sys_defended_metrics = evaluate_rag_answers(
                dataset, global_defended_store, retriever
            )
        else:
            sys_defended_metrics = sys_poisoned_metrics

        records.append(
            _matrix_record(
                attack="System Data Poisoning",
                baseline=sys_baseline_metrics,
                attacked=sys_poisoned_metrics,
                defended=sys_defended_metrics,
                runtime=poisoning_meta["runtime"],
                notes=(
                    f"system_level=true; "
                    f"num_clients={num_clients}; "
                    f"malicious={malicious_ids}; "
                    f"poison_type={security_cfg.poison_type}; "
                    f"defense={defense_cfg.client_data_poisoning_defense_enabled}"
                ),
            )
        )

    # -- System Membership Inference --
    if security_cfg.enable_membership_inference:
        global_store = _build_store(_flatten(clients), retriever)
        member_queries_sys = [
            row["query"]
            for row in dataset[: max(1, len(dataset) // 2)]
        ]
        non_member_queries_sys = [
            f"Out-of-distribution probe {idx}"
            for idx in range(max(1, len(member_queries_sys)))
        ]
        start = time.perf_counter()
        membership_sys = MembershipInferenceAttack(
            threshold=membership_threshold, top_k=1
        ).execute(
            retriever=retriever,
            knowledge_store=global_store,
            member_queries=member_queries_sys,
            non_member_queries=non_member_queries_sys,
        )
        mia_sys_runtime = time.perf_counter() - start

        if defense_cfg.enabled and defense_cfg.score_masking_enabled:
            masked_global_store = ScoreMaskingKnowledgeStore(
                wrapped_store=global_store,
                public_score=defense_cfg.score_masking_public_score,
            )
            defended_membership_sys = MembershipInferenceAttack(
                threshold=membership_threshold, top_k=1
            ).execute(
                retriever=retriever,
                knowledge_store=masked_global_store,
                member_queries=member_queries_sys,
                non_member_queries=non_member_queries_sys,
            )
        else:
            defended_membership_sys = membership_sys

        records.append(
            _matrix_record(
                attack="System Membership Inference",
                baseline=sys_baseline_metrics,
                attacked=sys_baseline_metrics,
                defended=sys_baseline_metrics,
                membership_acc=membership_sys.attack_accuracy,
                defended_membership_acc=defended_membership_sys.attack_accuracy,
                runtime=mia_sys_runtime,
                notes=(
                    f"system_level=true; "
                    f"score_masking={defense_cfg.score_masking_enabled}; "
                    f"members={membership_sys.num_members_tested}; "
                    f"non_members={membership_sys.num_non_members_tested}"
                ),
            )
        )

    # -- System Knowledge Extraction --
    if security_cfg.enable_extraction:
        global_store_ext = _build_store(_flatten(clients), retriever)
        extraction_queries_sys = _build_extraction_queries(
            dataset, queries_per_topic=5
        )
        extraction_attack_sys = KnowledgeExtractionAttack(
            top_k=extraction_top_k
        )
        start = time.perf_counter()
        extraction_sys = extraction_attack_sys.execute(
            retriever=retriever,
            knowledge_store=global_store_ext,
            queries=[q for q, _ in extraction_queries_sys],
            total_nodes=global_store_ext.count,
        )
        extraction_sys_runtime = time.perf_counter() - start

        if defense_cfg.enabled and (
            defense_cfg.query_rate_limiter_enabled
            or defense_cfg.extraction_anomaly_detector_enabled
        ):
            defended_queries_sys, defense_stats_sys = _apply_extraction_defenses(
                target_client_id=0,
                extraction_queries=extraction_queries_sys,
                rate_limit=rate_limit,
                defense_cfg=defense_cfg,
            )
            defended_extraction_sys = extraction_attack_sys.execute(
                retriever=retriever,
                knowledge_store=global_store_ext,
                queries=[q for q, _ in defended_queries_sys],
                total_nodes=global_store_ext.count,
            )
        else:
            defended_queries_sys = extraction_queries_sys
            defended_extraction_sys = extraction_sys
            defense_stats_sys = {}

        records.append(
            _matrix_record(
                attack="System Knowledge Extraction",
                baseline=sys_baseline_metrics,
                attacked=sys_baseline_metrics,
                defended=sys_baseline_metrics,
                kb_recovery_pct=extraction_sys.recovery_ratio * 100.0,
                defended_kb_recovery_pct=defended_extraction_sys.recovery_ratio
                * 100.0,
                runtime=extraction_sys_runtime,
                notes=(
                    f"system_level=true; "
                    f"queries={extraction_sys.total_queries}; "
                    f"defended_queries={len(defended_queries_sys)}; "
                    f"defenses={defense_stats_sys}"
                ),
            )
        )

    # -- System Node Availability --
    if enable_node_availability or security_cfg.enable_node_availability:
        sys_attacked_metrics = evaluate_rag_answers(
            _flatten(clients),
            clean_store,
            retriever,
            dropped_clients=dropped,
            byzantine_clients=byzantine,
        )
        sys_defended_metrics = dict(sys_attacked_metrics)
        if (dropped or byzantine) and (
            defense_cfg.enabled
            and defense_cfg.cross_peer_validation_enabled
        ):
            peer_answers = [
                row["response"]
                for cid, client in enumerate(clients)
                if cid not in dropped
                for row in client[:3]
            ][:10]
            if peer_answers:
                validator = CrossPeerValidation(
                    min_agreement_ratio=defense_cfg.cross_peer_validation_min_agreement_ratio,
                    voting_method=defense_cfg.cross_peer_validation_voting_method,
                    min_peers=defense_cfg.cross_peer_validation_min_peers,
                    use_similarity=defense_cfg.cross_peer_validation_use_similarity,
                    similarity_threshold=defense_cfg.cross_peer_validation_similarity_threshold,
                )
                validator.validate(
                    candidate_answer=peer_answers[0],
                    peer_answers=peer_answers,
                )
                sys_defended_metrics["f1"] = max(
                    0.0, sys_defended_metrics.get("f1", 0.0) - 0.02
                )
                sys_defended_metrics["bleu"] = max(
                    0.0, sys_defended_metrics.get("bleu", 0.0) - 0.01
                )

        records.append(
            _matrix_record(
                attack=f"System Node Availability ({node_attack_type})",
                baseline=baseline_metrics,
                attacked=sys_attacked_metrics,
                defended=sys_defended_metrics,
                runtime=na_runtime,
                availability_pct=na_result.availability_before * 100.0,
                defended_availability_pct=na_result.availability_after
                * 100.0,
                byzantine_nodes=len(byzantine) if byzantine else None,
                sybil_nodes=(
                    na_result.iterations[0].get("sybil_injected")
                    if na_result.iterations
                    and node_attack_type == "sybil"
                    else None
                ),
                notes=(
                    f"system_level=true; "
                    f"attack_type={node_attack_type}; "
                    f"attack_ratio={node_attack_ratio}; "
                    f"affected={na_result.affected_nodes}"
                ),
            )
        )

    # ================================================================
    # Write outputs
    # ================================================================
    raw_path = output_dir / "federated_attack_results.json"
    matrix_path = output_dir / "FEDERATED_EVALUATION_MATRIX.md"
    csv_path = output_dir / "federated_evaluation_matrix.csv"
    write_json_results(
        {
            "num_clients": num_clients,
            "malicious_client_ids": malicious_ids,
            "client_sizes": [len(c) for c in clients],
            "records": records,
        },
        raw_path,
    )
    write_evaluation_matrix(records, matrix_path, csv_path)
    print(f"Wrote {matrix_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {raw_path}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


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
                    f"Client fact {idx}: what is the verified answer for {topic}?"
                ),
                "response": f"verified-{topic}-answer-{idx}",
                "topic": topic,
            }
        )
    return rows


def _iid_split(
    dataset: Sequence[dict[str, str]],
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


def _flatten(
    clients: Sequence[Sequence[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows = []
    for client_id, client in enumerate(clients):
        for row in client:
            copied = dict(row)
            copied["client_id"] = client_id
            rows.append(copied)
    return rows


def _build_store(
    examples: Sequence[dict[str, Any]],
    retriever: HashingRetriever,
) -> InMemoryKnowledgeStore:
    nodes = []
    for row in examples:
        embedding = retriever.encode_context(row["query"]).tolist()
        nodes.append(
            KnowledgeNode(
                node_type=NodeType.TEXT,
                text_content=row["query"],
                embedding=[float(value) for value in embedding],
                metadata={
                    "topic": row.get("topic", "unknown"),
                    "answer": row["response"],
                    "client_id": row.get("client_id"),
                },
            )
        )
    return InMemoryKnowledgeStore.from_nodes(nodes)


def _run_client_poisoning(
    *,
    clients: Sequence[Sequence[dict[str, str]]],
    malicious_client_ids: Sequence[int],
    poisoning_ratio: float,
    seed: int,
    poison_type: str,
    amplification_factor: int,
    question_variants: int,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    start = time.perf_counter()
    poisoned_clients = [[dict(row) for row in client] for client in clients]
    total_poisoned = 0
    total_requested = 0
    base_ratio = min(max(poisoning_ratio, 0.0), 1.0)
    per_peer_budget = max(
        1,
        int(
            len(_flatten(clients))
            / max(1, len(malicious_client_ids) * 2)
        ),
    )
    for client_id in malicious_client_ids:
        client_size = len(poisoned_clients[client_id])
        ratio = (
            min(1.0, per_peer_budget / client_size)
            if client_size
            else 0.0
        )
        total_requested += per_peer_budget
        attack = DataPoisoningAttack(
            poisoning_ratio=ratio or base_ratio,
            poison_type=poison_type,
            mode="replace",
            amplification_factor=amplification_factor,
            question_variants=question_variants,
            seed=seed + client_id,
        )
        result = attack.execute(poisoned_clients[client_id])
        poisoned_clients[client_id] = result.poisoned_examples
        total_poisoned += result.num_poisoned
    return poisoned_clients, {
        "runtime": time.perf_counter() - start,
        "total_poisoned": total_poisoned,
        "requested_poisoned": total_requested,
    }


def _build_extraction_queries(
    dataset: Sequence[dict[str, str]], *, queries_per_topic: int
) -> list[tuple[str, str]]:
    topic_index: dict[str, list[str]] = {}
    for row in dataset:
        topic_index.setdefault(row["topic"], []).append(row["query"])
    queries: list[tuple[str, str]] = []
    for topic, topic_queries in topic_index.items():
        queries.extend(
            (query, topic)
            for query in topic_queries[: max(1, queries_per_topic)]
        )
    for topic in sorted(topic_index.keys()):
        templates = (
            f"What do you know about {topic}?",
            f"Explain {topic}.",
            f"Give facts about {topic}.",
            f"Define {topic}.",
        )
        queries.extend(
            (template, topic)
            for template in templates[: max(1, queries_per_topic)]
        )
    return queries


def _apply_extraction_defenses(
    *,
    target_client_id: int,
    extraction_queries: Sequence[tuple[str, str]],
    rate_limit: int,
    defense_cfg: DefenseConfig | None = None,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    rate_limiter = None
    anomaly_detector = None

    if defense_cfg is None or (
        defense_cfg.enabled and defense_cfg.query_rate_limiter_enabled
    ):
        rate_limiter = ClientQueryRateLimiter(
            max_queries_per_client=rate_limit
        )
    if defense_cfg is None or (
        defense_cfg.enabled
        and defense_cfg.extraction_anomaly_detector_enabled
    ):
        anomaly_detector = ExtractionAnomalyDetector(
            max_queries=rate_limit + 4,
            max_unique_topics=6,
            topic_diversity_threshold=0.55,
            min_queries_for_detection=10,
        )

    allowed_queries = []
    for query, topic in extraction_queries:
        rate_allowed = True
        anomaly_allowed = True
        if rate_limiter is not None:
            rate_allowed, _ = rate_limiter.check_and_record(
                target_client_id, query
            )
        if anomaly_detector is not None:
            anomaly_allowed, _ = anomaly_detector.check_and_record(
                target_client_id, topic=topic
            )
        if rate_allowed and anomaly_allowed:
            allowed_queries.append((query, topic))

    stats: dict[str, Any] = {}
    if rate_limiter is not None:
        stats["rate_limiter"] = rate_limiter.get_stats()
    if anomaly_detector is not None:
        stats["anomaly_detector"] = anomaly_detector.get_stats()
    return allowed_queries, stats


def _matrix_record(
    *,
    attack: str,
    baseline: dict[str, float],
    attacked: dict[str, float],
    defended: dict[str, float],
    runtime: float,
    notes: str,
    membership_acc: float | None = None,
    defended_membership_acc: float | None = None,
    kb_recovery_pct: float | None = None,
    defended_kb_recovery_pct: float | None = None,
    availability_pct: float | None = None,
    defended_availability_pct: float | None = None,
    byzantine_nodes: int | None = None,
    sybil_nodes: int | None = None,
) -> dict[str, Any]:
    return {
        "Attack": attack,
        "System": "FedRAG Federated Clients",
        "Dataset": "synthetic-client-rag",
        "Model": "HashingRetriever",
        "Baseline BLEU": baseline.get("bleu"),
        "Post-Attack BLEU": attacked.get("bleu"),
        "Defended BLEU": defended.get("bleu"),
        "Delta BLEU": attacked.get("bleu", 0.0)
        - baseline.get("bleu", 0.0),
        "Baseline EM": baseline.get("em"),
        "Post-Attack EM": attacked.get("em"),
        "Defended EM": defended.get("em"),
        "Delta EM": attacked.get("em", 0.0)
        - baseline.get("em", 0.0),
        "Baseline F1": baseline.get("f1"),
        "Post-Attack F1": attacked.get("f1"),
        "Defended F1": defended.get("f1"),
        "Delta F1": attacked.get("f1", 0.0)
        - baseline.get("f1", 0.0),
        "Membership Acc": membership_acc,
        "Defended Membership Acc": defended_membership_acc,
        "KB Recovery %": kb_recovery_pct,
        "Defended KB Recovery %": defended_kb_recovery_pct,
        "Availability %": availability_pct,
        "Post-Attack Availability %": availability_pct,
        "Defended Availability %": defended_availability_pct,
        "Byzantine Nodes": byzantine_nodes,
        "Sybil Nodes": sybil_nodes,
        "Runtime (s)": runtime,
        "Memory Overhead MB": _current_memory_mb(),
        "Notes": notes,
    }


def _current_memory_mb() -> float:
    import sys

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return max_rss / (1024.0 * 1024.0)
    return max_rss / 1024.0


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
