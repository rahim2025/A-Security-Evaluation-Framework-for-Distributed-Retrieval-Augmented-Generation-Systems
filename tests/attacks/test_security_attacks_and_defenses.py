from __future__ import annotations

from typing import Any

import torch

from fed_rag.attacks import KnowledgeExtractionAttack, MembershipInferenceAttack
from fed_rag.base.retriever import BaseRetriever
from fed_rag.data_structures.knowledge_node import KnowledgeNode
from fed_rag.defenses import (
    ClientDataPoisoningDefense,
    ClientQueryRateLimiter,
    ExtractionAnomalyDetector,
    ScoreMaskingKnowledgeStore,
)
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore


class ToyRetriever(BaseRetriever):
    vectors: dict[str, list[float]]

    def encode_query(self, query: Any, **kwargs: Any) -> torch.Tensor:
        return torch.tensor(self.vectors[str(query)], dtype=torch.float32)

    def encode_context(self, context: Any, **kwargs: Any) -> torch.Tensor:
        return self.encode_query(context)

    @property
    def encoder(self) -> torch.nn.Module | None:
        return None

    @property
    def query_encoder(self) -> torch.nn.Module | None:
        return None

    @property
    def context_encoder(self) -> torch.nn.Module | None:
        return None


def make_node(text: str, embedding: list[float]) -> KnowledgeNode:
    return KnowledgeNode(
        node_type="text",
        text_content=text,
        embedding=embedding,
    )


def test_membership_inference_uses_retrieval_confidence() -> None:
    store = InMemoryKnowledgeStore.from_nodes(
        [
            make_node("alpha private fact", [1.0, 0.0, 0.0]),
            make_node("beta private fact", [0.0, 1.0, 0.0]),
        ]
    )
    retriever = ToyRetriever(
        vectors={
            "alpha private fact": [1.0, 0.0, 0.0],
            "beta private fact": [0.0, 1.0, 0.0],
            "unseen outside fact": [0.0, 0.0, 1.0],
        }
    )

    result = MembershipInferenceAttack(threshold=0.8).execute(
        retriever=retriever,
        knowledge_store=store,
        member_queries=["alpha private fact", "beta private fact"],
        non_member_queries=["unseen outside fact"],
    )

    assert result.attack_accuracy == 1.0
    assert result.true_positive_rate == 1.0
    assert result.false_positive_rate == 0.0
    assert result.num_members_tested == 2
    assert result.num_non_members_tested == 1


def test_kb_extraction_recovers_nodes_from_public_retrieve() -> None:
    nodes = [
        make_node("alpha private fact", [1.0, 0.0, 0.0]),
        make_node("beta private fact", [0.0, 1.0, 0.0]),
        make_node("gamma private fact", [0.0, 0.0, 1.0]),
    ]
    store = InMemoryKnowledgeStore.from_nodes(nodes)
    retriever = ToyRetriever(
        vectors={
            "alpha": [1.0, 0.0, 0.0],
            "beta": [0.0, 1.0, 0.0],
            "gamma": [0.0, 0.0, 1.0],
        }
    )

    result = KnowledgeExtractionAttack(top_k=1).execute(
        retriever=retriever,
        knowledge_store=store,
        queries=["alpha", "beta", "gamma"],
    )

    assert result.total_nodes == 3
    assert result.total_queries == 3
    assert result.successful_queries == 3
    assert result.recovery_ratio == 1.0
    assert {node.node_id for node in result.recovered_nodes} == {
        node.node_id for node in nodes
    }


def test_kb_extraction_generates_topic_queries() -> None:
    queries = KnowledgeExtractionAttack().generate_queries(
        ["cryptography"], limit=2
    )

    assert queries == [
        "What do you know about cryptography?",
        "Explain cryptography.",
    ]


def test_client_data_poisoning_defense_quarantines_risky_client() -> None:
    clients = [
        [
            {"query": "q1", "response": "a1"},
            {"query": "q2", "response": "a2"},
        ],
        [
            {"query": "q3", "response": "POISONED wrong answer"},
            {"query": "q4", "response": "POISONED wrong answer"},
        ],
    ]

    sanitized, inspections = ClientDataPoisoningDefense(
        quarantine_threshold=0.5
    ).sanitize(clients)

    assert sanitized[0] == clients[0]
    assert sanitized[1] == []
    assert inspections[0].quarantined is False
    assert inspections[1].quarantined is True
    assert inspections[1].poisoned_marker_count == 2


def test_query_rate_limiter_blocks_after_client_budget() -> None:
    limiter = ClientQueryRateLimiter(max_queries_per_client=2)

    assert limiter.check_and_record(0, "q1")[0] is True
    assert limiter.check_and_record(0, "q2")[0] is True
    allowed, reason = limiter.check_and_record(0, "q3")

    assert allowed is False
    assert reason == "rate_limit_exceeded"
    assert limiter.get_stats()["blocked_count"] == 1


def test_extraction_anomaly_detector_flags_broad_topic_queries() -> None:
    detector = ExtractionAnomalyDetector(
        max_unique_topics=2,
        min_queries_for_detection=3,
        topic_diversity_threshold=0.9,
    )

    assert detector.check_and_record(0, topic="a")[0] is True
    assert detector.check_and_record(0, topic="b")[0] is True
    allowed, reason = detector.check_and_record(0, topic="c")

    assert allowed is False
    assert reason == "broad_topic_coverage"
    assert detector.get_stats()["flagged_clients"] == {0: "broad_topic_coverage"}


def test_score_masking_knowledge_store_preserves_nodes_and_masks_scores() -> None:
    node = make_node("private fact", [1.0, 0.0])
    wrapped = InMemoryKnowledgeStore.from_nodes([node])
    masked = ScoreMaskingKnowledgeStore(wrapped_store=wrapped, public_score=0.25)

    retrieved = masked.retrieve(query_emb=[1.0, 0.0], top_k=1)

    assert retrieved == [(0.25, node)]
    assert masked.count == 1
