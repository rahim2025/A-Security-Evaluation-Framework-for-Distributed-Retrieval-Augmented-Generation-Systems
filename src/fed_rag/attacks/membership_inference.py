"""Membership inference attack utilities for FedRAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.base.retriever import BaseRetriever


@dataclass(frozen=True)
class MembershipInferenceResult:
    """Per-query membership inference decision."""

    query: str
    is_member: bool
    score: float
    threshold: float
    retrieved_node_id: str | None = None


@dataclass(frozen=True)
class MembershipInferenceAttackResult:
    """Aggregate membership inference attack results."""

    results: list[MembershipInferenceResult]
    attack_accuracy: float
    true_positive_rate: float
    false_positive_rate: float
    precision: float
    recall: float
    num_members_tested: int
    num_non_members_tested: int
    threshold: float
    top_k: int


class MembershipInferenceAttack:
    """Infer whether query-like text is present in a FedRAG knowledge store.

    DRAG's implementation uses distributed routing features such as hop count
    and peer messages. FedRAG's common abstraction exposes retriever embeddings
    and knowledge-store retrieval scores, so this attack uses the strongest
    public signal available across stores: top-k retrieval confidence.
    """

    def __init__(self, threshold: float = 0.8, top_k: int = 1) -> None:
        self.threshold = threshold
        self.top_k = max(1, int(top_k))

    def infer(
        self,
        query: str,
        retriever: BaseRetriever,
        knowledge_store: BaseKnowledgeStore,
    ) -> MembershipInferenceResult:
        """Infer membership for one query."""

        query_emb = _to_embedding_list(retriever.encode_query(query))
        retrieved = knowledge_store.retrieve(query_emb=query_emb, top_k=self.top_k)
        if not retrieved:
            return MembershipInferenceResult(
                query=query,
                is_member=False,
                score=0.0,
                threshold=self.threshold,
            )

        score, node = max(retrieved, key=lambda item: item[0])
        node_id = getattr(node, "node_id", None)
        return MembershipInferenceResult(
            query=query,
            is_member=float(score) >= self.threshold,
            score=float(score),
            threshold=self.threshold,
            retrieved_node_id=node_id,
        )

    def execute(
        self,
        retriever: BaseRetriever,
        knowledge_store: BaseKnowledgeStore,
        member_queries: Sequence[str],
        non_member_queries: Sequence[str] | None = None,
    ) -> MembershipInferenceAttackResult:
        """Run membership inference and compute attack metrics.

        Args:
            retriever: FedRAG retriever used by the target RAG pipeline.
            knowledge_store: Target knowledge store.
            member_queries: Queries known to correspond to stored knowledge.
            non_member_queries: Queries known to be outside the store.
        """

        labeled_queries: list[tuple[str, bool]] = [
            (query, True) for query in member_queries
        ]
        labeled_queries.extend(
            (query, False) for query in (non_member_queries or [])
        )

        results = [
            self.infer(query, retriever, knowledge_store)
            for query, _ in labeled_queries
        ]
        labels = [label for _, label in labeled_queries]
        metrics = _classification_metrics(labels, [r.is_member for r in results])

        return MembershipInferenceAttackResult(
            results=results,
            attack_accuracy=metrics["accuracy"],
            true_positive_rate=metrics["true_positive_rate"],
            false_positive_rate=metrics["false_positive_rate"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            num_members_tested=sum(labels),
            num_non_members_tested=len(labels) - sum(labels),
            threshold=self.threshold,
            top_k=self.top_k,
        )


def _classification_metrics(
    labels: Sequence[bool], predictions: Sequence[bool]
) -> dict[str, float]:
    tp = sum(1 for y, pred in zip(labels, predictions) if y and pred)
    tn = sum(1 for y, pred in zip(labels, predictions) if not y and not pred)
    fp = sum(1 for y, pred in zip(labels, predictions) if not y and pred)
    fn = sum(1 for y, pred in zip(labels, predictions) if y and not pred)
    total = tp + tn + fp + fn

    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "true_positive_rate": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def _to_embedding_list(encoded: Any) -> list[float]:
    if hasattr(encoded, "embedding"):
        encoded = encoded.embedding
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu()
        if encoded.dim() > 1:
            encoded = encoded.squeeze(0)
        return [float(v) for v in encoded.tolist()]
    return [float(v) for v in encoded]
