"""Knowledge-base extraction attack utilities for FedRAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.base.retriever import BaseRetriever
from fed_rag.data_structures.knowledge_node import KnowledgeNode


DEFAULT_QUERY_TEMPLATES = (
    "What do you know about {topic}?",
    "Explain {topic}.",
    "Give facts about {topic}.",
    "Define {topic}.",
)


@dataclass(frozen=True)
class KBExtractionResult:
    """Results from extracting knowledge through public retrieval."""

    recovered_nodes: list[KnowledgeNode]
    total_nodes: int
    recovery_ratio: float
    total_queries: int
    successful_queries: int
    top_k: int


class KnowledgeExtractionAttack:
    """Recover stored nodes by repeatedly probing a FedRAG knowledge store.

    The attack is intentionally store-agnostic: it only uses
    BaseRetriever.encode_query and BaseKnowledgeStore.retrieve. This models an
    external attacker who can issue queries but does not need direct access to
    implementation-specific internals.
    """

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = max(1, int(top_k))

    def execute(
        self,
        retriever: BaseRetriever,
        knowledge_store: BaseKnowledgeStore,
        queries: Sequence[str],
        total_nodes: int | None = None,
    ) -> KBExtractionResult:
        """Run extraction with a caller-provided query set."""

        recovered_by_id: dict[str, KnowledgeNode] = {}
        successful_queries = 0

        for query in queries:
            query_emb = _to_embedding_list(retriever.encode_query(query))
            retrieved = knowledge_store.retrieve(query_emb=query_emb, top_k=self.top_k)
            if retrieved:
                successful_queries += 1
            for _, node in retrieved:
                recovered_by_id.setdefault(node.node_id, node)

        denominator = total_nodes if total_nodes is not None else knowledge_store.count
        recovery_ratio = (
            len(recovered_by_id) / denominator if denominator else 0.0
        )
        return KBExtractionResult(
            recovered_nodes=list(recovered_by_id.values()),
            total_nodes=denominator,
            recovery_ratio=recovery_ratio,
            total_queries=len(queries),
            successful_queries=successful_queries,
            top_k=self.top_k,
        )

    def generate_queries(
        self,
        topics: Iterable[str],
        templates: Sequence[str] = DEFAULT_QUERY_TEMPLATES,
        limit: int | None = None,
    ) -> list[str]:
        """Generate generic extraction probes from topic labels."""

        queries = [
            template.format(topic=topic)
            for topic in topics
            for template in templates
        ]
        return queries[:limit] if limit is not None else queries


def _to_embedding_list(encoded: Any) -> list[float]:
    if hasattr(encoded, "embedding"):
        encoded = encoded.embedding
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu()
        if encoded.dim() > 1:
            encoded = encoded.squeeze(0)
        return [float(v) for v in encoded.tolist()]
    return [float(v) for v in encoded]
