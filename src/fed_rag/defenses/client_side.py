"""Client-side defenses for federated RAG security experiments."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.data_structures.knowledge_node import KnowledgeNode


@dataclass(frozen=True)
class ClientInspection:
    """Risk summary for one federated client dataset."""

    client_id: int
    total_examples: int
    poisoned_marker_count: int
    duplicate_query_count: int
    conflicting_answer_count: int
    risk_score: float
    quarantined: bool


class ClientDataPoisoningDefense:
    """Sanitize poisoned client datasets before federated aggregation.

    The defense is intentionally client-side and data-local: each client is
    inspected for obvious poisoning markers, repeated query amplification, and
    conflicting answers for the same query. Clients above the risk threshold can
    be quarantined; lower-risk clients have only suspicious records removed.
    """

    def __init__(
        self,
        *,
        poison_markers: Iterable[str] = ("POISONED",),
        quarantine_threshold: float = 0.5,
        query_key: str = "query",
        response_key: str = "response",
    ) -> None:
        self.poison_markers = tuple(marker.lower() for marker in poison_markers)
        self.quarantine_threshold = quarantine_threshold
        self.query_key = query_key
        self.response_key = response_key

    def inspect(
        self, clients: Sequence[Sequence[dict[str, Any]]]
    ) -> list[ClientInspection]:
        """Inspect all clients and return per-client risk summaries."""

        return [
            self._inspect_client(client_id, examples)
            for client_id, examples in enumerate(clients)
        ]

    def sanitize(
        self, clients: Sequence[Sequence[dict[str, Any]]]
    ) -> tuple[list[list[dict[str, Any]]], list[ClientInspection]]:
        """Drop quarantined clients and suspicious records from others."""

        inspections = self.inspect(clients)
        sanitized: list[list[dict[str, Any]]] = []

        for inspection, examples in zip(inspections, clients):
            # Always filter at record level — even quarantined clients keep
            # their clean records so the 20% good data is not thrown away.
            seen_answers: dict[str, str] = {}
            clean_examples: list[dict[str, Any]] = []
            for example in examples:
                query = str(example.get(self.query_key, ""))
                answer = str(example.get(self.response_key, ""))
                if self._has_poison_marker(answer):
                    continue
                if query in seen_answers and seen_answers[query] != answer:
                    continue
                seen_answers[query] = answer
                clean_examples.append(dict(example))
            sanitized.append(clean_examples)

        return sanitized, inspections

    def _inspect_client(
        self, client_id: int, examples: Sequence[dict[str, Any]]
    ) -> ClientInspection:
        total = len(examples)
        answers = [str(row.get(self.response_key, "")) for row in examples]
        marker_count = sum(self._has_poison_marker(answer) for answer in answers)

        query_counts = Counter(str(row.get(self.query_key, "")) for row in examples)
        duplicate_count = sum(count - 1 for count in query_counts.values() if count > 1)

        answers_by_query: dict[str, set[str]] = defaultdict(set)
        for row in examples:
            answers_by_query[str(row.get(self.query_key, ""))].add(
                str(row.get(self.response_key, ""))
            )
        conflict_count = sum(
            len(answers) - 1 for answers in answers_by_query.values() if len(answers) > 1
        )

        denominator = max(1, total)
        risk_score = (marker_count + duplicate_count + conflict_count) / denominator
        return ClientInspection(
            client_id=client_id,
            total_examples=total,
            poisoned_marker_count=marker_count,
            duplicate_query_count=duplicate_count,
            conflicting_answer_count=conflict_count,
            risk_score=risk_score,
            quarantined=risk_score >= self.quarantine_threshold,
        )

    def _has_poison_marker(self, answer: str) -> bool:
        normalized = answer.lower()
        return any(marker in normalized for marker in self.poison_markers)


class ClientQueryRateLimiter:
    """Deterministic per-client query budget for extraction simulations."""

    def __init__(self, max_queries_per_client: int = 20) -> None:
        self.max_queries_per_client = max(1, int(max_queries_per_client))
        self._queries: dict[int, deque[str]] = defaultdict(deque)
        self.allowed_count = 0
        self.blocked_count = 0

    def check_and_record(self, client_id: int, query: str) -> tuple[bool, str]:
        """Return whether a client can issue another query."""

        client_queries = self._queries[client_id]
        if len(client_queries) >= self.max_queries_per_client:
            self.blocked_count += 1
            return False, "rate_limit_exceeded"

        client_queries.append(query)
        self.allowed_count += 1
        return True, "allowed"

    def get_stats(self) -> dict[str, Any]:
        """Return defense counters."""

        total = self.allowed_count + self.blocked_count
        return {
            "allowed_count": self.allowed_count,
            "blocked_count": self.blocked_count,
            "block_rate": self.blocked_count / max(1, total),
            "max_queries_per_client": self.max_queries_per_client,
        }


class ExtractionAnomalyDetector:
    """Flag client-side extraction behavior by query volume and topic spread."""

    def __init__(
        self,
        *,
        max_queries: int = 30,
        max_unique_topics: int = 6,
        topic_diversity_threshold: float = 0.6,
        min_queries_for_detection: int = 10,
    ) -> None:
        self.max_queries = max_queries
        self.max_unique_topics = max_unique_topics
        self.topic_diversity_threshold = topic_diversity_threshold
        self.min_queries_for_detection = min_queries_for_detection
        self._state: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "topics": set()}
        )
        self.flagged_clients: dict[int, str] = {}
        self.allowed_count = 0
        self.blocked_count = 0

    def check_and_record(
        self, client_id: int, *, topic: str | None = None
    ) -> tuple[bool, str]:
        """Record one query and decide whether it should be allowed."""

        if client_id in self.flagged_clients:
            self.blocked_count += 1
            return False, self.flagged_clients[client_id]

        state = self._state[client_id]
        state["total"] += 1
        if topic:
            state["topics"].add(topic)

        total = int(state["total"])
        unique_topics = len(state["topics"])
        diversity = unique_topics / max(1, total)
        reason = self._flag_reason(total, unique_topics, diversity)
        if reason:
            self.flagged_clients[client_id] = reason
            self.blocked_count += 1
            return False, reason

        self.allowed_count += 1
        return True, "allowed"

    def get_stats(self) -> dict[str, Any]:
        """Return defense counters."""

        total = self.allowed_count + self.blocked_count
        return {
            "allowed_count": self.allowed_count,
            "blocked_count": self.blocked_count,
            "block_rate": self.blocked_count / max(1, total),
            "flagged_clients": dict(self.flagged_clients),
        }

    def _flag_reason(
        self, total: int, unique_topics: int, diversity: float
    ) -> str | None:
        if total < self.min_queries_for_detection:
            return None
        if total > self.max_queries:
            return "high_query_volume"
        if unique_topics > self.max_unique_topics:
            return "broad_topic_coverage"
        if diversity >= self.topic_diversity_threshold:
            return "high_topic_diversity"
        return None


class ScoreMaskingKnowledgeStore(BaseKnowledgeStore):
    """Wrap a store and mask similarity scores returned to clients.

    Membership inference commonly relies on exact retrieval confidence. This
    wrapper preserves retrieved nodes while exposing coarser scores to the
    client, reducing the confidence signal available to the attacker.
    """

    wrapped_store: BaseKnowledgeStore
    public_score: float = 0.5

    def load_node(self, node: KnowledgeNode) -> None:
        self.wrapped_store.load_node(node)

    def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        self.wrapped_store.load_nodes(nodes)

    def retrieve(
        self, query_emb: list[float], top_k: int
    ) -> list[tuple[float, KnowledgeNode]]:
        return [
            (self.public_score, node)
            for _, node in self.wrapped_store.retrieve(query_emb=query_emb, top_k=top_k)
        ]

    def batch_retrieve(
        self, query_embs: list[list[float]], top_k: int
    ) -> list[list[tuple[float, KnowledgeNode]]]:
        return [
            self.retrieve(query_emb=query_emb, top_k=top_k)
            for query_emb in query_embs
        ]

    def delete_node(self, node_id: str) -> bool:
        return self.wrapped_store.delete_node(node_id)

    def clear(self) -> None:
        self.wrapped_store.clear()

    @property
    def count(self) -> int:
        return self.wrapped_store.count

    def persist(self) -> None:
        self.wrapped_store.persist()

    def load(self) -> None:
        self.wrapped_store.load()
