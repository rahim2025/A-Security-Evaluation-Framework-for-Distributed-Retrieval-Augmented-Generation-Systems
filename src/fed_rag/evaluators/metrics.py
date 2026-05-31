"""Shared evaluation metrics for FedRAG security benchmarking."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import torch

from fed_rag.base.retriever import BaseRetriever
from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def token_f1(pred: str, actual: str) -> float:
    pred_tokens = tokenize(pred)
    actual_tokens = tokenize(actual)
    common = Counter(pred_tokens) & Counter(actual_tokens)
    overlap = sum(common.values())
    if not pred_tokens or not actual_tokens or overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(actual_tokens)
    return 2 * precision * recall / (precision + recall)


def simple_bleu(pred: str, actual: str) -> float:
    pred_tokens = tokenize(pred)
    actual_tokens = tokenize(actual)
    if not pred_tokens or not actual_tokens:
        return 0.0
    overlap = sum(
        (Counter(pred_tokens) & Counter(actual_tokens)).values()
    )
    return overlap / len(pred_tokens)


def jaccard(pred: str, actual: str) -> float:
    pred_set = set(tokenize(pred))
    actual_set = set(tokenize(actual))
    return (
        len(pred_set & actual_set) / len(pred_set | actual_set)
        if pred_set or actual_set
        else 1.0
    )


def evaluate_lookup(
    dataset: list[dict[str, Any]], lookup: dict[str, str]
) -> dict[str, float]:
    scores: dict[str, list[float]] = {
        "em": [],
        "f1": [],
        "bleu": [],
        "jaccard": [],
    }
    for row in dataset:
        pred = lookup.get(row["query"], "")
        actual = row["response"]
        scores["em"].append(
            float(pred.strip().lower() == actual.strip().lower())
        )
        scores["f1"].append(token_f1(pred, actual))
        scores["bleu"].append(simple_bleu(pred, actual))
        scores["jaccard"].append(jaccard(pred, actual))
    return {
        key: sum(values) / max(1, len(values))
        for key, values in scores.items()
    }


def evaluate_rag_answers(
    dataset: list[dict[str, Any]],
    knowledge_store: InMemoryKnowledgeStore,
    retriever: BaseRetriever,
    dropped_clients: set[int] | None = None,
    byzantine_clients: set[int] | None = None,
) -> dict[str, float]:
    scores: dict[str, list[float]] = {
        "em": [],
        "f1": [],
        "bleu": [],
        "jaccard": [],
    }
    for row in dataset:
        client_id = row.get("client_id")
        if dropped_clients and client_id in dropped_clients:
            pred = "QUERY_FAILED_NODE_UNAVAILABLE"
        else:
            retrieved = knowledge_store.retrieve(
                query_emb=retriever.encode_query(row["query"]).tolist(),
                top_k=1,
            )
            pred = (
                str(retrieved[0][1].metadata.get("answer", ""))
                if retrieved
                else ""
            )
        if byzantine_clients and client_id in byzantine_clients:
            pred = "INCORRECT_BYZANTINE_RESPONSE"
        actual = row["response"]
        scores["em"].append(
            float(pred.strip().lower() == actual.strip().lower())
        )
        scores["f1"].append(token_f1(pred, actual))
        scores["bleu"].append(simple_bleu(pred, actual))
        scores["jaccard"].append(jaccard(pred, actual))
    return {
        key: sum(values) / max(1, len(values))
        for key, values in scores.items()
    }


class HashingRetriever(BaseRetriever):
    """Small deterministic retriever for offline experiments."""

    dimensions: int = 128

    def encode_query(self, query: Any, **kwargs: Any) -> torch.Tensor:
        return torch.tensor(
            self._encode(str(query)), dtype=torch.float32
        )

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

    def _encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else vector
