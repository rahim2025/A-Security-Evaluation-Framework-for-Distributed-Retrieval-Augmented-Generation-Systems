"""
FedRAGSimulator
===============
A lightweight, dependency-free simulator that models a federated RAG
network using only the Python standard library.  No GPU, no Hugging Face,
no Ollama — so the security evaluation framework can be exercised on any
machine without heavyweight ML dependencies.

Retrieval is implemented as hash-cosine similarity (deterministic and fast).
This intentionally mirrors the ``HashingRetriever`` already used in
``fed_rag.evaluators.metrics`` so results are reproducible.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class SimNode:
    """One knowledge-store node."""

    node_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimClient:
    """One federated client with its local dataset and knowledge store."""

    client_id: int
    examples: list[dict[str, Any]]
    nodes: list[SimNode] = field(default_factory=list)

    @property
    def data_size(self) -> int:
        return len(self.examples)


# ---------------------------------------------------------------------------
# Hashing retriever (no ML dependencies)
# ---------------------------------------------------------------------------


def _hash_embedding(text: str, dim: int = 32) -> list[float]:
    """Produce a deterministic unit-norm embedding from text via SHA-256."""
    digest = hashlib.sha256(text.encode()).digest()
    # Expand digest to *dim* floats in [-1, 1]
    floats: list[float] = []
    for i in range(dim):
        byte = digest[i % len(digest)]
        floats.append((byte / 127.5) - 1.0)
    norm = math.sqrt(sum(v * v for v in floats)) or 1.0
    return [v / norm for v in floats]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(y * y for y in b) or 1.0  # type: ignore[arg-type]
    return dot / (na * nb)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# FedRAGSimulator
# ---------------------------------------------------------------------------


TOPICS = [
    "cryptography", "medicine", "finance", "biology", "law",
    "networking", "history", "physics", "privacy", "safety",
    "databases", "governance",
]


class FedRAGSimulator:
    """Simulate a federated RAG deployment for security evaluation.

    Parameters
    ----------
    num_clients:
        Number of federated clients.
    num_examples:
        Total number of (query, response) pairs before IID splitting.
    seed:
        Random seed for reproducibility.
    embedding_dim:
        Dimensionality of the hash-based embedding vectors.
    dataset_jsonl:
        Optional path to an existing JSONL dataset (query/response/topic).
        If ``None``, a synthetic dataset is generated automatically.
    """

    def __init__(
        self,
        num_clients: int = 10,
        num_examples: int = 200,
        seed: int = 0,
        embedding_dim: int = 32,
        dataset_jsonl: Path | str | None = None,
    ) -> None:
        self.num_clients = num_clients
        self.num_examples = num_examples
        self.seed = seed
        self.embedding_dim = embedding_dim
        self._rng = random.Random(seed)

        if dataset_jsonl is not None:
            p = Path(dataset_jsonl)
            if not p.exists():
                raise FileNotFoundError(
                    f"\n\n  Dataset file not found: {dataset_jsonl}\n"
                    f"\n  To run without a dataset, omit the dataset_jsonl argument"
                    f"\n  and a synthetic dataset will be generated automatically.\n"
                    f"\n  Command:  python run_global_impact_analysis.py\n"
                )
            self.dataset = self._load_jsonl(p)
        else:
            self.dataset = self._make_synthetic_dataset()

        self.clients: list[SimClient] = self._build_clients()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        client_id: int | None = None,
        top_k: int = 1,
        score_mask: float | None = None,
    ) -> list[tuple[float, SimNode]]:
        """Retrieve the top-*k* nodes matching *query* from one or all clients.

        Parameters
        ----------
        query:
            The query string.
        client_id:
            If given, search only that client's store; otherwise search all.
        top_k:
            Number of results to return.
        score_mask:
            If set, replace actual scores with this fixed value (score masking
            defense).
        """
        q_emb = _hash_embedding(query, self.embedding_dim)
        nodes: list[SimNode] = []
        if client_id is not None:
            nodes = self.clients[client_id].nodes
        else:
            for client in self.clients:
                nodes.extend(client.nodes)

        scored = [(float(_cosine(q_emb, n.embedding)), n) for n in nodes]
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:top_k]
        if score_mask is not None:
            top = [(score_mask, n) for _, n in top]
        return top

    def get_answer(self, query: str, client_id: int | None = None) -> str:
        """Retrieve the best-matching answer for *query*."""
        results = self.retrieve(query, client_id=client_id)
        if not results:
            return ""
        _, node = results[0]
        return str(node.metadata.get("answer", ""))

    def rebuild_stores(self) -> None:
        """Rebuild every client's knowledge store from its current examples."""
        for client in self.clients:
            client.nodes = self._build_nodes(client.examples)

    def to_dict(self) -> dict[str, Any]:
        """Serialise simulator metadata (not the full node list)."""
        return {
            "num_clients": self.num_clients,
            "num_examples": self.num_examples,
            "seed": self.seed,
            "embedding_dim": self.embedding_dim,
            "client_sizes": [c.data_size for c in self.clients],
            "topics": TOPICS,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_synthetic_dataset(self) -> list[dict[str, Any]]:
        rows = []
        for idx in range(self.num_examples):
            topic = TOPICS[idx % len(TOPICS)]
            rows.append({
                "query": f"What is the key fact #{idx} about {topic}?",
                "response": f"verified-{topic}-answer-{idx}",
                "topic": topic,
            })
        return rows

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _build_clients(self) -> list[SimClient]:
        shuffled = list(self.dataset)
        self._rng.shuffle(shuffled)
        base, remainder = divmod(len(shuffled), self.num_clients)
        clients: list[SimClient] = []
        offset = 0
        for cid in range(self.num_clients):
            size = base + (1 if cid < remainder else 0)
            examples = shuffled[offset: offset + size]
            offset += size
            clients.append(SimClient(
                client_id=cid,
                examples=examples,
                nodes=self._build_nodes(examples),
            ))
        return clients

    def _build_nodes(self, examples: list[dict[str, Any]]) -> list[SimNode]:
        nodes: list[SimNode] = []
        for row in examples:
            query = row.get("query", "")
            emb = _hash_embedding(query, self.embedding_dim)
            nid = hashlib.md5(query.encode()).hexdigest()[:12]
            nodes.append(SimNode(
                node_id=nid,
                text=query,
                embedding=emb,
                metadata={
                    "topic": row.get("topic", "unknown"),
                    "answer": row.get("response", ""),
                },
            ))
        return nodes
