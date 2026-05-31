"""
FedRAGSimulator
===============
A lightweight, dependency-free simulator that models a federated RAG
network using only the Python standard library.  No GPU, no Hugging Face,
no Ollama — so the security evaluation framework can be exercised on any
machine without heavyweight ML dependencies.

Architecture — True Per-Client Local Stores (FedRAG Paper)
-----------------------------------------------------------
Each SimClient owns a PRIVATE, NON-REPLICATED local InMemoryKnowledgeStore
(its ``nodes`` list).  Data never leaves a client unless it explicitly
sends a response back to the central aggregator.

Query flow (matching the FedRAG architectural table):
  1. Server receives a query.
  2. Server fans out the query to EVERY active (non-dropped, non-quarantined) client.
  3. Each client runs the query against ITS OWN LOCAL knowledge store and
     returns its best (score, node) pair.
  4. Server selects the globally highest-scoring answer from all per-client responses.

Attack flow (DRAG-style attack translated to FedRAG):
  1. N out of total M clients are designated as malicious (e.g. 50 of 150).
  2. Each malicious client poisons ITS OWN LOCAL store independently — other
     clients' stores are completely unaffected at this stage.
  3. The central server aggregates answers from ALL M clients (clean + poisoned)
     during evaluation — exactly like a real federated deployment.
  4. Server-side defense inspects per-client data and may quarantine individual
     malicious clients before the final evaluation round.

Retrieval uses deterministic hash-cosine similarity — no ML dependencies needed.
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
    """One knowledge-store node (a single (query, answer) pair in a client's local store)."""

    node_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimClient:
    """One federated client — owns a PRIVATE local knowledge store.

    ``examples``   : raw (query, response, topic) dicts — the client's local dataset shard.
    ``nodes``      : the client's local InMemoryKnowledgeStore built from examples.
    ``poisoned``   : True if this client's local store has been attacked.
    ``quarantined``: True if the server-side defense has quarantined this client.
    """

    client_id: int
    examples: list[dict[str, Any]]
    nodes: list[SimNode] = field(default_factory=list)
    poisoned: bool = False
    quarantined: bool = False

    @property
    def data_size(self) -> int:
        return len(self.examples)


# ---------------------------------------------------------------------------
# Hashing retriever (no ML dependencies)
# ---------------------------------------------------------------------------


def _hash_embedding(text: str, dim: int = 32) -> list[float]:
    """Produce a deterministic unit-norm embedding from text via SHA-256."""
    digest = hashlib.sha256(text.encode()).digest()
    floats: list[float] = []
    for i in range(dim):
        byte = digest[i % len(digest)]
        floats.append((byte / 127.5) - 1.0)
    norm = math.sqrt(sum(v * v for v in floats)) or 1.0
    return [v / norm for v in floats]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _build_nodes(examples: list[dict[str, Any]], embedding_dim: int = 32) -> list[SimNode]:
    """Build SimNodes from raw examples — creates the client's local store contents."""
    nodes: list[SimNode] = []
    for row in examples:
        query = row.get("query", "")
        emb = _hash_embedding(query, embedding_dim)
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


# ---------------------------------------------------------------------------
# FedRAGSimulator
# ---------------------------------------------------------------------------


TOPICS = [
    "cryptography", "medicine", "finance", "biology", "law",
    "networking", "history", "physics", "privacy", "safety",
    "databases", "governance",
]


class FedRAGSimulator:
    """Simulate a federated RAG deployment with true per-client local stores.

    Each client holds a PRIVATE, NON-REPLICATED local InMemoryKnowledgeStore.
    Queries are answered by fanning out to every active client's local store
    and aggregating the best result at the server — matching the FedRAG paper.

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
    # Per-client local store retrieval
    # ------------------------------------------------------------------

    def retrieve_from_client(
        self,
        query: str,
        client_id: int,
        top_k: int = 1,
        score_mask: float | None = None,
    ) -> list[tuple[float, SimNode]]:
        """Query ONE client's LOCAL knowledge store.

        The client searches only its own private store.  No other client's
        data is touched.  This is the fundamental per-client query unit in
        the FedRAG architecture.

        Parameters
        ----------
        query:
            The query string.
        client_id:
            Which client's local store to query.
        top_k:
            Number of results to return from this client's store.
        score_mask:
            If set, replace actual scores with this fixed value (score
            masking defense — hides retrieval confidence from the server).
        """
        client = self.clients[client_id]
        if client.quarantined or not client.nodes:
            return []
        q_emb = _hash_embedding(query, self.embedding_dim)
        scored = [(float(_cosine(q_emb, n.embedding)), n) for n in client.nodes]
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:top_k]
        if score_mask is not None:
            top = [(score_mask, n) for _, n in top]
        return top

    # ------------------------------------------------------------------
    # Server-side federated aggregation
    # ------------------------------------------------------------------

    def federated_retrieve(
        self,
        query: str,
        top_k: int = 1,
        dropped_clients: set[int] | None = None,
        byzantine_clients: set[int] | None = None,
        score_mask: float | None = None,
    ) -> list[tuple[float, SimNode, int]]:
        """True federated retrieval — server fans out to each client's LOCAL store.

        Flow:
          1. Server sends query to every active client.
          2. Each client queries ITS OWN local store via retrieve_from_client().
          3. Server collects one best-result per client.
          4. Server returns the globally top-k answers across all client responses.

        Parameters
        ----------
        query:
            The query string.
        top_k:
            Number of globally best results to return.
        dropped_clients:
            Client IDs that are offline (node availability attack).
        byzantine_clients:
            Client IDs whose answers should be treated as corrupted.
        score_mask:
            Score masking defense value (applied at each client's retrieval).

        Returns
        -------
        List of (score, node, client_id) tuples, globally sorted best-first.
        """
        dropped = dropped_clients or set()
        byzantine = byzantine_clients or set()

        # Fan out: every active client answers from its OWN local store
        per_client_best: list[tuple[float, SimNode, int]] = []
        for client in self.clients:
            if client.client_id in dropped or client.quarantined:
                continue
            hits = self.retrieve_from_client(
                query, client.client_id, top_k=1, score_mask=score_mask
            )
            for score, node in hits:
                if client.client_id in byzantine:
                    corrupted = SimNode(
                        node_id=node.node_id,
                        text=node.text,
                        embedding=node.embedding,
                        metadata={
                            **node.metadata,
                            "answer": f"[BYZANTINE-CORRUPT] {node.metadata.get('answer', '')}",
                        },
                    )
                    per_client_best.append((score, corrupted, client.client_id))
                else:
                    per_client_best.append((score, node, client.client_id))

        # Server aggregates: pick globally best answers across all client responses
        per_client_best.sort(key=lambda t: t[0], reverse=True)
        return per_client_best[:top_k]

    def retrieve(
        self,
        query: str,
        client_id: int | None = None,
        top_k: int = 1,
        score_mask: float | None = None,
    ) -> list[tuple[float, SimNode]]:
        """Retrieve top-k nodes for *query*.

        ``client_id`` given → query that ONE client's LOCAL store only.
        ``client_id=None``  → TRUE FEDERATED fan-out: query each client's
                              LOCAL store individually, then aggregate at the
                              server.  This is NOT data pooling — each client's
                              store remains private and isolated.
        """
        if client_id is not None:
            return self.retrieve_from_client(query, client_id, top_k, score_mask)

        # True federated fan-out + server aggregation
        results_with_cid = self.federated_retrieve(query, top_k=top_k, score_mask=score_mask)
        return [(score, node) for score, node, _ in results_with_cid]

    def get_answer(self, query: str, client_id: int | None = None) -> str:
        """Retrieve the best-matching answer for *query*."""
        results = self.retrieve(query, client_id=client_id)
        if not results:
            return ""
        _, node = results[0]
        return str(node.metadata.get("answer", ""))

    # ------------------------------------------------------------------
    # Store management
    # ------------------------------------------------------------------

    def rebuild_stores(self) -> None:
        """Rebuild EVERY client's local knowledge store from its current examples.

        Call after modifying any client's examples (e.g. after an attack).
        Each client rebuilds only its own private store independently.
        """
        for client in self.clients:
            client.nodes = _build_nodes(client.examples, self.embedding_dim)

    def rebuild_client_store(self, client_id: int) -> None:
        """Rebuild ONE client's local knowledge store.

        Use for targeted attacks: only the attacked client's store changes.
        Other clients' stores are completely unaffected.
        """
        client = self.clients[client_id]
        client.nodes = _build_nodes(client.examples, self.embedding_dim)

    def quarantine_client(self, client_id: int) -> None:
        """Server-side defense: mark a client as quarantined (excluded from federation)."""
        self.clients[client_id].quarantined = True

    def unquarantine_all(self) -> None:
        """Clear all quarantine flags."""
        for client in self.clients:
            client.quarantined = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise simulator metadata."""
        return {
            "num_clients": self.num_clients,
            "num_examples": self.num_examples,
            "seed": self.seed,
            "embedding_dim": self.embedding_dim,
            "client_sizes": [c.data_size for c in self.clients],
            "topics": TOPICS,
            "architecture": "true-per-client-local-stores",
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
        """Split dataset IID across clients; each client builds its own private store."""
        shuffled = list(self.dataset)
        self._rng.shuffle(shuffled)
        base, remainder = divmod(len(shuffled), self.num_clients)
        clients: list[SimClient] = []
        offset = 0
        for cid in range(self.num_clients):
            size = base + (1 if cid < remainder else 0)
            examples = shuffled[offset: offset + size]
            offset += size
            # Each client builds its OWN private local store from its IID shard
            clients.append(SimClient(
                client_id=cid,
                examples=examples,
                nodes=_build_nodes(examples, self.embedding_dim),
            ))
        return clients
