"""
fedrag_client.py — True per-client InMemoryKnowledgeStore for FedRAG.
Each client holds its own private store; the server only aggregates results.
Matches the FedRAG paper architecture (Table 1: private per-client, non-replicated).
"""
from __future__ import annotations
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Shared embedding helper (same as DRAG) ───────────────────────────────────
def hash_embed(text: str, dim: int = 32) -> np.ndarray:
    """SHA-256 hash → deterministic float32 vector (no sentence-transformer needed)."""
    raw = hashlib.sha256(text.encode()).digest()
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    # tile/trim to dim, then L2-normalise
    reps = (dim // len(arr)) + 1
    vec = np.tile(arr, reps)[:dim]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ── Per-client knowledge store ────────────────────────────────────────────────
@dataclass
class KnowledgeRecord:
    query: str
    embedding: np.ndarray
    answer: str
    poisoned: bool = False


class InMemoryKnowledgeStore:
    """Private, non-replicated store.  One instance per client — never shared."""

    def __init__(self):
        self._records: list[KnowledgeRecord] = []

    def add(self, query: str, embedding: np.ndarray, answer: str, poisoned: bool = False):
        self._records.append(KnowledgeRecord(query, embedding, answer, poisoned))

    def search(self, query_embedding: np.ndarray, top_k: int = 1) -> list[dict]:
        if not self._records:
            return []
        scores = [
            float(np.dot(query_embedding, r.embedding))
            for r in self._records
        ]
        ranked = sorted(zip(scores, self._records), key=lambda x: -x[0])
        return [
            {"score": s, "answer": r.answer, "poisoned": r.poisoned}
            for s, r in ranked[:top_k]
        ]

    def __len__(self):
        return len(self._records)

    def poison_fraction(self, ratio: float, wrong_answer_fn=None):
        """Poison `ratio` fraction of this store's records in-place."""
        n_poison = int(len(self._records) * ratio)
        for rec in self._records[:n_poison]:
            rec.poisoned = True
            if wrong_answer_fn:
                rec.answer = wrong_answer_fn(rec.answer)
            else:
                rec.answer = "POISONED_" + rec.answer

    def restore_clean(self):
        """Undo poisoning (for defense eval)."""
        for rec in self._records:
            if rec.poisoned:
                rec.poisoned = False
                if rec.answer.startswith("POISONED_"):
                    rec.answer = rec.answer[len("POISONED_"):]


# ── FedRAG Client ─────────────────────────────────────────────────────────────
class FedRAGClient:
    """
    One node in the star topology.
    Holds its own private InMemoryKnowledgeStore.
    The server fans out queries; clients NEVER communicate with each other.
    """

    def __init__(self, client_id: int, local_data: list[dict], embed_dim: int = 32):
        self.client_id = client_id
        self.quarantined = False
        self.store = InMemoryKnowledgeStore()
        for item in local_data:
            emb = item.get("embedding") or hash_embed(item["query"], embed_dim)
            self.store.add(item["query"], emb, item["answer"])

    # ------------------------------------------------------------------
    def local_query(self, query_embedding: np.ndarray, top_k: int = 1) -> list[dict]:
        """Run retrieval locally.  Returns [] if quarantined."""
        if self.quarantined:
            return []
        results = self.store.search(query_embedding, top_k=top_k)
        for r in results:
            r["source_client"] = self.client_id
        return results

    # ------------------------------------------------------------------
    def poison_store(self, ratio: float):
        self.store.poison_fraction(ratio)

    def restore_store(self):
        self.store.restore_clean()

    # ------------------------------------------------------------------
    @property
    def poison_density(self) -> float:
        if len(self.store) == 0:
            return 0.0
        return sum(1 for r in self.store._records if r.poisoned) / len(self.store)


# ── FedRAG Server ─────────────────────────────────────────────────────────────
class FedRAGServer:
    """
    Central server — star topology hub.
    Fans out every query to ALL non-quarantined clients and aggregates top-1.
    """

    def __init__(self, clients: list[FedRAGClient]):
        self.clients = clients

    # ------------------------------------------------------------------
    def federated_query(self, query_embedding: np.ndarray) -> Optional[str]:
        """Fan-out → aggregate.  O(n_clients) messages per query."""
        all_results = []
        for client in self.clients:
            hits = client.local_query(query_embedding, top_k=1)
            all_results.extend(hits)

        if not all_results:
            return None
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[0]["answer"]

    # ------------------------------------------------------------------
    def quarantine_by_poison_density(self, threshold: float = 0.35) -> list[int]:
        """
        Defense: quarantine any client whose local poison density exceeds threshold.
        Server can compute this because it owns the fan-out channel (can inspect markers).
        """
        quarantined = []
        for client in self.clients:
            if client.poison_density >= threshold:
                client.quarantined = True
                quarantined.append(client.client_id)
        return quarantined

    # ------------------------------------------------------------------
    def evaluate(self, queries: list[dict]) -> dict:
        """
        Evaluate over `queries` list of {"query": str, "answer": str}.
        Returns {"f1": float, "em": float, "coverage": float, "poisoned_rate": float}.
        """
        correct = answered = poisoned_answers = 0
        for q in queries:
            emb = hash_embed(q["query"])
            pred = self.federated_query(emb)
            if pred is None:
                continue
            answered += 1
            if pred.startswith("POISONED_"):
                poisoned_answers += 1
            gold = q["answer"]
            if pred == gold or pred == "POISONED_" + gold:
                correct += 1

        total = len(queries)
        coverage = answered / total if total else 0.0
        f1 = correct / answered if answered else 0.0
        poisoned_rate = poisoned_answers / answered if answered else 0.0
        return {
            "f1": f1,
            "em": f1,          # exact match == F1 for single-token answers
            "coverage": coverage,
            "poisoned_rate": poisoned_rate,
            "answered": answered,
            "total": total,
        }


# ── Replication defense (Fix 3) ───────────────────────────────────────────────
def replicate_stores(clients: list, k: int = 2, seed: int = 0) -> None:
    """
    Each record is stored on k clients instead of 1.
    Defense against node availability attacks — losing 1 client only removes
    1/k copies, so the answer can still be retrieved from surviving replicas.

    k=1 → no replication (default, current behavior)
    k=2 → each item on 2 clients (halves availability loss)
    k=3 → each item on 3 clients (reduces loss to ~1/3)
    """
    import random
    rng = random.Random(seed)
    n = len(clients)
    for i, client in enumerate(clients):
        for rec in client.store._records:
            # Pick k-1 other clients to receive a copy
            others = rng.sample([c for c in clients if c.client_id != i], min(k - 1, n - 1))
            for target in others:
                target.store.add(rec.query, rec.embedding, rec.answer, rec.poisoned)
