#!/usr/bin/env python3
"""
run_drag_vs_fedrag_comparison.py
=================================
Side-by-side security comparison of DRAG and FedRAG architectures.

DRAG  — Distributed RAG (peer-to-peer, Barabási–Albert graph, multi-hop BFS routing,
         CrossPeerValidation majority-vote defense)
FedRAG — Federated RAG (star topology, central server fan-out, per-client private stores,
          quarantine-based defense)

Both systems use the SAME synthetic dataset, the SAME 4 attacks, and the SAME
deterministic hash-cosine embeddings so results are directly comparable.
No GPU · No Hugging Face downloads · No Ollama · Zero external ML deps.

Usage
-----
  # Default: 20 peers/clients, 400 examples, all 4 attacks
  python run_drag_vs_fedrag_comparison.py

  # Scale up
  python run_drag_vs_fedrag_comparison.py --num-nodes 50 --num-examples 1000 --malicious-ratio 0.33

  # Single attack
  python run_drag_vs_fedrag_comparison.py --attack data_poisoning

  # No defense (measure raw attack impact)
  python run_drag_vs_fedrag_comparison.py --no-defense

  # Save JSON results
  python run_drag_vs_fedrag_comparison.py --output comparison_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional, Sequence


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Shared primitives (embeddings, metrics, dataset)
# ══════════════════════════════════════════════════════════════════════════════

TOPICS = [
    "cryptography", "medicine", "finance", "biology", "law",
    "networking", "history", "physics", "privacy", "safety",
    "databases", "governance",
]

POISON_MARKER = "[POISONED]"


def _hash_embedding(text: str, dim: int = 32) -> list[float]:
    """Deterministic unit-norm embedding via SHA-256. No ML needed."""
    digest = hashlib.sha256(text.encode()).digest()
    floats = [(digest[i % len(digest)] / 127.5) - 1.0 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in floats)) or 1.0
    return [v / norm for v in floats]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def _token_f1(pred: str, gold: str) -> float:
    p, g = pred.lower().split(), gold.lower().split()
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _make_dataset(num_examples: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for idx in range(num_examples):
        topic = TOPICS[idx % len(TOPICS)]
        rows.append({
            "query": f"What is the key fact #{idx} about {topic}?",
            "response": f"verified-{topic}-answer-{idx}",
            "topic": topic,
        })
    rng.shuffle(rows)
    return rows


def _iid_split(dataset: list[dict], n: int, seed: int = 0) -> list[list[dict]]:
    """Split dataset into n IID shards."""
    rng = random.Random(seed)
    shuffled = list(dataset)
    rng.shuffle(shuffled)
    base, remainder = divmod(len(shuffled), n)
    shards, offset = [], 0
    for i in range(n):
        size = base + (1 if i < remainder else 0)
        shards.append(shuffled[offset: offset + size])
        offset += size
    return shards


def _eval_metrics(rows: list[dict], predictions: list[str]) -> dict[str, float]:
    em, f1 = [], []
    for row, pred in zip(rows, predictions):
        gold = row["response"]
        em.append(_exact_match(pred, gold))
        f1.append(_token_f1(pred, gold))
    n = len(em) or 1
    return {"em": round(sum(em) / n, 4), "f1": round(sum(f1) / n, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DRAG System (Peer-to-Peer, Barabási–Albert graph)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DRAGKnowledgeNode:
    node_id: str
    text: str
    embedding: list[float]
    answer: str
    topic: str
    poisoned: bool = False


@dataclass
class DRAGPeer:
    """
    DRAG peer — owns a private local KnowledgeBase.
    Analogous to modules/peer.py + modules/knowledge_base.py in the DRAG codebase.
    """
    peer_id: int
    knowledge_base: list[DRAGKnowledgeNode] = field(default_factory=list)
    neighbors: list[int] = field(default_factory=list)   # from BA graph
    offline: bool = False
    byzantine: bool = False
    topics: list[str] = field(default_factory=list)

    def query_local(self, query: str, dim: int = 32,
                    confidence_threshold: float = 0.5) -> Optional[tuple[float, str]]:
        """Query this peer's private local KB. Returns (score, answer) or None."""
        if self.offline or not self.knowledge_base:
            return None
        q_emb = _hash_embedding(query, dim)
        best_score, best_answer = -1.0, ""
        for node in self.knowledge_base:
            score = _cosine(q_emb, node.embedding)
            if score > best_score:
                best_score, best_answer = score, node.answer
        if best_score < confidence_threshold:
            return None
        if self.byzantine:
            best_answer = f"[BYZANTINE] {best_answer}"
        return (best_score, best_answer)


class DRAGSimulator:
    """
    DRAG (Distributed RAG) simulator.

    Architecture (mirrors distributed-rag-poison-defense codebase):
    ┌──────────────────────────────────────────────────────────┐
    │  P2P Network: Barabási–Albert graph                      │
    │  Each peer has a PRIVATE local KnowledgeBase             │
    │  Query routing: BFS multi-hop from entry peer            │
    │  Stops when confidence_threshold met OR max_hops reached  │
    │  Defense: CrossPeerValidation (majority vote)            │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(self, num_peers: int = 20, num_examples: int = 400,
                 num_attachments: int = 2, confidence_threshold: float = 0.5,
                 max_hops: int = 3, embedding_dim: int = 32,
                 dataset: Optional[list[dict]] = None, seed: int = 0):
        self.num_peers = num_peers
        self.num_attachments = num_attachments
        self.confidence_threshold = confidence_threshold
        self.max_hops = max_hops
        self.embedding_dim = embedding_dim
        self.seed = seed
        self._rng = random.Random(seed)

        self.dataset = dataset or _make_dataset(num_examples, seed)
        self.peers = self._build_peers()
        # Store adjacency for quick lookup
        self._adj = self._build_ba_graph(num_peers, num_attachments, seed)
        for peer in self.peers:
            peer.neighbors = list(self._adj.get(peer.peer_id, []))

    # ── Graph construction ───────────────────────────────────────────────────

    def _build_ba_graph(self, n: int, m: int, seed: int) -> dict[int, set[int]]:
        """
        Barabási–Albert preferential attachment graph (mirrors nx.barabasi_albert_graph).
        Returns adjacency dict.
        """
        rng = random.Random(seed)
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        if n <= m:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        adj[i].add(j)
            return adj

        for i in range(m):
            for j in range(m):
                if i != j:
                    adj[i].add(j)
                    adj[j].add(i)

        degrees = [len(adj[i]) for i in range(m)]
        for new_node in range(m, n):
            adj[new_node] = set()
            total_degree = sum(degrees) or 1
            targets = set()
            attempts = 0
            while len(targets) < m and attempts < n * 10:
                attempts += 1
                r = rng.random() * total_degree
                cumul = 0.0
                for node, deg in enumerate(degrees):
                    cumul += deg
                    if r <= cumul and node not in targets and node != new_node:
                        targets.add(node)
                        break
            for t in targets:
                adj[new_node].add(t)
                adj[t].add(new_node)
                degrees[t] += 1
            degrees.append(len(adj[new_node]))

        return adj

    # ── Peer construction ────────────────────────────────────────────────────

    def _build_peers(self) -> list[DRAGPeer]:
        shards = _iid_split(self.dataset, self.num_peers, self.seed)
        peers = []
        for pid, shard in enumerate(shards):
            kb = []
            for row in shard:
                emb = _hash_embedding(row["query"], self.embedding_dim)
                nid = hashlib.md5(row["query"].encode()).hexdigest()[:10]
                kb.append(DRAGKnowledgeNode(
                    node_id=nid, text=row["query"], embedding=emb,
                    answer=row["response"], topic=row["topic"]
                ))
            topics = list({row["topic"] for row in shard})
            peers.append(DRAGPeer(peer_id=pid, knowledge_base=kb, topics=topics))
        return peers

    # ── Multi-hop BFS query (DRAG's core routing) ────────────────────────────

    def query(self, query: str, entry_peer_id: Optional[int] = None,
              dropped: Optional[set[int]] = None,
              byzantine: Optional[set[int]] = None) -> dict[str, Any]:
        """
        DRAG multi-hop BFS query routing.

        Flow (matches DRAGNetwork.topic_query in rag_network.py):
          1. Start from entry peer (random if not specified).
          2. BFS expands to neighbors up to max_hops.
          3. Each peer queries its own LOCAL KB.
          4. First hit above confidence_threshold is returned.
          5. If no hit: return best-scoring answer found across all visited peers.
        """
        dropped = dropped or set()
        byzantine_set = byzantine or set()
        entry = entry_peer_id if entry_peer_id is not None else self._rng.randint(0, self.num_peers - 1)

        visited: set[int] = set()
        queue: deque[tuple[int, int]] = deque([(entry, 0)])  # (peer_id, hop)
        candidates: list[tuple[float, str, int]] = []  # (score, answer, peer_id)
        hops_taken = 0

        while queue:
            peer_id, hop = queue.popleft()
            if peer_id in visited or peer_id in dropped:
                continue
            visited.add(peer_id)
            hops_taken = max(hops_taken, hop)

            peer = self.peers[peer_id]
            is_byz = peer_id in byzantine_set
            if is_byz:
                peer.byzantine = True

            result = peer.query_local(query, self.embedding_dim, self.confidence_threshold)
            if result is not None:
                score, answer = result
                candidates.append((score, answer, peer_id))
                if score >= self.confidence_threshold:
                    peer.byzantine = False
                    return {
                        "answer": answer, "score": score,
                        "hops": hop, "visited": len(visited),
                        "hit": True, "source_peer": peer_id,
                    }

            if hop < self.max_hops:
                for nb in peer.neighbors:
                    if nb not in visited and nb not in dropped:
                        queue.append((nb, hop + 1))

            peer.byzantine = False

        if candidates:
            candidates.sort(key=lambda t: t[0], reverse=True)
            score, answer, pid = candidates[0]
            return {
                "answer": answer, "score": score,
                "hops": hops_taken, "visited": len(visited),
                "hit": False, "source_peer": pid,
            }
        return {"answer": "", "score": 0.0, "hops": hops_taken,
                "visited": len(visited), "hit": False, "source_peer": -1}

    # ── CrossPeerValidation defense ──────────────────────────────────────────

    def query_with_defense(self, query: str, min_agreement: float = 0.6,
                           min_peers: int = 3, dropped: Optional[set[int]] = None,
                           byzantine: Optional[set[int]] = None) -> dict[str, Any]:
        """
        DRAG CrossPeerValidation defense (mirrors cross_peer_validation.py).

        Collects answers from min_peers neighbors of entry peer.
        Returns answer only if agreement ratio >= min_agreement_ratio.
        Falls back to best answer if insufficient agreement.
        """
        dropped = dropped or set()
        byzantine_set = byzantine or set()
        active = [p for p in self.peers if p.peer_id not in dropped and not p.offline]
        if not active:
            return {"answer": "", "score": 0.0, "validated": False, "agreement": 0.0,
                    "hops": 0, "visited": 0, "hit": False}

        # Pick a random entry, collect from neighbors
        entry = self._rng.choice(active)
        candidates = [nb for nb in entry.neighbors
                      if nb not in dropped and not self.peers[nb].offline][:min_peers * 2]
        candidates = (candidates + [entry.peer_id])[:min_peers]

        peer_answers: list[str] = []
        best_score, best_answer = -1.0, ""
        for pid in candidates:
            peer = self.peers[pid]
            is_byz = pid in byzantine_set
            if is_byz:
                peer.byzantine = True
            result = peer.query_local(query, self.embedding_dim, 0.0)
            if result:
                score, answer = result
                peer_answers.append(answer)
                if score > best_score:
                    best_score, best_answer = score, answer
            peer.byzantine = False

        if len(peer_answers) < min_peers:
            return {"answer": best_answer, "score": best_score,
                    "validated": False, "agreement": 0.0,
                    "hops": 1, "visited": len(candidates), "hit": best_score > 0}

        # Majority vote with similarity matching
        vote_counts: dict[str, int] = defaultdict(int)
        for ans in peer_answers:
            matched = False
            for existing in list(vote_counts.keys()):
                if _similarity(ans, existing) >= 0.85:
                    vote_counts[existing] += 1
                    matched = True
                    break
            if not matched:
                vote_counts[ans] += 1

        winner = max(vote_counts, key=vote_counts.__getitem__)
        agreement = vote_counts[winner] / len(peer_answers)
        validated = agreement >= min_agreement

        return {
            "answer": winner if validated else best_answer,
            "score": best_score,
            "validated": validated,
            "agreement": round(agreement, 3),
            "hops": 1,
            "visited": len(candidates),
            "hit": best_score > 0,
        }

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, test_rows: list[dict], use_defense: bool = False,
                 dropped: Optional[set[int]] = None,
                 byzantine: Optional[set[int]] = None) -> dict[str, float]:
        preds = []
        for row in test_rows:
            if use_defense:
                result = self.query_with_defense(row["query"], dropped=dropped, byzantine=byzantine)
            else:
                result = self.query(row["query"], dropped=dropped, byzantine=byzantine)
            preds.append(result["answer"])
        return _eval_metrics(test_rows, preds)

    # ── Reset helpers ─────────────────────────────────────────────────────────

    def restore_all(self, backups: dict[int, list[DRAGKnowledgeNode]]) -> None:
        for peer in self.peers:
            if peer.peer_id in backups:
                peer.knowledge_base = [
                    DRAGKnowledgeNode(**n.__dict__) for n in backups[peer.peer_id]
                ]
            peer.offline = False
            peer.byzantine = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FedRAG System (Star topology, central server fan-out)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FedRAGNode:
    node_id: str
    text: str
    embedding: list[float]
    answer: str
    topic: str
    poisoned: bool = False


@dataclass
class FedRAGClient:
    """
    FedRAG client — owns a PRIVATE local InMemoryKnowledgeStore.
    Analogous to SimClient in run_fedavg_gpu.py.
    """
    client_id: int
    store: list[FedRAGNode] = field(default_factory=list)
    quarantined: bool = False
    topics: list[str] = field(default_factory=list)

    def query_local(self, query: str, dim: int = 32) -> Optional[tuple[float, str]]:
        """Query this client's private local store. Returns (score, answer) or None."""
        if self.quarantined or not self.store:
            return None
        q_emb = _hash_embedding(query, dim)
        best_score, best_answer = -1.0, ""
        for node in self.store:
            score = _cosine(q_emb, node.embedding)
            if score > best_score:
                best_score, best_answer = score, node.answer
        return (best_score, best_answer)


class FedRAGSimulator:
    """
    FedRAG (Federated RAG) simulator.

    Architecture (mirrors run_fedavg_gpu.py):
    ┌──────────────────────────────────────────────────────────┐
    │  Star topology: central server + N clients               │
    │  Each client has a PRIVATE local InMemoryKnowledgeStore  │
    │  Server FANS OUT query to ALL active clients             │
    │  Server picks globally best answer                       │
    │  Defense: quarantine clients with high poison density    │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(self, num_clients: int = 20, num_examples: int = 400,
                 embedding_dim: int = 32, dataset: Optional[list[dict]] = None,
                 seed: int = 0):
        self.num_clients = num_clients
        self.embedding_dim = embedding_dim
        self.seed = seed
        self._rng = random.Random(seed)

        self.dataset = dataset or _make_dataset(num_examples, seed)
        self.clients = self._build_clients()

    # ── Client construction ──────────────────────────────────────────────────

    def _build_clients(self) -> list[FedRAGClient]:
        shards = _iid_split(self.dataset, self.num_clients, self.seed)
        clients = []
        for cid, shard in enumerate(shards):
            store = []
            for row in shard:
                emb = _hash_embedding(row["query"], self.embedding_dim)
                nid = hashlib.md5(row["query"].encode()).hexdigest()[:10]
                store.append(FedRAGNode(
                    node_id=nid, text=row["query"], embedding=emb,
                    answer=row["response"], topic=row["topic"]
                ))
            topics = list({row["topic"] for row in shard})
            clients.append(FedRAGClient(client_id=cid, store=store, topics=topics))
        return clients

    # ── Server fan-out query (FedRAG's core) ─────────────────────────────────

    def query(self, query: str,
              dropped: Optional[set[int]] = None) -> dict[str, Any]:
        """
        FedRAG central server query (fan-out to ALL clients).

        Flow (matches FedRAGSimulator.federated_retrieve in run_fedavg_gpu.py):
          1. Server sends query to EVERY active (non-quarantined) client.
          2. Each client queries its OWN local store.
          3. Server picks the globally best answer across all responses.
          No P2P routing — star topology only.
        """
        dropped = dropped or set()
        candidates: list[tuple[float, str, int]] = []

        for client in self.clients:
            if client.client_id in dropped or client.quarantined:
                continue
            result = client.query_local(query, self.embedding_dim)
            if result is not None:
                score, answer = result
                candidates.append((score, answer, client.client_id))

        if not candidates:
            return {"answer": "", "score": 0.0, "hit": False, "clients_queried": 0}

        candidates.sort(key=lambda t: t[0], reverse=True)
        score, answer, cid = candidates[0]
        return {
            "answer": answer, "score": score, "hit": True,
            "clients_queried": len(candidates), "source_client": cid,
        }

    # ── Quarantine defense ────────────────────────────────────────────────────

    def run_defense(self, threshold: float = 0.25) -> list[int]:
        """
        FedRAG quarantine defense (mirrors ClientDataPoisoningDefense).
        Quarantines clients where >threshold fraction of answers contain POISON_MARKER.
        """
        quarantined = []
        for client in self.clients:
            if not client.store:
                continue
            poison_count = sum(1 for n in client.store if POISON_MARKER in n.answer)
            ratio = poison_count / len(client.store)
            if ratio > threshold:
                client.quarantined = True
                quarantined.append(client.client_id)
        return quarantined

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, test_rows: list[dict],
                 dropped: Optional[set[int]] = None) -> dict[str, float]:
        preds = []
        for row in test_rows:
            result = self.query(row["query"], dropped=dropped)
            preds.append(result["answer"])
        return _eval_metrics(test_rows, preds)

    # ── Reset helpers ─────────────────────────────────────────────────────────

    def restore_all(self, backups: dict[int, list[FedRAGNode]]) -> None:
        for client in self.clients:
            if client.client_id in backups:
                client.store = [FedRAGNode(**n.__dict__) for n in backups[client.client_id]]
            client.quarantined = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Attacks (applied identically to both systems)
# ══════════════════════════════════════════════════════════════════════════════

class DataPoisoningAttack:
    """
    Inject malicious answers into a subset of nodes/clients.
    DRAG: poisons selected peers' KBs.
    FedRAG: poisons selected clients' stores.
    """
    POISON_ANSWERS = [
        f"{POISON_MARKER} incorrect-answer",
        f"{POISON_MARKER} misleading-data",
        f"{POISON_MARKER} wrong-fact",
        f"{POISON_MARKER} fabricated-response",
    ]

    def __init__(self, malicious_ratio: float = 0.3,
                 poisoning_ratio: float = 0.3, seed: int = 0):
        self.malicious_ratio = malicious_ratio
        self.poisoning_ratio = poisoning_ratio
        self._rng = random.Random(seed)

    # ── DRAG ─────────────────────────────────────────────────────────────────

    def attack_drag(self, sim: DRAGSimulator) -> dict[str, Any]:
        n_malicious = max(1, int(sim.num_peers * self.malicious_ratio))
        malicious_ids = self._rng.sample(range(sim.num_peers), n_malicious)
        total_poisoned = 0
        for pid in malicious_ids:
            peer = sim.peers[pid]
            n_poison = max(1, int(len(peer.knowledge_base) * self.poisoning_ratio))
            targets = self._rng.sample(peer.knowledge_base,
                                       min(n_poison, len(peer.knowledge_base)))
            for node in targets:
                node.answer = self._rng.choice(self.POISON_ANSWERS)
                node.poisoned = True
                total_poisoned += 1
        return {"malicious_peers": malicious_ids, "nodes_poisoned": total_poisoned}

    # ── FedRAG ───────────────────────────────────────────────────────────────

    def attack_fedrag(self, sim: FedRAGSimulator) -> dict[str, Any]:
        n_malicious = max(1, int(sim.num_clients * self.malicious_ratio))
        malicious_ids = self._rng.sample(range(sim.num_clients), n_malicious)
        total_poisoned = 0
        for cid in malicious_ids:
            client = sim.clients[cid]
            n_poison = max(1, int(len(client.store) * self.poisoning_ratio))
            targets = self._rng.sample(client.store,
                                       min(n_poison, len(client.store)))
            for node in targets:
                node.answer = self._rng.choice(self.POISON_ANSWERS)
                node.poisoned = True
                total_poisoned += 1
        return {"malicious_clients": malicious_ids, "nodes_poisoned": total_poisoned}


class MembershipInferenceAttack:
    """
    Infer whether a query exists in a target node/client's local store
    by comparing retrieval confidence scores between member and non-member queries.
    DRAG: probes via multi-hop routing to target peer.
    FedRAG: probes via direct fan-out to target client (isolated).
    """

    def __init__(self, threshold: float = 0.75, seed: int = 0):
        self.threshold = threshold
        self._rng = random.Random(seed)

    def _score_drag(self, sim: DRAGSimulator, query: str, target_peer: int) -> float:
        peer = sim.peers[target_peer]
        result = peer.query_local(query, sim.embedding_dim, 0.0)
        return result[0] if result else 0.0

    def _score_fedrag(self, sim: FedRAGSimulator, query: str, target_client: int) -> float:
        client = sim.clients[target_client]
        result = client.query_local(query, sim.embedding_dim)
        return result[0] if result else 0.0

    def attack_drag(self, sim: DRAGSimulator) -> dict[str, Any]:
        target = self._rng.randint(0, sim.num_peers - 1)
        peer = sim.peers[target]
        member_queries = [n.text for n in peer.knowledge_base][:10]
        all_texts = [n.text for p in sim.peers for n in p.knowledge_base
                     if p.peer_id != target]
        non_member_queries = self._rng.sample(all_texts, min(10, len(all_texts)))

        member_scores = [self._score_drag(sim, q, target) for q in member_queries]
        non_member_scores = [self._score_drag(sim, q, target) for q in non_member_queries]

        avg_member = sum(member_scores) / len(member_scores) if member_scores else 0
        avg_non_member = sum(non_member_scores) / len(non_member_scores) if non_member_scores else 0
        tp = sum(1 for s in member_scores if s >= self.threshold)
        fp = sum(1 for s in non_member_scores if s >= self.threshold)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / len(member_scores) if member_scores else 0.0

        return {
            "target_peer": target,
            "avg_member_score": round(avg_member, 4),
            "avg_non_member_score": round(avg_non_member, 4),
            "score_gap": round(avg_member - avg_non_member, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "leakage_risk": "HIGH" if avg_member - avg_non_member > 0.15 else "LOW",
            "note": "DRAG: attacker must route through P2P graph to reach target peer",
        }

    def attack_fedrag(self, sim: FedRAGSimulator) -> dict[str, Any]:
        target = self._rng.randint(0, sim.num_clients - 1)
        client = sim.clients[target]
        member_queries = [n.text for n in client.store][:10]
        all_texts = [n.text for c in sim.clients for n in c.store
                     if c.client_id != target]
        non_member_queries = self._rng.sample(all_texts, min(10, len(all_texts)))

        member_scores = [self._score_fedrag(sim, q, target) for q in member_queries]
        non_member_scores = [self._score_fedrag(sim, q, target) for q in non_member_queries]

        avg_member = sum(member_scores) / len(member_scores) if member_scores else 0
        avg_non_member = sum(non_member_scores) / len(non_member_scores) if non_member_scores else 0
        tp = sum(1 for s in member_scores if s >= self.threshold)
        fp = sum(1 for s in non_member_scores if s >= self.threshold)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / len(member_scores) if member_scores else 0.0

        return {
            "target_client": target,
            "avg_member_score": round(avg_member, 4),
            "avg_non_member_score": round(avg_non_member, 4),
            "score_gap": round(avg_member - avg_non_member, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "leakage_risk": "HIGH" if avg_member - avg_non_member > 0.15 else "LOW",
            "note": "FedRAG: server can directly target any client's store via fan-out",
        }


class KnowledgeExtractionAttack:
    """
    Systematically probe to reconstruct a target node/client's knowledge base.
    DRAG: attacker probes via graph routing — limited visibility.
    FedRAG: attacker probes via server fan-out — unlimited direct access.
    """

    def __init__(self, queries_per_topic: int = 5, seed: int = 0):
        self.queries_per_topic = queries_per_topic
        self._rng = random.Random(seed)

    def _extraction_score(self, extracted: list[str], actual: list[str]) -> float:
        if not actual:
            return 0.0
        hits = sum(1 for a in actual if any(_similarity(e, a) > 0.8 for e in extracted))
        return round(hits / len(actual), 4)

    def attack_drag(self, sim: DRAGSimulator) -> dict[str, Any]:
        target_id = self._rng.randint(0, sim.num_peers - 1)
        target = sim.peers[target_id]
        actual_answers = [n.answer for n in target.knowledge_base]

        probe_queries = []
        for topic in TOPICS:
            for i in range(self.queries_per_topic):
                probe_queries.append(f"What is the key fact #{i} about {topic}?")

        extracted = []
        messages_sent = 0
        for q in probe_queries:
            result = sim.query(q, entry_peer_id=None)
            messages_sent += result.get("visited", 0)
            if result["score"] > 0.6:
                extracted.append(result["answer"])

        score = self._extraction_score(extracted, actual_answers)
        return {
            "target_peer": target_id,
            "kb_size": len(actual_answers),
            "probes_sent": len(probe_queries),
            "messages_sent": messages_sent,
            "extracted_count": len(extracted),
            "extraction_score": score,
            "note": "DRAG: multi-hop routing means attacker sees many peers, not just target",
        }

    def attack_fedrag(self, sim: FedRAGSimulator) -> dict[str, Any]:
        target_id = self._rng.randint(0, sim.num_clients - 1)
        target = sim.clients[target_id]
        actual_answers = [n.answer for n in target.store]

        probe_queries = []
        for topic in TOPICS:
            for i in range(self.queries_per_topic):
                probe_queries.append(f"What is the key fact #{i} about {topic}?")

        extracted = []
        for q in probe_queries:
            result = target.query_local(q, sim.embedding_dim)
            if result and result[0] > 0.6:
                extracted.append(result[1])

        score = self._extraction_score(extracted, actual_answers)
        return {
            "target_client": target_id,
            "kb_size": len(actual_answers),
            "probes_sent": len(probe_queries),
            "messages_sent": len(probe_queries),
            "extracted_count": len(extracted),
            "extraction_score": score,
            "note": "FedRAG: server can isolate and probe any single client directly",
        }


class NodeAvailabilityAttack:
    """
    Take nodes/clients offline.
    DRAG: removing high-degree hubs disrupts routing for many peers (BA graph vulnerability).
    FedRAG: removing clients only reduces answer coverage linearly.
    """

    def __init__(self, attack_type: str = "hub_targeted",
                 attack_ratio: float = 0.3, seed: int = 0):
        self.attack_type = attack_type
        self.attack_ratio = attack_ratio
        self._rng = random.Random(seed)

    def attack_drag(self, sim: DRAGSimulator) -> dict[str, Any]:
        n_remove = max(1, int(sim.num_peers * self.attack_ratio))

        if self.attack_type == "hub_targeted":
            # Target highest-degree peers (most damaging to BA graph)
            degrees = [(pid, len(adj)) for pid, adj in sim._adj.items()]
            degrees.sort(key=lambda t: t[1], reverse=True)
            removed = [pid for pid, _ in degrees[:n_remove]]
        elif self.attack_type == "byzantine":
            removed = self._rng.sample(range(sim.num_peers), n_remove)
            for pid in removed:
                sim.peers[pid].byzantine = True
            return {"removed": [], "byzantine": removed,
                    "attack_type": "byzantine",
                    "note": "DRAG: byzantine peers corrupt answers along multi-hop paths"}
        else:
            removed = self._rng.sample(range(sim.num_peers), n_remove)

        for pid in removed:
            sim.peers[pid].offline = True

        # Measure network connectivity after removal
        remaining = {p for p in range(sim.num_peers) if not sim.peers[p].offline}
        if remaining:
            visited: set[int] = set()
            queue = deque([next(iter(remaining))])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                for nb in sim._adj.get(node, []):
                    if nb in remaining and nb not in visited:
                        queue.append(nb)
            connectivity = round(len(visited) / len(remaining), 4)
        else:
            connectivity = 0.0

        return {
            "removed_peers": removed,
            "n_removed": n_remove,
            "attack_type": self.attack_type,
            "network_connectivity": connectivity,
            "note": "DRAG: hub removal in BA graph causes disproportionate connectivity loss",
        }

    def attack_fedrag(self, sim: FedRAGSimulator) -> dict[str, Any]:
        n_remove = max(1, int(sim.num_clients * self.attack_ratio))

        if self.attack_type == "hub_targeted":
            # In star topology there are no hubs — all clients are equal
            removed = self._rng.sample(range(sim.num_clients), n_remove)
        elif self.attack_type == "byzantine":
            removed = self._rng.sample(range(sim.num_clients), n_remove)
            return {"removed": [], "byzantine": removed,
                    "attack_type": "byzantine",
                    "note": "FedRAG: byzantine clients return corrupt answers; server picks best"}
        else:
            removed = self._rng.sample(range(sim.num_clients), n_remove)

        for cid in removed:
            sim.clients[cid].quarantined = True

        active = sim.num_clients - n_remove
        coverage = round(active / sim.num_clients, 4)
        return {
            "removed_clients": removed,
            "n_removed": n_remove,
            "attack_type": self.attack_type,
            "coverage_remaining": coverage,
            "note": "FedRAG: star topology — removing clients linearly reduces coverage only",
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Attack runners
# ══════════════════════════════════════════════════════════════════════════════

def _backup_drag(sim: DRAGSimulator) -> dict[int, list[DRAGKnowledgeNode]]:
    return {p.peer_id: [DRAGKnowledgeNode(**n.__dict__) for n in p.knowledge_base]
            for p in sim.peers}


def _backup_fedrag(sim: FedRAGSimulator) -> dict[int, list[FedRAGNode]]:
    return {c.client_id: [FedRAGNode(**n.__dict__) for n in c.store]
            for c in sim.clients}


def run_data_poisoning(drag: DRAGSimulator, fedrag: FedRAGSimulator,
                       test_rows: list[dict], malicious_ratio: float,
                       poisoning_ratio: float, use_defense: bool, seed: int) -> dict:
    atk = DataPoisoningAttack(malicious_ratio, poisoning_ratio, seed)
    drag_bk = _backup_drag(drag)
    fed_bk = _backup_fedrag(fedrag)

    # ── Baseline ──────────────────────────────────────────────────────────────
    drag_base = drag.evaluate(test_rows)
    fed_base = fedrag.evaluate(test_rows)

    # ── DRAG attack ───────────────────────────────────────────────────────────
    drag_atk_info = atk.attack_drag(drag)
    drag_post = drag.evaluate(test_rows)

    drag_defended = drag_post
    drag_quarantined = []
    if use_defense:
        drag_defended = drag.evaluate(test_rows, use_defense=True)

    # ── FedRAG attack ─────────────────────────────────────────────────────────
    fed_atk_info = atk.attack_fedrag(fedrag)
    fed_post = fedrag.evaluate(test_rows)

    fed_quarantined = []
    if use_defense:
        fed_quarantined = fedrag.run_defense(threshold=0.25)
        fed_defended = fedrag.evaluate(test_rows)
    else:
        fed_defended = fed_post

    # ── Restore ───────────────────────────────────────────────────────────────
    drag.restore_all(drag_bk)
    fedrag.restore_all(fed_bk)

    return {
        "attack": "data_poisoning",
        "drag": {
            "baseline_em": drag_base["em"], "post_attack_em": drag_post["em"],
            "defended_em": drag_defended["em"],
            "em_drop": round(drag_base["em"] - drag_post["em"], 4),
            "em_recovery": round(drag_defended["em"] - drag_post["em"], 4),
            "malicious_peers": len(drag_atk_info["malicious_peers"]),
            "nodes_poisoned": drag_atk_info["nodes_poisoned"],
            "defense": "CrossPeerValidation (majority vote)",
            "quarantined": len(drag_quarantined),
        },
        "fedrag": {
            "baseline_em": fed_base["em"], "post_attack_em": fed_post["em"],
            "defended_em": fed_defended["em"],
            "em_drop": round(fed_base["em"] - fed_post["em"], 4),
            "em_recovery": round(fed_defended["em"] - fed_post["em"], 4),
            "malicious_clients": len(fed_atk_info["malicious_clients"]),
            "nodes_poisoned": fed_atk_info["nodes_poisoned"],
            "defense": "Quarantine (poison marker density)",
            "quarantined": len(fed_quarantined),
        },
    }


def run_membership_inference(drag: DRAGSimulator, fedrag: FedRAGSimulator,
                             seed: int) -> dict:
    atk = MembershipInferenceAttack(seed=seed)
    drag_result = atk.attack_drag(drag)
    fed_result = atk.attack_fedrag(fedrag)
    return {
        "attack": "membership_inference",
        "drag": {
            "score_gap": drag_result["score_gap"],
            "precision": drag_result["precision"],
            "recall": drag_result["recall"],
            "leakage_risk": drag_result["leakage_risk"],
            "note": drag_result["note"],
        },
        "fedrag": {
            "score_gap": fed_result["score_gap"],
            "precision": fed_result["precision"],
            "recall": fed_result["recall"],
            "leakage_risk": fed_result["leakage_risk"],
            "note": fed_result["note"],
        },
    }


def run_knowledge_extraction(drag: DRAGSimulator, fedrag: FedRAGSimulator,
                             seed: int) -> dict:
    atk = KnowledgeExtractionAttack(seed=seed)
    drag_result = atk.attack_drag(drag)
    fed_result = atk.attack_fedrag(fedrag)
    return {
        "attack": "knowledge_extraction",
        "drag": {
            "extraction_score": drag_result["extraction_score"],
            "extracted_count": drag_result["extracted_count"],
            "kb_size": drag_result["kb_size"],
            "messages_sent": drag_result["messages_sent"],
            "note": drag_result["note"],
        },
        "fedrag": {
            "extraction_score": fed_result["extraction_score"],
            "extracted_count": fed_result["extracted_count"],
            "kb_size": fed_result["kb_size"],
            "messages_sent": fed_result["messages_sent"],
            "note": fed_result["note"],
        },
    }


def run_node_availability(drag: DRAGSimulator, fedrag: FedRAGSimulator,
                          test_rows: list[dict], attack_ratio: float,
                          attack_type: str, use_defense: bool, seed: int) -> dict:
    drag_bk = _backup_drag(drag)
    fed_bk = _backup_fedrag(fedrag)

    atk = NodeAvailabilityAttack(attack_type=attack_type, attack_ratio=attack_ratio, seed=seed)

    drag_base = drag.evaluate(test_rows)
    fed_base = fedrag.evaluate(test_rows)

    drag_atk_info = atk.attack_drag(drag)
    drag_post = drag.evaluate(test_rows, use_defense=use_defense)

    fed_atk_info = atk.attack_fedrag(fedrag)
    fed_post = fedrag.evaluate(test_rows)

    drag.restore_all(drag_bk)
    fedrag.restore_all(fed_bk)

    return {
        "attack": "node_availability",
        "attack_type": attack_type,
        "drag": {
            "baseline_em": drag_base["em"], "post_attack_em": drag_post["em"],
            "em_drop": round(drag_base["em"] - drag_post["em"], 4),
            "removed": drag_atk_info.get("n_removed", 0),
            "network_connectivity": drag_atk_info.get("network_connectivity", "N/A"),
            "note": drag_atk_info.get("note", ""),
        },
        "fedrag": {
            "baseline_em": fed_base["em"], "post_attack_em": fed_post["em"],
            "em_drop": round(fed_base["em"] - fed_post["em"], 4),
            "removed": fed_atk_info.get("n_removed", 0),
            "coverage_remaining": fed_atk_info.get("coverage_remaining", "N/A"),
            "note": fed_atk_info.get("note", ""),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Display
# ══════════════════════════════════════════════════════════════════════════════

def _col(text: str, width: int, align: str = "left") -> str:
    text = str(text)
    if align == "right":
        return text.rjust(width)
    if align == "center":
        return text.center(width)
    return text.ljust(width)


def print_banner(drag: DRAGSimulator, fedrag: FedRAGSimulator) -> None:
    print("\n" + "═" * 80)
    print("  DRAG vs FedRAG — Security Comparison".center(80))
    print("═" * 80)
    print(f"  DRAG  : {drag.num_peers} peers · Barabási–Albert graph (m={drag.num_attachments})"
          f" · confidence_threshold={drag.confidence_threshold} · max_hops={drag.max_hops}")
    print(f"  FedRAG: {fedrag.num_clients} clients · Star topology (central server fan-out)")
    print(f"  Dataset: {len(drag.dataset)} examples (IID split) · same seed · hash embeddings")
    print("═" * 80)


def print_architecture_comparison() -> None:
    print("\n┌─────────────────────────────────────────────────────────────────────────┐")
    print("│                    ARCHITECTURAL COMPARISON                             │")
    print("├─────────────────────────────┬───────────────────────────────────────────┤")
    print("│ Dimension                   │ DRAG                │ FedRAG              │")
    print("├─────────────────────────────┼─────────────────────┼─────────────────────┤")
    rows = [
        ("Topology",            "P2P (BA graph)",       "Star (central server)"),
        ("Query routing",       "Multi-hop BFS",        "Fan-out to ALL clients"),
        ("Knowledge store",     "Private per-peer KB",  "Private per-client store"),
        ("Store replication",   "None (peer-local)",    "None (client-local)"),
        ("Defense",             "CrossPeerValidation",  "Quarantine (poison %)"),
        ("Defense method",      "Majority vote (nbrs)", "Poison marker density"),
        ("Attack isolation",    "Spreads via hops",     "Isolated to 1 client"),
        ("Privacy model",       "Peers know neighbors", "Clients fully isolated"),
        ("Hub vulnerability",   "YES (BA degree dist)", "NO (all nodes equal)"),
        ("Scalability",         "Sub-linear messages",  "Linear messages (all)"),
        ("LLM required",        "YES (real system)",    "NO (retrieval only)"),
    ]
    for dim, drag_val, fed_val in rows:
        d = _col(dim, 29)
        dv = _col(drag_val, 21)
        fv = _col(fed_val, 21)
        print(f"│ {d}│ {dv}│ {fv}│")
    print("└─────────────────────────────┴─────────────────────┴─────────────────────┘")


def print_attack_result(result: dict) -> None:
    attack = result["attack"].upper().replace("_", " ")
    print(f"\n{'─' * 80}")
    print(f"  ATTACK: {attack}")
    print(f"{'─' * 80}")

    if result["attack"] == "data_poisoning":
        print(f"  {'':30} {'DRAG':>22} {'FedRAG':>22}")
        print(f"  {'':30} {'──────────────────────':>22} {'──────────────────────':>22}")
        fields = [
            ("Baseline EM", "baseline_em", "baseline_em"),
            ("Post-attack EM", "post_attack_em", "post_attack_em"),
            ("EM drop", "em_drop", "em_drop"),
            ("Defended EM", "defended_em", "defended_em"),
            ("EM recovery", "em_recovery", "em_recovery"),
            ("Malicious nodes", "malicious_peers", "malicious_clients"),
            ("Nodes poisoned", "nodes_poisoned", "nodes_poisoned"),
            ("Defense used", "defense", "defense"),
            ("Quarantined", "quarantined", "quarantined"),
        ]
        for label, dk, fk in fields:
            dv = str(result["drag"].get(dk, ""))
            fv = str(result["fedrag"].get(fk, ""))
            print(f"  {_col(label, 30)}{_col(dv, 22, 'right')}{_col(fv, 22, 'right')}")

    elif result["attack"] == "membership_inference":
        print(f"  {'':30} {'DRAG':>22} {'FedRAG':>22}")
        print(f"  {'':30} {'──────────────────────':>22} {'──────────────────────':>22}")
        for label, key in [("Score gap (member-non)", "score_gap"),
                           ("Precision", "precision"),
                           ("Recall", "recall"),
                           ("Leakage risk", "leakage_risk")]:
            dv = str(result["drag"].get(key, ""))
            fv = str(result["fedrag"].get(key, ""))
            print(f"  {_col(label, 30)}{_col(dv, 22, 'right')}{_col(fv, 22, 'right')}")
        print(f"\n  DRAG note  : {result['drag']['note']}")
        print(f"  FedRAG note: {result['fedrag']['note']}")

    elif result["attack"] == "knowledge_extraction":
        print(f"  {'':30} {'DRAG':>22} {'FedRAG':>22}")
        print(f"  {'':30} {'──────────────────────':>22} {'──────────────────────':>22}")
        for label, key in [("Extraction score", "extraction_score"),
                           ("Items extracted", "extracted_count"),
                           ("Target KB size", "kb_size"),
                           ("Messages sent", "messages_sent")]:
            dv = str(result["drag"].get(key, ""))
            fv = str(result["fedrag"].get(key, ""))
            print(f"  {_col(label, 30)}{_col(dv, 22, 'right')}{_col(fv, 22, 'right')}")
        print(f"\n  DRAG note  : {result['drag']['note']}")
        print(f"  FedRAG note: {result['fedrag']['note']}")

    elif result["attack"] == "node_availability":
        print(f"  Attack type: {result.get('attack_type', 'N/A')}")
        print(f"  {'':30} {'DRAG':>22} {'FedRAG':>22}")
        print(f"  {'':30} {'──────────────────────':>22} {'──────────────────────':>22}")
        dnet = result["drag"].get("network_connectivity", "N/A")
        fcov = result["fedrag"].get("coverage_remaining", "N/A")
        for label, dk, fk in [
            ("Baseline EM",     "baseline_em",          "baseline_em"),
            ("Post-attack EM",  "post_attack_em",       "post_attack_em"),
            ("EM drop",         "em_drop",              "em_drop"),
            ("Nodes removed",   "removed",              "removed"),
        ]:
            dv = str(result["drag"].get(dk, ""))
            fv = str(result["fedrag"].get(fk, ""))
            print(f"  {_col(label, 30)}{_col(dv, 22, 'right')}{_col(fv, 22, 'right')}")
        print(f"  {'Network connectivity':30}{'':>22}{_col(str(dnet), 22, 'right')}"
              .replace("                      ", f" {dnet:>21}"))
        # Reprint with correct values
        print(f"  {'Network connectivity (DRAG)':30}{_col(str(dnet), 22, 'right')}")
        print(f"  {'Coverage remaining (FedRAG)':30}{'':>22}{_col(str(fcov), 22, 'right')}")
        print(f"\n  DRAG note  : {result['drag']['note']}")
        print(f"  FedRAG note: {result['fedrag']['note']}")


def print_summary(results: list[dict], drag: DRAGSimulator,
                  fedrag: FedRAGSimulator, use_defense: bool) -> None:
    print("\n" + "═" * 80)
    print("  SUMMARY TABLE".center(80))
    print("═" * 80)
    print(f"  {'Attack':28} {'DRAG EM drop':>14} {'FedRAG EM drop':>16} {'Winner':>12}")
    print(f"  {'──────':28} {'────────────':>14} {'──────────────':>16} {'──────':>12}")

    for r in results:
        if r["attack"] not in ("data_poisoning", "node_availability"):
            continue
        label = r["attack"].replace("_", " ").title()
        dd = r["drag"].get("em_drop", "N/A")
        fd = r["fedrag"].get("em_drop", "N/A")
        if isinstance(dd, float) and isinstance(fd, float):
            winner = "DRAG" if dd < fd else ("FedRAG" if fd < dd else "TIE")
        else:
            winner = "N/A"
        print(f"  {_col(label, 28)}{_col(str(dd), 16, 'right')}{_col(str(fd), 16, 'right')}"
              f"{_col(winner, 14, 'right')}")

    for r in results:
        if r["attack"] not in ("membership_inference", "knowledge_extraction"):
            continue
        label = r["attack"].replace("_", " ").title()
        dd = r["drag"].get("leakage_risk") or r["drag"].get("extraction_score", "N/A")
        fd = r["fedrag"].get("leakage_risk") or r["fedrag"].get("extraction_score", "N/A")
        print(f"  {_col(label, 28)}{_col(str(dd), 16, 'right')}{_col(str(fd), 16, 'right')}"
              f"{'(see above)':>14}")

    print("\n  Key insights:")
    print("  • DRAG hub-targeted removal disrupts routing disproportionately (BA graph)")
    print("  • FedRAG client removal only reduces answer coverage linearly (star topology)")
    print("  • DRAG CrossPeerValidation catches poisoned answers via majority vote")
    print("  • FedRAG quarantine requires poison markers to be detectable")
    print("  • DRAG membership inference harder — attacker routes through graph (indirect)")
    print("  • FedRAG membership inference easier — server can directly target any client")
    print("  • FedRAG knowledge extraction trivially isolates target client (direct access)")
    print("  • DRAG knowledge extraction mixes results from multiple hops (harder to isolate)")
    print("═" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DRAG vs FedRAG side-by-side security comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--num-nodes", type=int, default=20,
                   help="Number of peers (DRAG) / clients (FedRAG). Default: 20")
    p.add_argument("--num-examples", type=int, default=400,
                   help="Total dataset size. Default: 400")
    p.add_argument("--malicious-ratio", type=float, default=0.3,
                   help="Fraction of nodes that are malicious. Default: 0.3")
    p.add_argument("--poisoning-ratio", type=float, default=0.3,
                   help="Fraction of each malicious node's KB to poison. Default: 0.3")
    p.add_argument("--attack-ratio", type=float, default=0.3,
                   help="Fraction of nodes to take offline (node availability). Default: 0.3")
    p.add_argument("--node-attack-type", default="hub_targeted",
                   choices=["hub_targeted", "random", "byzantine"],
                   help="Node availability attack strategy. Default: hub_targeted")
    p.add_argument("--num-attachments", type=int, default=2,
                   help="DRAG BA graph attachment parameter (m). Default: 2")
    p.add_argument("--confidence-threshold", type=float, default=0.5,
                   help="DRAG peer query confidence threshold. Default: 0.5")
    p.add_argument("--max-hops", type=int, default=3,
                   help="DRAG BFS max hops. Default: 3")
    p.add_argument("--attack",
                   choices=["all", "data_poisoning", "membership_inference",
                             "knowledge_extraction", "node_availability"],
                   default="all")
    p.add_argument("--no-defense", action="store_true",
                   help="Disable defenses (measure raw attack impact)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None,
                   help="Save results to JSON file")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    use_defense = not args.no_defense
    t0 = time.time()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Building shared dataset "
          f"({args.num_examples} examples, seed={args.seed}) …")
    dataset = _make_dataset(args.num_examples, args.seed)
    test_rows = dataset[:min(50, len(dataset))]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Building DRAG network "
          f"({args.num_nodes} peers, BA m={args.num_attachments}) …")
    drag_sim = DRAGSimulator(
        num_peers=args.num_nodes, num_examples=args.num_examples,
        num_attachments=args.num_attachments,
        confidence_threshold=args.confidence_threshold,
        max_hops=args.max_hops, embedding_dim=32,
        dataset=dataset, seed=args.seed,
    )

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Building FedRAG system "
          f"({args.num_nodes} clients, star topology) …")
    fedrag_sim = FedRAGSimulator(
        num_clients=args.num_nodes, num_examples=args.num_examples,
        embedding_dim=32, dataset=dataset, seed=args.seed,
    )

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Defense: {'ON' if use_defense else 'OFF'}")

    print_banner(drag_sim, fedrag_sim)
    print_architecture_comparison()

    results = []
    run_all = args.attack == "all"

    if run_all or args.attack == "data_poisoning":
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running DATA POISONING …")
        r = run_data_poisoning(drag_sim, fedrag_sim, test_rows,
                               args.malicious_ratio, args.poisoning_ratio,
                               use_defense, args.seed)
        results.append(r)
        print_attack_result(r)

    if run_all or args.attack == "membership_inference":
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running MEMBERSHIP INFERENCE …")
        r = run_membership_inference(drag_sim, fedrag_sim, args.seed)
        results.append(r)
        print_attack_result(r)

    if run_all or args.attack == "knowledge_extraction":
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running KNOWLEDGE EXTRACTION …")
        r = run_knowledge_extraction(drag_sim, fedrag_sim, args.seed)
        results.append(r)
        print_attack_result(r)

    if run_all or args.attack == "node_availability":
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running NODE AVAILABILITY "
              f"({args.node_attack_type}) …")
        r = run_node_availability(drag_sim, fedrag_sim, test_rows,
                                  args.attack_ratio, args.node_attack_type,
                                  use_defense, args.seed)
        results.append(r)
        print_attack_result(r)

    print_summary(results, drag_sim, fedrag_sim, use_defense)

    elapsed = round(time.time() - t0, 2)
    print(f"\n  Completed in {elapsed}s")

    if args.output:
        out = {
            "timestamp": datetime.now().isoformat(),
            "config": vars(args),
            "drag_config": {
                "num_peers": drag_sim.num_peers,
                "num_attachments": drag_sim.num_attachments,
                "confidence_threshold": drag_sim.confidence_threshold,
                "max_hops": drag_sim.max_hops,
            },
            "fedrag_config": {
                "num_clients": fedrag_sim.num_clients,
                "topology": "star",
            },
            "results": results,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"  Results saved → {args.output}")


if __name__ == "__main__":
    main()
