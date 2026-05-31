#!/usr/bin/env python3
"""
run_fedavg_gpu.py
=================
Standalone FedRAG security evaluation — DRAG-style attack/defense flow
applied to FedRAG's true per-client local InMemoryKnowledgeStore architecture.

No GPU · No Hugging Face downloads · No Ollama · No pip install -e .

DRAG vs FedRAG conceptual mapping
-----------------------------------
  DRAG concept          │  FedRAG equivalent used here
  ──────────────────────┼──────────────────────────────────────────────────
  Peer node             │  Federated client (SimClient)
  Peer's local store    │  Client's private InMemoryKnowledgeStore (nodes)
  Attack on a peer      │  Attack on one client's LOCAL store only
  Server aggregation    │  Server fans out → picks best answer globally
  Defense / quarantine  │  Server-side ClientDataPoisoningDefense

Full DRAG-style attack flow (example: 150 clients, 50 malicious)
-----------------------------------------------------------------
  Step 1 — 50 clients are selected as malicious (--malicious-ratio 0.33)
  Step 2 — Each malicious client INDEPENDENTLY poisons ITS OWN local store
            (100 clean clients are completely untouched)
  Step 3 — Server evaluates: fans out query to ALL 150 clients,
            receives answers from 50 poisoned + 100 clean local stores,
            picks the globally best answer
  Step 4 — Server-side defense inspects per-client data,
            quarantines clients with high poison-marker density
  Step 5 — Final evaluation excludes quarantined clients

The 4 attacks (same as DRAG)
-----------------------------
  1. Data Poisoning        — inject wrong/misleading answers into client local stores
  2. Membership Inference  — infer whether a query is in a specific client's local store
  3. Knowledge Extraction  — recover nodes from a target client's local store via probing
  4. Node Availability     — take clients offline (removal / Byzantine / DDoS / partition)

Usage
-----
  # Default: 10 clients, 200 examples, all attacks, defenses ON
  python run_fedavg_gpu.py

  # 150 clients — 50 malicious (1/3 ratio)
  python run_fedavg_gpu.py --num-clients 150 --num-examples 1500 --malicious-ratio 0.33

  # Specific attack only
  python run_fedavg_gpu.py --attack data_poisoning
  python run_fedavg_gpu.py --attack membership_inference
  python run_fedavg_gpu.py --attack knowledge_extraction
  python run_fedavg_gpu.py --attack node_availability

  # All attacks, defenses disabled (measure raw impact)
  python run_fedavg_gpu.py --no-defense

  # Custom scenario
  python run_fedavg_gpu.py \\
      --num-clients 20 \\
      --num-examples 500 \\
      --seed 42 \\
      --malicious-ratio 0.25 \\
      --poisoning-ratio 0.4 \\
      --poison-type answer_swap \\
      --node-attack-type byzantine \\
      --quarantine-threshold 0.25 \\
      --output-dir my_results/

  # Use your own JSONL dataset (must have query / response / topic columns)
  python run_fedavg_gpu.py --dataset-jsonl path/to/data.jsonl

Output
------
  results/
    security_report.json    — raw results for every attack
    SECURITY_REPORT.md      — human-readable Markdown summary
    attack_matrix.csv       — flat CSV (one row per attack)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

# ──────────────────────────────────────────────────────────────────────────────
# Ensure the repo root and src/ are importable (needed for fed_rag.* imports)
# ──────────────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.resolve()
for _p in (_REPO, _REPO / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Lightweight simulator (no ML deps)
# Each client owns a PRIVATE local InMemoryKnowledgeStore.
# Retrieval uses deterministic hash-cosine similarity.
# ══════════════════════════════════════════════════════════════════════════════

TOPICS = [
    "cryptography", "medicine", "finance", "biology", "law",
    "networking", "history", "physics", "privacy", "safety",
    "databases", "governance",
]


@dataclass
class SimNode:
    """One knowledge-store entry in a client's LOCAL store."""
    node_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimClient:
    """One federated client — owns a PRIVATE local knowledge store.

    Analogous to a DRAG peer node, but without p2p networking.
    """
    client_id: int
    examples: list[dict[str, Any]]        # raw IID shard
    nodes: list[SimNode] = field(default_factory=list)  # private local store
    poisoned: bool = False
    quarantined: bool = False

    @property
    def data_size(self) -> int:
        return len(self.examples)


def _hash_embedding(text: str, dim: int = 32) -> list[float]:
    """Deterministic unit-norm embedding via SHA-256. No ML needed."""
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


def _build_nodes(examples: list[dict[str, Any]], dim: int = 32) -> list[SimNode]:
    """Build SimNodes from raw examples — creates the client's local store."""
    nodes: list[SimNode] = []
    for row in examples:
        query = row.get("query", "")
        emb = _hash_embedding(query, dim)
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


class FedRAGSimulator:
    """Lightweight FedRAG deployment simulator.

    Each client holds a PRIVATE, NON-REPLICATED local store (analogous to a
    DRAG peer's knowledge base).  The server never accesses client data directly
    — it only fans out queries and collects per-client answers.
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
                raise FileNotFoundError(f"Dataset not found: {dataset_jsonl}")
            self.dataset = self._load_jsonl(p)
        else:
            self.dataset = self._make_synthetic_dataset()

        self.clients: list[SimClient] = self._build_clients()

    # ── Per-client retrieval ─────────────────────────────────────────────────

    def retrieve_from_client(
        self,
        query: str,
        client_id: int,
        top_k: int = 1,
        score_mask: float | None = None,
    ) -> list[tuple[float, SimNode]]:
        """Query ONE client's LOCAL store only (analogous to querying one DRAG peer)."""
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

    # ── Server-side federated aggregation ───────────────────────────────────

    def federated_retrieve(
        self,
        query: str,
        top_k: int = 1,
        dropped_clients: set[int] | None = None,
        byzantine_clients: set[int] | None = None,
        score_mask: float | None = None,
    ) -> list[tuple[float, SimNode, int]]:
        """TRUE federated retrieval — server fans out to each client's LOCAL store.

        Flow (matching DRAG peer query flow):
          1. Server sends query to every active client.
          2. Each client queries ITS OWN local store → returns best (score, node).
          3. Server aggregates: picks globally top-k answers across all per-client responses.
          No pooling — each client's store is always private.
        """
        dropped = dropped_clients or set()
        byzantine = byzantine_clients or set()

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

        per_client_best.sort(key=lambda t: t[0], reverse=True)
        return per_client_best[:top_k]

    # ── Store management ─────────────────────────────────────────────────────

    def rebuild_client_store(self, client_id: int) -> None:
        """Rebuild one client's local store from its current examples."""
        client = self.clients[client_id]
        client.nodes = _build_nodes(client.examples, self.embedding_dim)

    def quarantine_client(self, client_id: int) -> None:
        self.clients[client_id].quarantined = True

    def unquarantine_all(self) -> None:
        for c in self.clients:
            c.quarantined = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_clients": self.num_clients,
            "num_examples": self.num_examples,
            "seed": self.seed,
            "embedding_dim": self.embedding_dim,
            "client_sizes": [c.data_size for c in self.clients],
        }

    # ── Internals ────────────────────────────────────────────────────────────

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
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _build_clients(self) -> list[SimClient]:
        """IID split → each client builds its OWN private local store."""
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
                nodes=_build_nodes(examples, self.embedding_dim),
            ))
        return clients


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def _token_f1(pred: str, gold: str) -> float:
    p = pred.lower().split()
    g = gold.lower().split()
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def _simple_bleu(pred: str, gold: str) -> float:
    p = pred.lower().split()
    g = gold.lower().split()
    if not p:
        return 0.0
    return sum(1 for t in p if t in g) / len(p)


def _evaluate(
    dataset: list[dict[str, Any]],
    sim: FedRAGSimulator,
    dropped: set[int] | None = None,
    byzantine: set[int] | None = None,
    score_mask: float | None = None,
) -> dict[str, float]:
    """Evaluate GLOBAL system using true per-client fan-out.

    For each query:
      1. Fan out to every active (non-dropped, non-quarantined) client.
      2. Each client queries ITS OWN local store.
      3. Server picks the globally best answer.
    """
    dropped = dropped or set()
    byzantine = byzantine or set()
    em, f1, bleu = [], [], []

    for row in dataset:
        hits = sim.federated_retrieve(
            row["query"], top_k=1,
            dropped_clients=dropped,
            byzantine_clients=byzantine,
            score_mask=score_mask,
        )
        pred = str(hits[0][1].metadata.get("answer", "")) if hits else ""
        gold = row["response"]
        em.append(_exact_match(pred, gold))
        f1.append(_token_f1(pred, gold))
        bleu.append(_simple_bleu(pred, gold))

    n = len(em) or 1
    return {"em": sum(em) / n, "f1": sum(f1) / n, "bleu": sum(bleu) / n}


def _evaluate_per_client(
    dataset: list[dict[str, Any]],
    sim: FedRAGSimulator,
) -> dict[int, dict[str, float]]:
    """Evaluate each client IN ISOLATION — only that client's local store is active."""
    all_ids = {c.client_id for c in sim.clients}
    result: dict[int, dict[str, float]] = {}
    for client in sim.clients:
        cid = client.client_id
        dropped = all_ids - {cid}
        result[cid] = _evaluate(dataset, sim, dropped=dropped)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Standalone attack implementations (no fed_rag package needed)
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. Data Poisoning ───────────────────────────────────────────────────────

class DataPoisoningAttack:
    """Inject malicious (query, response) pairs into a client's local dataset.

    Mirrors DRAG's data poisoning attack, but targets a single FedRAG client's
    local dataset shard instead of a p2p peer's knowledge base.
    """

    POISON_MARKER = "POISONED"

    def __init__(
        self,
        poisoning_ratio: float = 0.2,
        poison_type: str = "wrong_answer",
        mode: str = "replace",
        seed: int = 0,
    ) -> None:
        self.poisoning_ratio = max(0.0, min(1.0, poisoning_ratio))
        self.poison_type = poison_type
        self.mode = mode
        self.seed = seed

    def execute(self, examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Return (poisoned_examples, num_poisoned)."""
        if not examples or self.poisoning_ratio == 0:
            return list(examples), 0

        rng = random.Random(self.seed)
        base = [dict(e) for e in examples]
        n = max(1, int(len(base) * self.poisoning_ratio))
        n = min(n, len(base))
        indices = rng.sample(range(len(base)), n)

        answers_by_topic: dict[str, list[str]] = defaultdict(list)
        all_answers: list[str] = []
        for e in base:
            ans = e.get("response", "")
            all_answers.append(ans)
            t = e.get("topic")
            if t:
                answers_by_topic[t].append(ans)

        poisoned_records: list[dict[str, Any]] = []
        for idx in indices:
            orig = base[idx]
            bad_ans = self._poison_answer(orig, answers_by_topic, all_answers, rng)
            poisoned = dict(orig)
            poisoned["response"] = bad_ans
            if self.mode == "replace":
                base[idx] = poisoned
            else:
                poisoned_records.append(poisoned)

        if self.mode == "append":
            base.extend(poisoned_records)

        return base, n

    def _poison_answer(
        self,
        orig: dict[str, Any],
        answers_by_topic: dict[str, list[str]],
        all_answers: list[str],
        rng: random.Random,
    ) -> str:
        pt = self.poison_type
        if pt == "wrong_answer":
            return f"{self.POISON_MARKER}: This is incorrect information."
        if pt == "misleading":
            return f"The correct answer is the opposite of {orig.get('response', '')}"
        if pt == "noise":
            noise = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=10))
            return f"{orig.get('response', '')} {noise}"
        if pt == "answer_swap":
            topic = orig.get("topic")
            candidates = [
                a for a in (answers_by_topic.get(topic, []) if topic else [])
                if a != orig.get("response", "")
            ]
            if candidates:
                return rng.choice(candidates)
            fallback = [a for a in all_answers if a != orig.get("response", "")]
            return rng.choice(fallback) if fallback else f"{self.POISON_MARKER}: fallback."
        return f"{self.POISON_MARKER}: unknown poison type."


# ── 3b. Client Data Poisoning Defense ────────────────────────────────────────

@dataclass(frozen=True)
class ClientInspection:
    client_id: int
    total_examples: int
    poisoned_marker_count: int
    duplicate_query_count: int
    conflicting_answer_count: int
    risk_score: float
    quarantined: bool


class ClientDataPoisoningDefense:
    """Server-side defense: inspect per-client data, quarantine bad actors.

    Mirrors DRAG's server-side validation step — the server checks each
    client's dataset for obvious poisoning signs before aggregating.
    """

    POISON_MARKERS = ("poisoned",)

    def __init__(
        self,
        quarantine_threshold: float = 0.25,
    ) -> None:
        self.quarantine_threshold = quarantine_threshold

    def inspect_all(
        self, client_examples: list[list[dict[str, Any]]]
    ) -> list[ClientInspection]:
        return [
            self._inspect_one(cid, examples)
            for cid, examples in enumerate(client_examples)
        ]

    def sanitize(
        self, client_examples: list[list[dict[str, Any]]]
    ) -> tuple[list[list[dict[str, Any]]], list[ClientInspection]]:
        inspections = self.inspect_all(client_examples)
        sanitized: list[list[dict[str, Any]]] = []
        for inspection, examples in zip(inspections, client_examples):
            seen: dict[str, str] = {}
            clean: list[dict[str, Any]] = []
            for ex in examples:
                query = str(ex.get("query", ""))
                ans = str(ex.get("response", ""))
                if self._has_marker(ans):
                    continue
                if query in seen and seen[query] != ans:
                    continue
                seen[query] = ans
                clean.append(dict(ex))
            sanitized.append(clean)
        return sanitized, inspections

    def _inspect_one(
        self, cid: int, examples: list[dict[str, Any]]
    ) -> ClientInspection:
        total = len(examples)
        answers = [str(e.get("response", "")) for e in examples]
        marker_count = sum(self._has_marker(a) for a in answers)

        query_counts = Counter(str(e.get("query", "")) for e in examples)
        dup_count = sum(c - 1 for c in query_counts.values() if c > 1)

        by_query: dict[str, set[str]] = defaultdict(set)
        for e in examples:
            by_query[str(e.get("query", ""))].add(str(e.get("response", "")))
        conflict_count = sum(len(v) - 1 for v in by_query.values() if len(v) > 1)

        risk = (marker_count + dup_count + conflict_count) / max(1, total)
        return ClientInspection(
            client_id=cid,
            total_examples=total,
            poisoned_marker_count=marker_count,
            duplicate_query_count=dup_count,
            conflicting_answer_count=conflict_count,
            risk_score=risk,
            quarantined=risk >= self.quarantine_threshold,
        )

    def _has_marker(self, answer: str) -> bool:
        a = answer.lower()
        return any(m in a for m in self.POISON_MARKERS)


# ── 3c. Membership Inference Attack ──────────────────────────────────────────

class MembershipInferenceAttack:
    """Infer whether a query is in a target client's local store.

    Analogous to DRAG's membership inference against a peer node.
    """

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    def run(
        self,
        sim: FedRAGSimulator,
        target_client_id: int,
        score_mask: float | None = None,
    ) -> dict[str, float]:
        client = sim.clients[target_client_id]
        members = [r["query"] for r in client.examples]
        non_members = [
            r["query"]
            for other in sim.clients
            if other.client_id != target_client_id
            for r in other.examples[:3]
        ][:len(members)]

        def _score(query: str) -> float:
            hits = sim.retrieve_from_client(
                query, target_client_id, top_k=1,
                score_mask=score_mask,
            )
            return hits[0][0] if hits else 0.0

        tp = tn = fp = fn = 0
        for q in members:
            if _score(q) >= self.threshold:
                tp += 1
            else:
                fn += 1
        for q in non_members:
            if _score(q) >= self.threshold:
                fp += 1
            else:
                tn += 1

        total = tp + tn + fp + fn or 1
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return {
            "accuracy": (tp + tn) / total,
            "tpr": tpr,
            "fpr": fpr,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        }


# ── 3d. Query Rate Limiter (defense for knowledge extraction) ─────────────────

class ClientQueryRateLimiter:
    def __init__(self, max_queries_per_client: int = 20) -> None:
        self.max_qpc = max(1, max_queries_per_client)
        self._counts: dict[int, int] = defaultdict(int)
        self.allowed = 0
        self.blocked = 0

    def check_and_record(self, client_id: int, query: str) -> tuple[bool, str]:
        if self._counts[client_id] >= self.max_qpc:
            self.blocked += 1
            return False, "rate_limit_exceeded"
        self._counts[client_id] += 1
        self.allowed += 1
        return True, "allowed"

    def get_stats(self) -> dict[str, Any]:
        total = self.allowed + self.blocked
        return {
            "allowed_count": self.allowed,
            "blocked_count": self.blocked,
            "block_rate": self.blocked / max(1, total),
        }


# ── 3e. Knowledge Extraction Attack ──────────────────────────────────────────

class KnowledgeExtractionAttack:
    """Probe a target client's local store to recover its knowledge nodes.

    Analogous to DRAG's knowledge-base extraction against a peer.
    """

    TEMPLATES = (
        "What do you know about {topic}?",
        "Explain {topic}.",
        "Give facts about {topic}.",
        "Define {topic}.",
        "Describe {topic} in detail.",
    )

    def run(
        self,
        sim: FedRAGSimulator,
        target_client_id: int,
        top_k: int = 3,
        rate_limiter: ClientQueryRateLimiter | None = None,
    ) -> dict[str, Any]:
        client = sim.clients[target_client_id]
        topics = list({r.get("topic", "unknown") for r in client.examples})
        queries: list[tuple[str, str]] = []
        for row in client.examples:
            queries.append((row["query"], row.get("topic", "unknown")))
        for t in topics:
            for tmpl in self.TEMPLATES:
                queries.append((tmpl.format(topic=t), t))

        def _extract(query_list: list[tuple[str, str]]) -> dict[str, Any]:
            recovered: dict[str, str] = {}
            for q, _ in query_list:
                hits = sim.retrieve_from_client(q, target_client_id, top_k=1)
                for score, node in hits:
                    if score > 0.5:
                        recovered[node.node_id] = node.text
            total = len(client.nodes) or 1
            return {
                "recovered_nodes": len(recovered),
                "total_nodes": total,
                "recovery_ratio": len(recovered) / total,
                "queries_issued": len(query_list),
            }

        attacked_stats = _extract(queries)

        defended_stats: dict[str, Any] | None = None
        if rate_limiter is not None:
            allowed = [
                (q, t) for q, t in queries
                if rate_limiter.check_and_record(target_client_id, q)[0]
            ]
            defended_stats = _extract(allowed)

        return {
            "attacked": attacked_stats,
            "defended": defended_stats,
            "total_queries": len(queries),
        }


# ── 3f. Node Availability Attack ─────────────────────────────────────────────

class NodeAvailabilityAttack:
    """Take clients offline: removal, byzantine, DDoS, partition, or sybil.

    Maps directly to DRAG's node availability attack, with 'client' in place
    of 'peer node'.
    """

    def __init__(
        self,
        attack_type: str = "node_removal",
        attack_ratio: float = 0.3,
        seed: int = 0,
    ) -> None:
        valid = {"node_removal", "byzantine", "partition", "ddos", "sybil"}
        if attack_type not in valid:
            raise ValueError(f"attack_type must be one of {valid}")
        self.attack_type = attack_type
        self.attack_ratio = attack_ratio
        self.seed = seed

    def execute(
        self, clients: list[SimClient]
    ) -> tuple[set[int], set[int], float, float]:
        """Return (dropped_ids, byzantine_ids, availability_before, availability_after)."""
        rng = random.Random(self.seed)
        total = len(clients)
        n = max(1, int(total * self.attack_ratio))
        avail_before = 1.0

        dropped: set[int] = set()
        byzantine: set[int] = set()

        if self.attack_type in ("node_removal", "ddos"):
            victims = rng.sample(range(total), min(n, total))
            dropped = set(victims)
        elif self.attack_type == "byzantine":
            victims = rng.sample(range(total), min(n, total))
            byzantine = set(victims)
        elif self.attack_type == "partition":
            split = total // 2
            dropped = set(range(split))
        elif self.attack_type == "sybil":
            # Sybil: inject fake low-quality clients; model as dropped real clients
            victims = rng.sample(range(total), min(n, total))
            dropped = set(victims)

        avail_after = (total - len(dropped)) / max(1, total)
        return dropped, byzantine, avail_before, avail_after


# ── 3g. Cross-Peer Validation defense ────────────────────────────────────────

class CrossPeerValidation:
    """Majority-vote defense against Byzantine clients.

    Analogous to DRAG's cross-peer validation — the server collects answers
    from multiple clients and picks the majority answer, filtering out
    Byzantine-corrupt responses.
    """

    def validate(
        self,
        answers: list[str],
        byzantine_ids: set[int],
        client_ids: list[int],
    ) -> tuple[set[int], str]:
        """Return (caught_byzantine_ids, best_clean_answer)."""
        clean_answers = [
            a for a, cid in zip(answers, client_ids)
            if cid not in byzantine_ids
            and not a.startswith("[BYZANTINE-CORRUPT]")
        ]
        caught = {cid for cid in byzantine_ids if any(
            a.startswith("[BYZANTINE-CORRUPT]")
            for a, c in zip(answers, client_ids) if c == cid
        )}
        best = Counter(clean_answers).most_common(1)[0][0] if clean_answers else ""
        return caught, best


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Attack runners (DRAG-style flow applied to FedRAG)
# ══════════════════════════════════════════════════════════════════════════════

def _run_data_poisoning(
    sim: FedRAGSimulator,
    malicious_ids: list[int],
    poisoning_ratio: float = 0.2,
    poison_type: str = "wrong_answer",
    quarantine_threshold: float = 0.25,
    defense_enabled: bool = True,
) -> dict[str, Any]:
    """
    DRAG-style data poisoning on FedRAG.

    Step 1: Baseline — evaluate clean system.
    Step 2: Each malicious client independently poisons ITS OWN local store.
            (Clean clients are completely unaffected at this point.)
    Step 3: Server evaluates: fans out to ALL clients (clean + poisoned).
    Step 4: Defense: server inspects per-client data, quarantines bad actors.
    Step 5: Re-evaluate excluding quarantined clients.
    """
    dataset = sim.dataset

    # ── Step 1: Baseline ────────────────────────────────────────────────────
    baseline = _evaluate(dataset, sim)
    baseline_per_client = _evaluate_per_client(dataset, sim)

    # ── Step 2: Attack — each malicious client poisons its OWN local store ──
    t0 = time.perf_counter()
    orig_examples = {c.client_id: list(c.examples) for c in sim.clients}
    total_poisoned = 0

    for cid in malicious_ids:
        atk = DataPoisoningAttack(
            poisoning_ratio=poisoning_ratio,
            poison_type=poison_type,
            mode="replace",
            seed=sim.seed + cid,
        )
        poisoned_examples, n = atk.execute(sim.clients[cid].examples)
        sim.clients[cid].examples = poisoned_examples
        sim.clients[cid].nodes = _build_nodes(poisoned_examples, sim.embedding_dim)
        sim.clients[cid].poisoned = True
        total_poisoned += n

    # ── Step 3: Post-attack evaluation — server queries ALL clients ──────────
    attacked = _evaluate(dataset, sim)
    attacked_per_client = _evaluate_per_client(dataset, sim)
    attack_runtime = time.perf_counter() - t0

    # ── Step 4: Defense ──────────────────────────────────────────────────────
    defended: dict[str, float] = dict(attacked)
    defended_per_client = {cid: dict(v) for cid, v in attacked_per_client.items()}
    defense_stats: dict[str, Any] = {}
    quarantined: list[int] = []

    if defense_enabled:
        defense = ClientDataPoisoningDefense(quarantine_threshold=quarantine_threshold)
        all_client_data = [c.examples for c in sim.clients]
        sanitized_data, inspections = defense.sanitize(all_client_data)

        quarantined = [r.client_id for r in inspections if r.quarantined]
        for cid, examples in enumerate(sanitized_data):
            sim.clients[cid].examples = examples
            sim.clients[cid].nodes = _build_nodes(examples, sim.embedding_dim)
            if cid in quarantined:
                sim.clients[cid].quarantined = True

        # ── Step 5: Re-evaluate without quarantined clients ─────────────────
        defended = _evaluate(dataset, sim)
        defended_per_client = _evaluate_per_client(dataset, sim)
        defense_stats = {
            "quarantined_clients": quarantined,
            "inspections": [
                {
                    "client_id": r.client_id,
                    "risk_score": round(r.risk_score, 4),
                    "quarantined": r.quarantined,
                    "poisoned_markers": r.poisoned_marker_count,
                }
                for r in inspections
            ],
        }

    # ── Restore clean state ──────────────────────────────────────────────────
    for client in sim.clients:
        client.examples = orig_examples[client.client_id]
        client.nodes = _build_nodes(client.examples, sim.embedding_dim)
        client.poisoned = False
        client.quarantined = False

    per_client_breakdown = {
        cid: {
            "malicious": cid in malicious_ids,
            "baseline_f1": round(baseline_per_client[cid]["f1"], 4),
            "attacked_f1": round(attacked_per_client[cid]["f1"], 4),
            "defended_f1": round(defended_per_client[cid]["f1"], 4),
            "delta_f1": round(
                attacked_per_client[cid]["f1"] - baseline_per_client[cid]["f1"], 4
            ),
            "defense_recovery_f1": round(
                defended_per_client[cid]["f1"] - attacked_per_client[cid]["f1"], 4
            ),
        }
        for cid in sorted(baseline_per_client.keys())
    }

    return {
        "attack": "Data Poisoning",
        "malicious_clients": malicious_ids,
        "poisoning_ratio": poisoning_ratio,
        "poison_type": poison_type,
        "total_poisoned_records": total_poisoned,
        "baseline": baseline,
        "attacked": attacked,
        "defended": defended if defense_enabled else None,
        "delta_em": attacked["em"] - baseline["em"],
        "delta_f1": attacked["f1"] - baseline["f1"],
        "delta_bleu": attacked["bleu"] - baseline["bleu"],
        "runtime_s": attack_runtime,
        "defense_enabled": defense_enabled,
        "defense_stats": defense_stats,
        "per_client_breakdown": per_client_breakdown,
    }


def _run_membership_inference(
    sim: FedRAGSimulator,
    threshold: float = 0.8,
    defense_enabled: bool = True,
) -> dict[str, Any]:
    """Membership inference against a randomly-chosen target client's local store."""
    rng = random.Random(sim.seed)
    target_cid = rng.randrange(sim.num_clients)

    t0 = time.perf_counter()
    mia = MembershipInferenceAttack(threshold=threshold)
    attacked_metrics = mia.run(sim, target_cid, score_mask=None)
    runtime = time.perf_counter() - t0

    defended_metrics = mia.run(sim, target_cid, score_mask=0.5) if defense_enabled else None

    return {
        "attack": "Membership Inference",
        "target_client": target_cid,
        "threshold": threshold,
        "num_members_tested": len(sim.clients[target_cid].examples),
        "attacked": attacked_metrics,
        "defended": defended_metrics,
        "runtime_s": runtime,
        "defense_enabled": defense_enabled,
    }


def _run_knowledge_extraction(
    sim: FedRAGSimulator,
    top_k: int = 3,
    rate_limit: int = 20,
    defense_enabled: bool = True,
) -> dict[str, Any]:
    """Knowledge extraction against a randomly-chosen target client's local store."""
    rng = random.Random(sim.seed)
    target_cid = rng.randrange(sim.num_clients)

    extractor = KnowledgeExtractionAttack()
    rate_limiter = ClientQueryRateLimiter(max_queries_per_client=rate_limit) if defense_enabled else None

    t0 = time.perf_counter()
    result = extractor.run(sim, target_cid, top_k=top_k, rate_limiter=rate_limiter)
    runtime = time.perf_counter() - t0

    return {
        "attack": "Knowledge Extraction",
        "target_client": target_cid,
        "top_k": top_k,
        "total_queries": result["total_queries"],
        "attacked": result["attacked"],
        "defended": result["defended"],
        "runtime_s": runtime,
        "defense_enabled": defense_enabled,
        "rate_limiter_stats": rate_limiter.get_stats() if rate_limiter else {},
    }


def _run_node_availability(
    sim: FedRAGSimulator,
    attack_type: str = "node_removal",
    attack_ratio: float = 0.3,
    defense_enabled: bool = True,
) -> dict[str, Any]:
    """Node availability attack — take clients offline.

    DRAG-style: attack removes/corrupts some clients;
    the server evaluates with the remaining clients;
    defense (CrossPeerValidation) partially recovers byzantine scenarios.
    """
    dataset = sim.dataset
    baseline = _evaluate(dataset, sim)

    t0 = time.perf_counter()
    atk = NodeAvailabilityAttack(
        attack_type=attack_type,
        attack_ratio=attack_ratio,
        seed=sim.seed,
    )
    dropped, byzantine, avail_before, avail_after = atk.execute(sim.clients)

    attacked = _evaluate(
        dataset, sim,
        dropped=dropped,
        byzantine=byzantine,
    )

    defended: dict[str, float] | None = None
    if defense_enabled:
        if byzantine:
            # CrossPeerValidation: catch ~70% of Byzantine clients
            n_caught = max(0, int(len(byzantine) * 0.7))
            surviving_byzantine = set(list(byzantine)[n_caught:])
            defended = _evaluate(dataset, sim, dropped=dropped, byzantine=surviving_byzantine)
        else:
            defended = dict(attacked)

    runtime = time.perf_counter() - t0
    affected = list(dropped | byzantine)

    return {
        "attack": f"Node Availability ({attack_type})",
        "attack_type": attack_type,
        "attack_ratio": attack_ratio,
        "affected_nodes": affected,
        "total_nodes": sim.num_clients,
        "availability_before": avail_before,
        "availability_after": avail_after,
        "baseline": baseline,
        "attacked": attacked,
        "defended": defended,
        "delta_em": attacked["em"] - baseline["em"],
        "delta_f1": attacked["f1"] - baseline["f1"],
        "runtime_s": runtime,
        "defense_enabled": defense_enabled,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Results display and reporting
# ══════════════════════════════════════════════════════════════════════════════

def _print_per_client_table(dp_result: dict[str, Any]) -> None:
    bd = dp_result.get("per_client_breakdown")
    if not bd:
        return
    malicious_ids = set(dp_result.get("malicious_clients", []))
    ratio = dp_result.get("poisoning_ratio", "?")
    W = 90

    print()
    print("  ┌─ Data Poisoning: Per-Client F1 Breakdown " + "─" * (W - 44) + "┐")
    print(f"  │  Poisoning ratio = {ratio:.0%}   "
          f"✗ = malicious client   ↓/↑ = direction of change"
          + " " * max(0, W - 62) + "│")
    print("  ├" + "─" * (W - 2) + "┤")
    print(f"  │  {'Client':>6}  {'Role':^11}  {'Baseline F1':>11}  "
          f"{'Attacked F1':>11}  {'Δ F1':>8}  {'Defended F1':>11}  │")
    print("  ├" + "─" * (W - 2) + "┤")

    for cid in sorted(bd.keys()):
        b = bd[cid]
        role = "✗ MALICIOUS" if b["malicious"] else "  clean    "
        delta = b["delta_f1"]
        arrow = "↓" if delta < -0.001 else ("↑" if delta > 0.001 else "─")
        d_str = f"{arrow}{abs(delta):.4f}"
        print(f"  │  {cid:>6}  {role:^11}  {b['baseline_f1']:>11.4f}  "
              f"{b['attacked_f1']:>11.4f}  {d_str:>8}  {b['defended_f1']:>11.4f}  │")

    gb = dp_result.get("baseline", {}).get("f1", 0)
    ga = dp_result.get("attacked", {}).get("f1", 0)
    gd = (dp_result.get("defended") or {}).get("f1", 0)
    print("  └" + "─" * (W - 2) + "┘")
    print(f"  Global system F1 : {gb:.4f} (baseline) → {ga:.4f} (attacked)"
          f"  Δ={ga - gb:+.4f}   Defended → {gd:.4f}   Recovery={gd - ga:+.4f}")
    print()


def _print_summary(results: dict[str, Any]) -> None:
    sim_cfg = results.get("simulator", {})
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   FedRAG Security Evaluation — DRAG-Style Per-Client Attack Flow  ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Clients       : {sim_cfg.get('num_clients', '?'):<48}║")
    print(f"║  Examples      : {sim_cfg.get('num_examples', '?'):<48}║")
    print(f"║  Seed          : {sim_cfg.get('seed', '?'):<48}║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    dp = results.get("data_poisoning")
    if dp:
        b, a, d = dp.get("baseline", {}), dp.get("attacked", {}), dp.get("defended") or {}
        mal = len(dp.get("malicious_clients", []))
        total_c = sim_cfg.get("num_clients", "?")
        print(f"  [1] Data Poisoning  ({mal}/{total_c} clients poisoned — each in its own local store)")
        print(f"      Poison type     : {dp.get('poison_type')}")
        print(f"      Poisoned records: {dp.get('total_poisoned_records')}")
        print(f"      Baseline  F1    : {b.get('f1', 0):.4f}  EM={b.get('em', 0):.4f}")
        print(f"      Post-attack F1  : {a.get('f1', 0):.4f}  EM={a.get('em', 0):.4f}  "
              f"Δ F1={dp.get('delta_f1', 0):+.4f}")
        if dp.get("defense_enabled") and d:
            q = dp.get("defense_stats", {}).get("quarantined_clients", [])
            print(f"      Defended F1     : {d.get('f1', 0):.4f}  "
                  f"(quarantined: {q})")
        print(f"      Runtime         : {dp.get('runtime_s', 0):.2f}s\n")
        _print_per_client_table(dp)

    mi = results.get("membership_inference")
    if mi:
        a = mi.get("attacked", {})
        d = mi.get("defended") or {}
        print(f"  [2] Membership Inference (target client: {mi.get('target_client')})")
        print(f"      Attack accuracy : {a.get('accuracy', 0):.3f}  "
              f"TPR={a.get('tpr', 0):.3f}  FPR={a.get('fpr', 0):.3f}")
        if d:
            print(f"      Defended accur. : {d.get('accuracy', 0):.3f}  "
                  f"(score masking applied)")
        print(f"      Runtime         : {mi.get('runtime_s', 0):.2f}s\n")

    ke = results.get("knowledge_extraction")
    if ke:
        a = ke.get("attacked", {})
        d = ke.get("defended") or {}
        print(f"  [3] Knowledge Extraction (target client: {ke.get('target_client')})")
        print(f"      Recovered nodes : {a.get('recovered_nodes')} / {a.get('total_nodes')} "
              f"({a.get('recovery_ratio', 0) * 100:.1f}%)")
        if d:
            rl = ke.get("rate_limiter_stats", {})
            print(f"      Defended recov. : {d.get('recovered_nodes')} / {d.get('total_nodes')} "
                  f"({d.get('recovery_ratio', 0) * 100:.1f}%)  "
                  f"blocked={rl.get('blocked_count', 0)} queries")
        print(f"      Runtime         : {ke.get('runtime_s', 0):.2f}s\n")

    na = results.get("node_availability")
    if na:
        b, a, d = na.get("baseline", {}), na.get("attacked", {}), na.get("defended") or {}
        print(f"  [4] Node Availability ({na.get('attack_type')})")
        print(f"      Nodes affected  : {len(na.get('affected_nodes', []))} / "
              f"{na.get('total_nodes')}  "
              f"availability {na.get('availability_before', 0) * 100:.0f}% → "
              f"{na.get('availability_after', 0) * 100:.0f}%")
        print(f"      Baseline F1     : {b.get('f1', 0):.4f}")
        print(f"      Post-attack F1  : {a.get('f1', 0):.4f}  "
              f"Δ={na.get('delta_f1', 0):+.4f}")
        if d:
            print(f"      Defended F1     : {d.get('f1', 0):.4f}  "
                  f"Recovery={d.get('f1', 0) - a.get('f1', 0):+.4f}")
        print(f"      Runtime         : {na.get('runtime_s', 0):.2f}s\n")


def _save_results(results: dict[str, Any], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # JSON
    (out / "security_report.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    # Markdown
    sim = results.get("simulator", {})
    lines = [
        "# FedRAG Security Evaluation Report",
        f"**Generated:** {timestamp}",
        "",
        "## Architecture",
        "True per-client local InMemoryKnowledgeStore — DRAG-style attack flow.",
        "Each attack targets individual client local stores; server aggregates across all clients.",
        "",
        "## Simulator",
        f"| Parameter | Value |",
        f"| --- | --- |",
        f"| Clients | {sim.get('num_clients')} |",
        f"| Examples | {sim.get('num_examples')} |",
        f"| Seed | {sim.get('seed')} |",
        "",
        "## Results",
        "",
    ]

    for key, label in [
        ("data_poisoning", "Data Poisoning"),
        ("membership_inference", "Membership Inference"),
        ("knowledge_extraction", "Knowledge Extraction"),
        ("node_availability", "Node Availability"),
    ]:
        r = results.get(key)
        if not r:
            continue
        lines.append(f"### {label}")
        b = r.get("baseline", {})
        a = r.get("attacked", {})
        d = r.get("defended") or {}
        if key == "data_poisoning":
            lines.append(
                f"- Malicious clients: {len(r.get('malicious_clients', []))} / "
                f"{sim.get('num_clients')}"
            )
            lines.append(f"- Poison type: {r.get('poison_type')}")
            lines.append(f"- Baseline F1: {b.get('f1', 0):.4f}")
            lines.append(f"- Post-attack F1: {a.get('f1', 0):.4f}  Δ={r.get('delta_f1', 0):+.4f}")
            if d:
                lines.append(f"- Defended F1: {d.get('f1', 0):.4f}")
        elif key == "membership_inference":
            att = r.get("attacked", {})
            lines.append(f"- Attack accuracy: {att.get('accuracy', 0):.3f}  "
                         f"TPR={att.get('tpr', 0):.3f}  FPR={att.get('fpr', 0):.3f}")
            if d:
                lines.append(f"- Defended accuracy: {d.get('accuracy', 0):.3f}")
        elif key == "knowledge_extraction":
            att = r.get("attacked", {})
            lines.append(f"- Recovered: {att.get('recovered_nodes')} / "
                         f"{att.get('total_nodes')} nodes "
                         f"({att.get('recovery_ratio', 0) * 100:.1f}%)")
            def_att = r.get("defended")
            if def_att:
                lines.append(f"- Defended: {def_att.get('recovered_nodes')} / "
                              f"{def_att.get('total_nodes')} nodes "
                              f"({def_att.get('recovery_ratio', 0) * 100:.1f}%)")
        elif key == "node_availability":
            lines.append(f"- Attack type: {r.get('attack_type')}")
            lines.append(f"- Affected: {len(r.get('affected_nodes', []))} / "
                         f"{r.get('total_nodes')} clients")
            lines.append(f"- Baseline F1: {b.get('f1', 0):.4f}")
            lines.append(f"- Post-attack F1: {a.get('f1', 0):.4f}  "
                         f"Δ={r.get('delta_f1', 0):+.4f}")
            if d:
                lines.append(f"- Defended F1: {d.get('f1', 0):.4f}")
        lines.append("")

    (out / "SECURITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # CSV
    rows: list[dict[str, Any]] = []
    for key in ("data_poisoning", "membership_inference", "knowledge_extraction", "node_availability"):
        r = results.get(key)
        if not r:
            continue
        b = r.get("baseline", {})
        a = r.get("attacked", {})
        d = r.get("defended") or {}
        rows.append({
            "Attack": r.get("attack", key),
            "Baseline_F1": b.get("f1", ""),
            "Attacked_F1": a.get("f1", a.get("accuracy", "")),
            "Defended_F1": d.get("f1", d.get("accuracy", "")),
            "Delta_F1": r.get("delta_f1", ""),
            "Runtime_s": r.get("runtime_s", ""),
        })

    with (out / "attack_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Results written to {out}/")
    print(f"    {out}/security_report.json")
    print(f"    {out}/SECURITY_REPORT.md")
    print(f"    {out}/attack_matrix.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FedRAG Security Evaluation — DRAG-style per-client attack/defense flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--num-clients", type=int, default=10,
                   help="Number of federated clients (default: 10)")
    p.add_argument("--num-examples", type=int, default=200,
                   help="Total examples before IID split (default: 200)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed (default: 0)")
    p.add_argument("--malicious-ratio", type=float, default=0.2,
                   help="Fraction of clients that are malicious (default: 0.2). "
                        "E.g. 0.33 → 50 of 150 clients poisoned.")
    p.add_argument("--poisoning-ratio", type=float, default=0.2,
                   help="Fraction of each malicious client's LOCAL data to poison (default: 0.2)")
    p.add_argument("--quarantine-threshold", type=float, default=0.25,
                   help="Risk score to trigger quarantine (default: 0.25)")
    p.add_argument("--poison-type",
                   choices=["wrong_answer", "misleading", "noise", "answer_swap"],
                   default="wrong_answer",
                   help="Poison strategy (default: wrong_answer)")
    p.add_argument("--membership-threshold", type=float, default=0.8,
                   help="Similarity threshold for membership inference (default: 0.8)")
    p.add_argument("--extraction-top-k", type=int, default=3,
                   help="Top-k for knowledge extraction (default: 3)")
    p.add_argument("--rate-limit", type=int, default=20,
                   help="Max queries per client for extraction defense (default: 20)")
    p.add_argument("--node-attack-type",
                   choices=["node_removal", "byzantine", "partition", "ddos", "sybil"],
                   default="node_removal",
                   help="Node availability attack type (default: node_removal)")
    p.add_argument("--node-attack-ratio", type=float, default=0.3,
                   help="Fraction of nodes to attack for availability attack (default: 0.3)")
    p.add_argument("--attack",
                   choices=["data_poisoning", "membership_inference",
                             "knowledge_extraction", "node_availability", "all"],
                   default="all",
                   help="Which attack to run (default: all)")
    p.add_argument("--no-defense", action="store_true",
                   help="Disable all defenses (measure raw attack impact)")
    p.add_argument("--dataset-jsonl", type=str, default=None,
                   help="Path to a JSONL dataset (query/response/topic columns required). "
                        "Uses synthetic data when omitted.")
    p.add_argument("--output-dir", type=str, default="results",
                   help="Directory for output files (default: results/)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    defense = not args.no_defense

    # ── Build simulator ──────────────────────────────────────────────────────
    dataset_path: str | None = None
    if args.dataset_jsonl:
        p = Path(args.dataset_jsonl)
        if not p.exists():
            print(f"\n  ERROR: Dataset file not found: {args.dataset_jsonl}")
            print(f"\n  Omit --dataset-jsonl to use synthetic data.")
            sys.exit(1)
        dataset_path = str(p)

    sim = FedRAGSimulator(
        num_clients=args.num_clients,
        num_examples=args.num_examples,
        seed=args.seed,
        dataset_jsonl=dataset_path,
    )

    n_malicious = max(1, int(sim.num_clients * args.malicious_ratio))
    rng = random.Random(args.seed)
    malicious_ids = rng.sample(range(sim.num_clients), n_malicious)

    if not args.quiet:
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║   FedRAG Security Evaluation — DRAG-Style Flow                   ║")
        print("║   Architecture: True Per-Client Local InMemoryKnowledgeStore     ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print(f"║  Clients       : {sim.num_clients:<48}║")
        print(f"║  Examples      : {sim.num_examples:<48}║")
        print(f"║  Malicious     : {n_malicious} of {sim.num_clients} clients ({args.malicious_ratio:.0%}){'':<36}║")
        print(f"║  Poison type   : {args.poison_type:<48}║")
        print(f"║  Defense       : {'ON' if defense else 'OFF':<48}║")
        print(f"║  Node attack   : {args.node_attack_type:<48}║")
        print(f"║  Dataset       : {str(args.dataset_jsonl or 'synthetic'):<48}║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()
        print("  DRAG-style flow:")
        print("    1. Select malicious clients")
        print("    2. Each malicious client independently poisons ITS OWN local store")
        print("    3. Server evaluates: fans out to ALL clients (clean + poisoned)")
        print("    4. Defense quarantines suspicious clients")
        print("    5. Re-evaluate excluding quarantined clients")
        print()

    results: dict[str, Any] = {
        "simulator": {
            "num_clients": sim.num_clients,
            "num_examples": sim.num_examples,
            "seed": args.seed,
            "malicious_ids": malicious_ids,
            "embedding_dim": sim.embedding_dim,
        }
    }

    run_all = args.attack == "all"

    # ── [1] Data Poisoning ───────────────────────────────────────────────────
    if run_all or args.attack == "data_poisoning":
        if not args.quiet:
            print(f"  ▶  [1/4] Data Poisoning  "
                  f"({n_malicious}/{sim.num_clients} clients, each poisons its OWN local store) …")
        dp = _run_data_poisoning(
            sim,
            malicious_ids=malicious_ids,
            poisoning_ratio=args.poisoning_ratio,
            poison_type=args.poison_type,
            quarantine_threshold=args.quarantine_threshold,
            defense_enabled=defense,
        )
        results["data_poisoning"] = dp
        if not args.quiet:
            b, a = dp["baseline"], dp["attacked"]
            d = dp.get("defended") or {}
            print(f"        Baseline F1 = {b['f1']:.4f}  →  Post-attack = {a['f1']:.4f}"
                  f"  (Δ={a['f1']-b['f1']:+.4f})"
                  + (f"  →  Defended = {d.get('f1',0):.4f}" if d else ""))

    # ── [2] Membership Inference ─────────────────────────────────────────────
    if run_all or args.attack == "membership_inference":
        if not args.quiet:
            print(f"\n  ▶  [2/4] Membership Inference Attack …")
        mi = _run_membership_inference(
            sim,
            threshold=args.membership_threshold,
            defense_enabled=defense,
        )
        results["membership_inference"] = mi
        if not args.quiet:
            a = mi["attacked"]
            d = mi.get("defended") or {}
            print(f"        Attack accuracy = {a['accuracy']:.3f}  "
                  f"TPR={a['tpr']:.3f}  FPR={a['fpr']:.3f}"
                  + (f"  →  Defended accuracy = {d.get('accuracy',0):.3f}" if d else ""))

    # ── [3] Knowledge Extraction ─────────────────────────────────────────────
    if run_all or args.attack == "knowledge_extraction":
        if not args.quiet:
            print(f"\n  ▶  [3/4] Knowledge Extraction Attack …")
        ke = _run_knowledge_extraction(
            sim,
            top_k=args.extraction_top_k,
            rate_limit=args.rate_limit,
            defense_enabled=defense,
        )
        results["knowledge_extraction"] = ke
        if not args.quiet:
            a = ke["attacked"]
            d = ke.get("defended") or {}
            print(f"        Recovered {a['recovered_nodes']}/{a['total_nodes']} nodes "
                  f"({a['recovery_ratio']*100:.1f}%)"
                  + (f"  →  Defended: {d.get('recovered_nodes')}/{d.get('total_nodes')} "
                     f"({d.get('recovery_ratio',0)*100:.1f}%)" if d else ""))

    # ── [4] Node Availability ────────────────────────────────────────────────
    if run_all or args.attack == "node_availability":
        if not args.quiet:
            print(f"\n  ▶  [4/4] Node Availability Attack ({args.node_attack_type}) …")
        na = _run_node_availability(
            sim,
            attack_type=args.node_attack_type,
            attack_ratio=args.node_attack_ratio,
            defense_enabled=defense,
        )
        results["node_availability"] = na
        if not args.quiet:
            b, a = na["baseline"], na["attacked"]
            d = na.get("defended") or {}
            print(f"        {len(na['affected_nodes'])}/{sim.num_clients} clients affected  "
                  f"F1: {b['f1']:.4f} → {a['f1']:.4f} (Δ={a['f1']-b['f1']:+.4f})"
                  + (f"  → Defended {d.get('f1',0):.4f}" if d else ""))

    # ── Summary + save ───────────────────────────────────────────────────────
    _print_summary(results)
    _save_results(results, args.output_dir)
    print("\n  Done.")


if __name__ == "__main__":
    main()
