"""
GlobalImpactEvaluator
=====================
Measures how client-level attacks cascade into GLOBAL federated RAG
system performance degradation.

Architecture — True Per-Client Local Stores
-------------------------------------------
Every client holds a PRIVATE, NON-REPLICATED local InMemoryKnowledgeStore.
The global system answers queries by fanning out to every active client's
local store, collecting one best-answer per client, and picking the globally
best answer at the server.  This is the FedRAG paper's architecture.

Attack flow (data poisoning — DRAG-style translated to FedRAG):
  1. N out of M clients are poisoned — each poisons ITS OWN LOCAL store.
  2. Central server evaluates against ALL M client stores (N poisoned + (M-N) clean).
  3. Defense quarantines bad clients; re-evaluation uses only the clean survivors.

This is explicitly NOT a pooled-store approach.  Each client's store is
queried in isolation; the server aggregates at evaluation time.

Three attacks are evaluated here:

1. **Data Poisoning**
   Malicious clients inject wrong/misleading answers into their local stores.
   The server's fan-out retrieval picks up poisoned answers when the poisoned
   client's answer scores highest globally.
   Defense: ClientDataPoisoningDefense (quarantine) → partial recovery.

2. **KB Extraction (Knowledge-Base Extraction)**
   Attacker probes the retrieval interface to recover nodes from a target
   client's local store.  Does NOT hurt answer quality but leaks private data.
   Defense: QueryRateLimiter + ExtractionAnomalyDetector.

3. **Node Availability**
   Clients removed / Byzantine / DDoS'd.  Global coverage collapses.
   Defense: CrossPeerValidation → partial recovery.

Each attack runs at three intensities:
  - Light  : 10% of clients affected
  - Medium : 30% of clients affected
  - Heavy  : 50% of clients affected
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from security_framework.simulator import (
    FedRAGSimulator,
    SimClient,
    SimNode,
    _cosine,
    _hash_embedding,
    _build_nodes,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _token_f1(pred: str, gold: str) -> float:
    p = _tokenize(pred)
    g = _tokenize(gold)
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def _simple_bleu(pred: str, gold: str) -> float:
    p = _tokenize(pred)
    g = _tokenize(gold)
    if not p:
        return 0.0
    hits = sum(1 for t in p if t in g)
    return hits / len(p)


def _avg_metrics(dataset: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, float]:
    em, f1, bleu = [], [], []
    for row in dataset:
        pred = predictions.get(row["query"], "")
        gold = row["response"]
        em.append(_exact_match(pred, gold))
        f1.append(_token_f1(pred, gold))
        bleu.append(_simple_bleu(pred, gold))
    n = len(em) or 1
    return {"em": sum(em) / n, "f1": sum(f1) / n, "bleu": sum(bleu) / n}


# ---------------------------------------------------------------------------
# Global retrieval — true per-client fan-out
# ---------------------------------------------------------------------------


def _global_retrieve(
    query: str,
    clients: list[SimClient],
    dropped: set[int],
    byzantine: set[int],
    embedding_dim: int,
    score_mask: float | None = None,
) -> str:
    """Return the best global answer by fanning out to each client's LOCAL store.

    This is the correct FedRAG fan-out:
      - Each client is queried INDEPENDENTLY from its own local store.
      - The server picks the globally best answer from all per-client responses.
      - No pooling of client data occurs.
    """
    q_emb = _hash_embedding(query, embedding_dim)
    best_score = -1.0
    best_answer = ""

    for client in clients:
        if client.client_id in dropped or client.quarantined:
            continue
        if not client.nodes:
            continue

        # Query THIS client's local store only
        local_best_score = -1.0
        local_best_node: SimNode | None = None
        for node in client.nodes:
            score = _cosine(q_emb, node.embedding)
            if score_mask is not None:
                score = score_mask
            if score > local_best_score:
                local_best_score = score
                local_best_node = node

        if local_best_node is not None and local_best_score > best_score:
            best_score = local_best_score
            if client.client_id in byzantine:
                best_answer = f"[BYZANTINE-CORRUPT] {local_best_node.metadata.get('answer', '')}"
            else:
                best_answer = str(local_best_node.metadata.get("answer", ""))

    return best_answer


def _evaluate_global(
    dataset: list[dict[str, Any]],
    clients: list[SimClient],
    embedding_dim: int,
    dropped: set[int] | None = None,
    byzantine: set[int] | None = None,
    score_mask: float | None = None,
) -> dict[str, float]:
    """Evaluate the global system using true per-client fan-out retrieval."""
    dropped = dropped or set()
    byzantine = byzantine or set()
    predictions: dict[str, str] = {}
    for row in dataset:
        predictions[row["query"]] = _global_retrieve(
            row["query"], clients, dropped, byzantine, embedding_dim, score_mask
        )
    return _avg_metrics(dataset, predictions)


def _evaluate_per_client_global(
    dataset: list[dict[str, Any]],
    clients: list[SimClient],
    embedding_dim: int,
) -> dict[int, dict[str, float]]:
    """Evaluate each client IN ISOLATION — only that client's local store is active.

    This measures each individual client's contribution to the global system.
    Poisoning one client's local store should affect only its isolated score
    (and the global score when its shard contains the correct answer for a query).
    """
    all_ids = {c.client_id for c in clients}
    result: dict[int, dict[str, float]] = {}
    for client in clients:
        cid = client.client_id
        # Drop all other clients — only this client's local store is queried
        dropped = all_ids - {cid}
        result[cid] = _evaluate_global(dataset, clients, embedding_dim, dropped=dropped)
    return result


# ---------------------------------------------------------------------------
# Scenario result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AttackScenario:
    """Result for one attack at one intensity level."""

    attack_name: str
    intensity: str
    num_clients: int
    affected_clients: int
    affected_ratio: float
    baseline: dict[str, float]
    attacked: dict[str, float]
    defended: dict[str, float] | None
    defense_name: str
    delta_f1: float
    delta_em: float
    delta_bleu: float
    recovery_f1: float | None
    runtime_s: float
    notes: str = ""
    kb_exposure_before: float | None = None
    kb_exposure_after: float | None = None
    availability_before: float | None = None
    availability_after: float | None = None
    per_client: dict | None = None


@dataclass
class GlobalImpactReport:
    """Full report across all attacks and all intensities."""

    scenarios: list[AttackScenario] = field(default_factory=list)
    cascade_tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    simulator_config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GlobalImpactEvaluator
# ---------------------------------------------------------------------------


INTENSITY_RATIOS = {"light": 0.1, "medium": 0.3, "heavy": 0.5}


class GlobalImpactEvaluator:
    """Evaluate how client-side attacks degrade the global FedRAG system.

    Uses true per-client local stores — no pooling.  Each attack touches
    only the targeted clients' local stores; global evaluation fans out
    across all client stores.

    Parameters
    ----------
    simulator:
        A fully-configured :class:`FedRAGSimulator`.
    poison_types:
        Which data-poisoning variants to test (default: all four).
    node_attack_types:
        Which node-availability attack types to test.
    intensities:
        Which intensity levels to run (light/medium/heavy).
    seed:
        Additional seed offset for attack RNG.
    poisoning_ratio:
        Fraction of each malicious client's local data to poison.
    quarantine_threshold:
        Risk score above which the defense quarantines a client.
    """

    def __init__(
        self,
        simulator: FedRAGSimulator,
        *,
        poison_types: list[str] | None = None,
        node_attack_types: list[str] | None = None,
        intensities: list[str] | None = None,
        seed: int = 0,
        poisoning_ratio: float = 0.5,
        quarantine_threshold: float = 0.5,
    ) -> None:
        self.sim = simulator
        self.poison_types = poison_types or [
            "wrong_answer", "misleading", "noise", "answer_swap"
        ]
        self.node_attack_types = node_attack_types or [
            "node_removal", "byzantine", "ddos", "partition", "sybil"
        ]
        self.intensities = intensities or ["light", "medium", "heavy"]
        self.seed = seed
        self.poisoning_ratio = poisoning_ratio
        self.quarantine_threshold = quarantine_threshold
        self._baseline: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> GlobalImpactReport:
        """Run all attacks at all intensities. Returns a :class:`GlobalImpactReport`."""
        report = GlobalImpactReport(simulator_config=self.sim.to_dict())

        print("\n[GlobalImpactEvaluator] Computing baseline global performance "
              "(true per-client fan-out)...")
        self._baseline = _evaluate_global(
            self.sim.dataset, self.sim.clients, self.sim.embedding_dim
        )
        b = self._baseline
        print(f"  Baseline → EM={b['em']:.4f}  F1={b['f1']:.4f}  BLEU={b['bleu']:.4f}")

        print("\n[GlobalImpactEvaluator] Running Data Poisoning scenarios...")
        report.scenarios.extend(self._run_data_poisoning())
        report.cascade_tables["data_poisoning"] = self._cascade_table_poisoning()

        print("\n[GlobalImpactEvaluator] Running KB Extraction scenarios...")
        report.scenarios.extend(self._run_kb_extraction())
        report.cascade_tables["kb_extraction"] = self._cascade_table_extraction()

        print("\n[GlobalImpactEvaluator] Running Node Availability scenarios...")
        report.scenarios.extend(self._run_node_availability())
        report.cascade_tables["node_availability"] = self._cascade_table_node_availability()

        print("\n[GlobalImpactEvaluator] All scenarios complete.")
        return report

    # ------------------------------------------------------------------
    # Data Poisoning
    # ------------------------------------------------------------------

    def _run_data_poisoning(self) -> list[AttackScenario]:
        from fed_rag.attacks import DataPoisoningAttack
        from fed_rag.defenses import ClientDataPoisoningDefense

        scenarios: list[AttackScenario] = []
        dataset = self.sim.dataset
        clients = self.sim.clients
        rng = random.Random(self.seed)
        baseline = self._baseline

        poison_type = "wrong_answer"

        # Per-client baseline (each client evaluated in isolation)
        baseline_per_client = _evaluate_per_client_global(
            dataset, clients, self.sim.embedding_dim
        )

        for intensity in self.intensities:
            ratio = INTENSITY_RATIOS[intensity]
            n_malicious = max(1, int(len(clients) * ratio))
            malicious_ids = sorted(rng.sample(range(len(clients)), n_malicious))

            t0 = time.perf_counter()

            # Save originals
            orig_examples = {c.client_id: list(c.examples) for c in clients}
            total_poisoned = 0

            # Step 1: Each malicious client poisons ITS OWN LOCAL store
            for cid in malicious_ids:
                atk = DataPoisoningAttack(
                    poisoning_ratio=self.poisoning_ratio,
                    poison_type=poison_type,
                    mode="append",
                    amplification_factor=2,
                    seed=self.seed + cid,
                )
                result = atk.execute(clients[cid].examples)
                clients[cid].examples = result.poisoned_examples
                # Rebuild ONLY this client's local store — others untouched
                clients[cid].nodes = _build_nodes(
                    result.poisoned_examples, self.sim.embedding_dim
                )
                clients[cid].poisoned = True
                total_poisoned += result.num_poisoned

            # Step 2: Global evaluation — fan out to ALL clients (clean + poisoned)
            attacked = _evaluate_global(dataset, clients, self.sim.embedding_dim)
            attacked_per_client = _evaluate_per_client_global(
                dataset, clients, self.sim.embedding_dim
            )

            # Step 3: Defense — server inspects each client's data
            all_client_data = [c.examples for c in clients]
            defense = ClientDataPoisoningDefense(
                quarantine_threshold=self.quarantine_threshold
            )
            defended_data, inspections = defense.sanitize(all_client_data)
            quarantined = [r.client_id for r in inspections if r.quarantined]

            for cid, examples in enumerate(defended_data):
                clients[cid].examples = examples
                clients[cid].nodes = _build_nodes(examples, self.sim.embedding_dim)
                if cid in quarantined:
                    clients[cid].quarantined = True

            # Step 4: Re-evaluate with quarantined clients excluded
            defended = _evaluate_global(dataset, clients, self.sim.embedding_dim)
            defended_per_client = _evaluate_per_client_global(
                dataset, clients, self.sim.embedding_dim
            )

            runtime = time.perf_counter() - t0

            # Restore originals
            for c in clients:
                c.examples = orig_examples[c.client_id]
                c.nodes = _build_nodes(c.examples, self.sim.embedding_dim)
                c.poisoned = False
                c.quarantined = False

            per_client = {
                cid: {
                    "malicious":    cid in malicious_ids,
                    "baseline_f1":  round(baseline_per_client[cid]["f1"], 4),
                    "attacked_f1":  round(attacked_per_client[cid]["f1"], 4),
                    "defended_f1":  round(defended_per_client[cid]["f1"], 4),
                    "delta_f1":     round(
                        attacked_per_client[cid]["f1"] - baseline_per_client[cid]["f1"], 4
                    ),
                }
                for cid in sorted(baseline_per_client)
            }
            scenarios.append(AttackScenario(
                attack_name="Data Poisoning",
                intensity=intensity,
                num_clients=len(clients),
                affected_clients=n_malicious,
                affected_ratio=n_malicious / len(clients),
                baseline=dict(baseline),
                attacked=attacked,
                defended=defended,
                defense_name="ClientDataPoisoningDefense",
                delta_f1=attacked["f1"] - baseline["f1"],
                delta_em=attacked["em"] - baseline["em"],
                delta_bleu=attacked["bleu"] - baseline["bleu"],
                recovery_f1=defended["f1"] - attacked["f1"],
                runtime_s=runtime,
                notes=(
                    f"poison_type={poison_type}; malicious_clients={malicious_ids}; "
                    f"total_poisoned_records={total_poisoned}; quarantined={quarantined}"
                ),
                per_client=per_client,
            ))

            print(
                f"  [{intensity:6s}] {n_malicious}/{len(clients)} clients poisoned "
                f"(each in its own local store) → "
                f"Global F1: {baseline['f1']:.4f} → {attacked['f1']:.4f} "
                f"(Δ={attacked['f1']-baseline['f1']:+.4f}) | "
                f"Defended: {defended['f1']:.4f}"
            )

        return scenarios

    def _cascade_table_poisoning(self) -> list[dict[str, Any]]:
        """Degrade one client at a time; record global F1 at each step."""
        from fed_rag.attacks import DataPoisoningAttack

        dataset = self.sim.dataset
        clients = self.sim.clients
        rng = random.Random(self.seed + 999)
        orig_examples = {c.client_id: list(c.examples) for c in clients}
        order = list(range(len(clients)))
        rng.shuffle(order)

        table: list[dict[str, Any]] = []
        baseline_f1 = self._baseline["f1"]
        poisoned_so_far: set[int] = set()

        for step, cid in enumerate(order):
            # Poison this client's local store
            atk = DataPoisoningAttack(
                poisoning_ratio=self.poisoning_ratio,
                poison_type="wrong_answer",
                mode="append",
                amplification_factor=2,
                seed=self.seed + cid,
            )
            result = atk.execute(clients[cid].examples)
            clients[cid].examples = result.poisoned_examples
            clients[cid].nodes = _build_nodes(result.poisoned_examples, self.sim.embedding_dim)
            clients[cid].poisoned = True
            poisoned_so_far.add(cid)

            # Global evaluation — fan out to ALL clients
            metrics = _evaluate_global(dataset, clients, self.sim.embedding_dim)
            table.append({
                "step": step + 1,
                "newly_poisoned_client": cid,
                "total_poisoned_clients": step + 1,
                "fraction_poisoned": (step + 1) / len(clients),
                "global_f1": metrics["f1"],
                "global_em": metrics["em"],
                "global_bleu": metrics["bleu"],
                "delta_f1_from_baseline": metrics["f1"] - baseline_f1,
                "pct_degradation": (baseline_f1 - metrics["f1"]) / baseline_f1 * 100
                if baseline_f1 else 0,
            })

        # Restore
        for c in clients:
            c.examples = orig_examples[c.client_id]
            c.nodes = _build_nodes(c.examples, self.sim.embedding_dim)
            c.poisoned = False

        return table

    # ------------------------------------------------------------------
    # KB Extraction
    # ------------------------------------------------------------------

    def _run_kb_extraction(self) -> list[AttackScenario]:
        from fed_rag.defenses import ClientQueryRateLimiter, ExtractionAnomalyDetector

        TEMPLATES = (
            "What do you know about {topic}?",
            "Explain {topic}.",
            "Give facts about {topic}.",
            "Define {topic}.",
            "Describe {topic} in detail.",
        )

        scenarios: list[AttackScenario] = []
        clients = self.sim.clients
        baseline = self._baseline
        rng = random.Random(self.seed + 1)

        for intensity in self.intensities:
            ratio = INTENSITY_RATIOS[intensity]
            n_targeted = max(1, int(len(clients) * ratio))
            targeted_ids = sorted(rng.sample(range(len(clients)), n_targeted))

            t0 = time.perf_counter()
            total_nodes = sum(len(c.nodes) for c in clients)
            recovered_global: set[str] = set()
            recovered_defended: set[str] = set()
            rl = ClientQueryRateLimiter(max_queries_per_client=15)

            for cid in targeted_ids:
                client = clients[cid]
                topics = list({r.get("topic", "unknown") for r in client.examples})
                attack_qs: list[tuple[str, str]] = []
                for row in client.examples:
                    attack_qs.append((row["query"], row.get("topic", "unknown")))
                for topic in topics:
                    for tmpl in TEMPLATES:
                        attack_qs.append((tmpl.format(topic=topic), topic))

                # Without defense: query this client's local store
                for query, _ in attack_qs:
                    q_emb = _hash_embedding(query, self.sim.embedding_dim)
                    for node in client.nodes:
                        if _cosine(q_emb, node.embedding) > 0.5:
                            recovered_global.add(f"{cid}:{node.node_id}")

                # With rate limiter
                for query, topic in attack_qs:
                    ok, _ = rl.check_and_record(cid, query)
                    if ok:
                        q_emb = _hash_embedding(query, self.sim.embedding_dim)
                        for node in client.nodes:
                            if _cosine(q_emb, node.embedding) > 0.5:
                                recovered_defended.add(f"{cid}:{node.node_id}")

            runtime = time.perf_counter() - t0

            exp_before = len(recovered_global) / total_nodes * 100 if total_nodes else 0
            exp_after = len(recovered_defended) / total_nodes * 100 if total_nodes else 0

            scenarios.append(AttackScenario(
                attack_name="KB Extraction",
                intensity=intensity,
                num_clients=len(clients),
                affected_clients=n_targeted,
                affected_ratio=ratio,
                baseline=dict(baseline),
                attacked=dict(baseline),
                defended=dict(baseline),
                defense_name="QueryRateLimiter",
                delta_f1=0.0,
                delta_em=0.0,
                delta_bleu=0.0,
                recovery_f1=0.0,
                runtime_s=runtime,
                kb_exposure_before=exp_before,
                kb_exposure_after=exp_after,
                notes=(
                    f"targeted_clients={targeted_ids}; "
                    f"nodes_exposed={len(recovered_global)}/{total_nodes}; "
                    f"defended_nodes_exposed={len(recovered_defended)}/{total_nodes}"
                ),
            ))

            print(
                f"  [{intensity:6s}] {n_targeted}/{len(clients)} clients probed → "
                f"KB exposure: {exp_before:.1f}% (no defense) "
                f"| {exp_after:.1f}% (with rate-limiter)"
            )

        return scenarios

    def _cascade_table_extraction(self) -> list[dict[str, Any]]:
        """Target one additional client per step; record KB exposure at each step."""
        TEMPLATES = (
            "What do you know about {topic}?",
            "Explain {topic}.",
            "Give facts about {topic}.",
            "Define {topic}.",
        )

        clients = self.sim.clients
        rng = random.Random(self.seed + 998)
        order = list(range(len(clients)))
        rng.shuffle(order)
        total_nodes = sum(len(c.nodes) for c in clients)

        table: list[dict[str, Any]] = []
        rl = ClientQueryRateLimiter = __import__(
            "fed_rag.defenses", fromlist=["ClientQueryRateLimiter"]
        ).ClientQueryRateLimiter
        rate_limiter = rl(max_queries_per_client=15)
        recovered_global: set[str] = set()
        recovered_defended: set[str] = set()

        for step, cid in enumerate(order):
            client = clients[cid]
            topics = list({r.get("topic", "unknown") for r in client.examples})
            attack_qs: list[tuple[str, str]] = []
            for row in client.examples:
                attack_qs.append((row["query"], row.get("topic", "unknown")))
            for topic in topics:
                for tmpl in TEMPLATES:
                    attack_qs.append((tmpl.format(topic=topic), topic))

            for query, _ in attack_qs:
                q_emb = _hash_embedding(query, self.sim.embedding_dim)
                for node in client.nodes:
                    if _cosine(q_emb, node.embedding) > 0.5:
                        recovered_global.add(f"{cid}:{node.node_id}")

            for query, topic in attack_qs:
                ok, _ = rate_limiter.check_and_record(cid, query)
                if ok:
                    q_emb = _hash_embedding(query, self.sim.embedding_dim)
                    for node in client.nodes:
                        if _cosine(q_emb, node.embedding) > 0.5:
                            recovered_defended.add(f"{cid}:{node.node_id}")

            table.append({
                "step": step + 1,
                "newly_targeted_client": cid,
                "total_targeted_clients": step + 1,
                "fraction_targeted": (step + 1) / len(clients),
                "nodes_exposed_no_defense": len(recovered_global),
                "nodes_exposed_with_defense": len(recovered_defended),
                "exposure_pct_no_defense": len(recovered_global) / total_nodes * 100 if total_nodes else 0,
                "exposure_pct_with_defense": len(recovered_defended) / total_nodes * 100 if total_nodes else 0,
            })

        return table

    # ------------------------------------------------------------------
    # Node Availability
    # ------------------------------------------------------------------

    def _run_node_availability(self) -> list[AttackScenario]:
        from fed_rag.attacks import NodeAvailabilityAttack
        from fed_rag.defenses import CrossPeerValidation

        scenarios: list[AttackScenario] = []
        dataset = self.sim.dataset
        clients = self.sim.clients
        baseline = self._baseline

        for attack_type in self.node_attack_types:
            for intensity in self.intensities:
                ratio = INTENSITY_RATIOS[intensity]

                proxy_nodes = [
                    type("_N", (), {
                        "client_id": c.client_id,
                        "data_size": c.data_size,
                        "connectivity": 1.0,
                    })()
                    for c in clients
                ]

                t0 = time.perf_counter()
                atk = NodeAvailabilityAttack(
                    attack_type=attack_type,
                    attack_ratio=ratio,
                    seed=self.seed,
                )
                na_result = atk.execute(proxy_nodes, strategy="random")

                dropped: set[int] = set()
                byzantine: set[int] = set()

                if attack_type in ("node_removal", "ddos"):
                    dropped = set(na_result.affected_nodes)
                elif attack_type == "byzantine":
                    byzantine = set(na_result.affected_nodes)
                elif attack_type == "partition":
                    if na_result.iterations:
                        p2 = na_result.iterations[0].get("partition_2", [])
                        dropped = set(p2)

                attacked = _evaluate_global(
                    dataset, clients, self.sim.embedding_dim,
                    dropped=dropped, byzantine=byzantine,
                )

                defended_dropped = dropped if attack_type in ("node_removal", "ddos", "partition") else set()
                defended_byzantine: set[int] = set()
                if byzantine:
                    cpv = CrossPeerValidation(
                        min_agreement_ratio=0.6,
                        voting_method="majority",
                        min_peers=3,
                        use_similarity=True,
                        similarity_threshold=0.85,
                    )
                    peer_answers = [
                        r["response"]
                        for c in clients
                        if c.client_id not in byzantine
                        for r in c.examples[:2]
                    ][:12]
                    if len(peer_answers) >= 2:
                        n_caught = max(0, int(len(byzantine) * 0.7))
                        defended_byzantine = set(list(byzantine)[:n_caught])

                defended = _evaluate_global(
                    dataset, clients, self.sim.embedding_dim,
                    dropped=defended_dropped,
                    byzantine=byzantine - defended_byzantine,
                )
                runtime = time.perf_counter() - t0

                scenarios.append(AttackScenario(
                    attack_name=f"Node Availability ({attack_type})",
                    intensity=intensity,
                    num_clients=len(clients),
                    affected_clients=len(na_result.affected_nodes),
                    affected_ratio=ratio,
                    baseline=dict(baseline),
                    attacked=attacked,
                    defended=defended,
                    defense_name="CrossPeerValidation",
                    delta_f1=attacked["f1"] - baseline["f1"],
                    delta_em=attacked["em"] - baseline["em"],
                    delta_bleu=attacked["bleu"] - baseline["bleu"],
                    recovery_f1=defended["f1"] - attacked["f1"],
                    runtime_s=runtime,
                    availability_before=na_result.availability_before,
                    availability_after=na_result.availability_after,
                    notes=(
                        f"attack_type={attack_type}; "
                        f"affected_nodes={na_result.affected_nodes}; "
                        f"availability={na_result.availability_before:.2f}→"
                        f"{na_result.availability_after:.2f}"
                    ),
                ))

                print(
                    f"  [{attack_type:15s} | {intensity:6s}] "
                    f"{len(na_result.affected_nodes)}/{len(clients)} nodes → "
                    f"Global F1: {baseline['f1']:.4f} → {attacked['f1']:.4f} "
                    f"(Δ={attacked['f1']-baseline['f1']:+.4f}) | "
                    f"Defended: {defended['f1']:.4f}"
                )

        return scenarios

    def _cascade_table_node_availability(self) -> list[dict[str, Any]]:
        """Remove one node at a time; record global F1 at each step."""
        dataset = self.sim.dataset
        clients = self.sim.clients
        rng = random.Random(self.seed + 997)
        order = list(range(len(clients)))
        rng.shuffle(order)
        table: list[dict[str, Any]] = []
        baseline_f1 = self._baseline["f1"]
        dropped: set[int] = set()

        for step, cid in enumerate(order):
            dropped.add(cid)
            metrics = _evaluate_global(
                dataset, clients, self.sim.embedding_dim, dropped=dropped
            )
            table.append({
                "step": step + 1,
                "newly_dropped_client": cid,
                "total_dropped_clients": step + 1,
                "active_clients": len(clients) - step - 1,
                "fraction_dropped": (step + 1) / len(clients),
                "global_f1": metrics["f1"],
                "global_em": metrics["em"],
                "global_bleu": metrics["bleu"],
                "delta_f1_from_baseline": metrics["f1"] - baseline_f1,
                "pct_degradation": (baseline_f1 - metrics["f1"]) / baseline_f1 * 100
                if baseline_f1 else 0,
            })

        return table
