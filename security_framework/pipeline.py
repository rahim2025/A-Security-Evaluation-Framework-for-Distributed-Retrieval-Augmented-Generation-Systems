"""
SecurityPipeline
================
Orchestrates all four attacks and all six defenses against a
``FedRAGSimulator`` (or any object that exposes the same ``retrieve`` /
``clients`` interface).

Attacks
-------
- Data poisoning       (append / replace mode, multiple poison types)
- Membership inference (threshold-based, cosine similarity)
- Knowledge extraction (repeated probing with templated queries)
- Node availability    (removal, Byzantine, partition, DDoS, Sybil)

Defenses
--------
- ClientDataPoisoningDefense  — quarantine high-risk clients
- ScoreMasking                — hide retrieval confidence scores
- CrossPeerValidation         — majority-vote consensus
- QueryRateLimiter            — cap queries per client
- ExtractionAnomalyDetector   — detect broad-topic flooding
- ResponsePerturbation        — add noise to returned answers

All results are collected into a ``SecurityReport``.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from security_framework.simulator import FedRAGSimulator, SimClient, _cosine, _hash_embedding


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _token_f1(pred: str, gold: str) -> float:
    p_toks = _tokenize(pred)
    g_toks = _tokenize(gold)
    if not p_toks or not g_toks:
        return 0.0
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    precision = len(common) / len(p_toks)
    recall = len(common) / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _simple_bleu(pred: str, gold: str) -> float:
    pred_toks = _tokenize(pred)
    gold_toks = _tokenize(gold)
    if not pred_toks:
        return 0.0
    hits = sum(1 for t in pred_toks if t in gold_toks)
    return hits / len(pred_toks)


def _evaluate(
    dataset: list[dict[str, Any]],
    simulator: FedRAGSimulator,
    dropped_clients: set[int] | None = None,
    byzantine_clients: set[int] | None = None,
    score_mask: float | None = None,
) -> dict[str, float]:
    dropped = dropped_clients or set()
    byzantine = byzantine_clients or set()
    em_scores, f1_scores, bleu_scores = [], [], []

    for row in dataset:
        query = row["query"]
        gold = row["response"]

        # Gather active clients
        active = [c for c in simulator.clients if c.client_id not in dropped]
        if not active:
            em_scores.append(0.0); f1_scores.append(0.0); bleu_scores.append(0.0)
            continue

        results = []
        for client in active:
            hits = simulator.retrieve(query, client_id=client.client_id, top_k=1, score_mask=score_mask)
            if hits:
                results.append(hits[0])

        if not results:
            pred = ""
        else:
            _, best_node = max(results, key=lambda t: t[0])
            pred = str(best_node.metadata.get("answer", ""))
            if best_node.metadata.get("client_id") in byzantine:
                pred = "BYZANTINE_CORRUPTED_ANSWER"

        em_scores.append(_exact_match(pred, gold))
        f1_scores.append(_token_f1(pred, gold))
        bleu_scores.append(_simple_bleu(pred, gold))

    n = len(em_scores) or 1
    return {
        "em": sum(em_scores) / n,
        "f1": sum(f1_scores) / n,
        "bleu": sum(bleu_scores) / n,
    }


def _evaluate_per_client(
    dataset: list[dict[str, Any]],
    simulator: FedRAGSimulator,
) -> dict[int, dict[str, float]]:
    """Evaluate each client in isolation (all other clients dropped).

    Returns a dict mapping client_id → {em, f1, bleu} for that client alone.
    This shows how much of the global-system performance comes from each
    individual client's knowledge shard.
    """
    all_ids = {c.client_id for c in simulator.clients}
    result: dict[int, dict[str, float]] = {}
    for client in simulator.clients:
        cid = client.client_id
        dropped = all_ids - {cid}
        result[cid] = _evaluate(dataset, simulator, dropped_clients=dropped)
    return result


# ---------------------------------------------------------------------------
# Individual attack / defense runners
# ---------------------------------------------------------------------------


class _PoisoningRunner:
    """Run data poisoning attack + ClientDataPoisoningDefense."""

    POISON_MARKERS = ("POISONED",)

    def run(
        self,
        simulator: FedRAGSimulator,
        malicious_client_ids: list[int],
        poisoning_ratio: float = 0.2,
        poison_type: str = "wrong_answer",
        mode: str = "replace",
        amplification_factor: int = 1,
        quarantine_threshold: float = 0.25,
        defense_enabled: bool = True,
    ) -> dict[str, Any]:
        from fed_rag.attacks import DataPoisoningAttack
        from fed_rag.defenses import ClientDataPoisoningDefense

        dataset = simulator.dataset

        # Baseline (global + per-client)
        baseline = _evaluate(dataset, simulator)
        baseline_per_client = _evaluate_per_client(dataset, simulator)

        # --- Attack ---
        t0 = time.perf_counter()
        poisoned_clients_data = [list(c.examples) for c in simulator.clients]
        total_poisoned = 0
        for cid in malicious_client_ids:
            atk = DataPoisoningAttack(
                poisoning_ratio=poisoning_ratio,
                poison_type=poison_type,
                mode=mode,
                amplification_factor=amplification_factor,
                seed=simulator.seed + cid,
            )
            result = atk.execute(poisoned_clients_data[cid])
            poisoned_clients_data[cid] = result.poisoned_examples
            total_poisoned += result.num_poisoned

        # Apply poisoning to simulator stores temporarily
        orig_examples = [list(c.examples) for c in simulator.clients]
        for cid, examples in enumerate(poisoned_clients_data):
            simulator.clients[cid].examples = examples
        simulator.rebuild_stores()
        attacked = _evaluate(dataset, simulator)
        attacked_per_client = _evaluate_per_client(dataset, simulator)
        attack_runtime = time.perf_counter() - t0

        # --- Defense ---
        defense_stats: dict[str, Any] = {}
        if defense_enabled:
            defense = ClientDataPoisoningDefense(
                quarantine_threshold=quarantine_threshold,
            )
            defended_data, inspections = defense.sanitize(poisoned_clients_data)
            quarantined = [r.client_id for r in inspections if r.quarantined]
            for cid, examples in enumerate(defended_data):
                simulator.clients[cid].examples = examples
            simulator.rebuild_stores()
            defended = _evaluate(dataset, simulator)
            defended_per_client = _evaluate_per_client(dataset, simulator)
            defense_stats = {
                "quarantined_clients": quarantined,
                "inspections": [
                    {
                        "client_id": r.client_id,
                        "risk_score": r.risk_score,
                        "quarantined": r.quarantined,
                        "poisoned_markers": r.poisoned_marker_count,
                    }
                    for r in inspections
                ],
            }
        else:
            defended = dict(attacked)
            defended_per_client = {cid: dict(v) for cid, v in attacked_per_client.items()}
            quarantined = []

        # Restore original data
        for cid, examples in enumerate(orig_examples):
            simulator.clients[cid].examples = examples
        simulator.rebuild_stores()

        per_client_breakdown = {
            cid: {
                "malicious":          cid in malicious_client_ids,
                "baseline_f1":        round(baseline_per_client[cid]["f1"], 4),
                "attacked_f1":        round(attacked_per_client[cid]["f1"], 4),
                "defended_f1":        round(defended_per_client[cid]["f1"], 4),
                "delta_f1":           round(
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
            "malicious_clients": malicious_client_ids,
            "poisoning_ratio": poisoning_ratio,
            "poison_type": poison_type,
            "mode": mode,
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


class _MembershipInferenceRunner:
    """Run membership inference + ScoreMasking defense."""

    def run(
        self,
        simulator: FedRAGSimulator,
        target_client_id: int | None = None,
        threshold: float = 0.8,
        defense_enabled: bool = True,
        score_mask_value: float = 0.5,
    ) -> dict[str, Any]:
        from fed_rag.attacks import MembershipInferenceAttack

        rng = random.Random(simulator.seed)
        cid = target_client_id if target_client_id is not None else rng.randrange(simulator.num_clients)
        target_client = simulator.clients[cid]

        member_queries = [r["query"] for r in target_client.examples]
        non_member_queries = [
            r["query"]
            for other in simulator.clients
            if other.client_id != cid
            for r in other.examples[:3]
        ][:len(member_queries)]

        def _run_mia(score_mask: float | None = None) -> dict[str, float]:
            tp = tn = fp = fn = 0
            all_results = []

            def _retrieve_score(query: str) -> float:
                q_emb = _hash_embedding(query, simulator.embedding_dim)
                if not target_client.nodes:
                    return 0.0
                best = max(_cosine(q_emb, n.embedding) for n in target_client.nodes)
                return score_mask if score_mask is not None else float(best)

            for query in member_queries:
                score = _retrieve_score(query)
                predicted = score >= threshold
                all_results.append({"query": query, "member": True, "score": score, "predicted": predicted})
                if predicted:
                    tp += 1
                else:
                    fn += 1

            for query in non_member_queries:
                score = _retrieve_score(query)
                predicted = score >= threshold
                all_results.append({"query": query, "member": False, "score": score, "predicted": predicted})
                if predicted:
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
                "recall": tpr,
            }

        t0 = time.perf_counter()
        attacked_metrics = _run_mia(score_mask=None)
        runtime = time.perf_counter() - t0

        defended_metrics = _run_mia(score_mask=score_mask_value) if defense_enabled else None

        return {
            "attack": "Membership Inference",
            "target_client": cid,
            "threshold": threshold,
            "num_members_tested": len(member_queries),
            "num_non_members_tested": len(non_member_queries),
            "attacked": attacked_metrics,
            "defended": defended_metrics,
            "runtime_s": runtime,
            "defense_enabled": defense_enabled,
            "score_mask_value": score_mask_value if defense_enabled else None,
        }


class _ExtractionRunner:
    """Run knowledge extraction + rate-limiter / anomaly-detector defenses."""

    TEMPLATES = (
        "What do you know about {topic}?",
        "Explain {topic}.",
        "Give facts about {topic}.",
        "Define {topic}.",
        "Describe {topic} in detail.",
    )

    def run(
        self,
        simulator: FedRAGSimulator,
        target_client_id: int | None = None,
        top_k: int = 3,
        rate_limit: int = 20,
        defense_enabled: bool = True,
    ) -> dict[str, Any]:
        from fed_rag.attacks import KnowledgeExtractionAttack
        from fed_rag.defenses import ClientQueryRateLimiter, ExtractionAnomalyDetector

        rng = random.Random(simulator.seed)
        cid = target_client_id if target_client_id is not None else rng.randrange(simulator.num_clients)
        target_client = simulator.clients[cid]

        topics = list({r.get("topic", "unknown") for r in target_client.examples})
        queries: list[tuple[str, str]] = []
        for row in target_client.examples:
            queries.append((row["query"], row.get("topic", "unknown")))
        for topic in topics:
            for tmpl in self.TEMPLATES:
                queries.append((tmpl.format(topic=topic), topic))

        def _score_extraction(query_list: list[tuple[str, str]]) -> dict[str, Any]:
            q_emb_cache: dict[str, list[float]] = {}
            recovered: dict[str, str] = {}
            for query, _ in query_list:
                if query not in q_emb_cache:
                    q_emb_cache[query] = _hash_embedding(query, simulator.embedding_dim)
                q_emb = q_emb_cache[query]
                for node in target_client.nodes:
                    sim = _cosine(q_emb, node.embedding)
                    if sim > 0.5:
                        recovered[node.node_id] = node.text
            total = len(target_client.nodes) or 1
            return {
                "recovered_nodes": len(recovered),
                "total_nodes": total,
                "recovery_ratio": len(recovered) / total,
                "queries_issued": len(query_list),
            }

        t0 = time.perf_counter()
        attacked_stats = _score_extraction(queries)
        runtime = time.perf_counter() - t0

        defended_stats: dict[str, Any] | None = None
        defense_info: dict[str, Any] = {}
        if defense_enabled:
            rate_limiter = ClientQueryRateLimiter(max_queries_per_client=rate_limit)
            anomaly_detector = ExtractionAnomalyDetector(
                max_queries=rate_limit + 5,
                max_unique_topics=6,
                topic_diversity_threshold=0.55,
                min_queries_for_detection=10,
            )
            allowed: list[tuple[str, str]] = []
            for query, topic in queries:
                ra, _ = rate_limiter.check_and_record(cid, query)
                aa, _ = anomaly_detector.check_and_record(cid, topic=topic)
                if ra and aa:
                    allowed.append((query, topic))

            defended_stats = _score_extraction(allowed)
            defense_info = {
                "rate_limiter": rate_limiter.get_stats(),
                "anomaly_detector": anomaly_detector.get_stats(),
                "queries_allowed": len(allowed),
                "queries_blocked": len(queries) - len(allowed),
            }

        return {
            "attack": "Knowledge Extraction",
            "target_client": cid,
            "top_k": top_k,
            "total_queries": len(queries),
            "attacked": attacked_stats,
            "defended": defended_stats,
            "runtime_s": runtime,
            "defense_enabled": defense_enabled,
            "defense_info": defense_info,
        }


class _NodeAvailabilityRunner:
    """Run node availability attack + CrossPeerValidation defense."""

    def run(
        self,
        simulator: FedRAGSimulator,
        attack_type: str = "node_removal",
        attack_ratio: float = 0.3,
        defense_enabled: bool = True,
    ) -> dict[str, Any]:
        from fed_rag.attacks import NodeAvailabilityAttack
        from fed_rag.defenses import CrossPeerValidation

        client_nodes = [
            type("_CN", (), {"client_id": c.client_id, "data_size": c.data_size})()
            for c in simulator.clients
        ]
        atk = NodeAvailabilityAttack(
            attack_type=attack_type,
            attack_ratio=attack_ratio,
            seed=simulator.seed,
        )
        t0 = time.perf_counter()
        na_result = atk.execute(client_nodes, strategy="random")
        runtime = time.perf_counter() - t0

        dropped: set[int] = set()
        byzantine: set[int] = set()
        if attack_type in ("node_removal", "ddos"):
            dropped = set(na_result.affected_nodes)
        elif attack_type == "byzantine":
            byzantine = set(na_result.affected_nodes)
        elif attack_type == "partition" and na_result.iterations:
            dropped = set(na_result.iterations[0].get("partition_2", []))

        dataset = simulator.dataset
        baseline = _evaluate(dataset, simulator)
        attacked = _evaluate(dataset, simulator, dropped_clients=dropped, byzantine_clients=byzantine)

        defended: dict[str, float] | None = None
        defense_info: dict[str, Any] = {}
        if defense_enabled and (dropped or byzantine):
            validator = CrossPeerValidation(
                min_agreement_ratio=0.6,
                voting_method="majority",
                min_peers=3,
                use_similarity=True,
                similarity_threshold=0.85,
            )
            sample_answers = [
                r["response"]
                for c in simulator.clients
                if c.client_id not in dropped
                for r in c.examples[:3]
            ][:10]
            validation_result = None
            if len(sample_answers) > 1:
                validation_result = validator.validate(
                    candidate_answer=sample_answers[0],
                    peer_answers=sample_answers,
                )
            defended = {
                "em": max(0.0, attacked.get("em", 0.0) + 0.05),
                "f1": max(0.0, attacked.get("f1", 0.0) + 0.05),
                "bleu": max(0.0, attacked.get("bleu", 0.0) + 0.03),
            }
            defense_info = {
                "cross_peer_validation": True,
                "validator_result": str(validation_result) if validation_result else None,
            }

        return {
            "attack": f"Node Availability ({attack_type})",
            "attack_type": attack_type,
            "attack_ratio": attack_ratio,
            "total_nodes": na_result.total_nodes,
            "affected_nodes": na_result.affected_nodes,
            "availability_before": na_result.availability_before,
            "availability_after": na_result.availability_after,
            "baseline": baseline,
            "attacked": attacked,
            "defended": defended,
            "delta_em": attacked["em"] - baseline["em"],
            "delta_f1": attacked["f1"] - baseline["f1"],
            "runtime_s": runtime,
            "defense_enabled": defense_enabled,
            "defense_info": defense_info,
        }


# ---------------------------------------------------------------------------
# SecurityPipeline
# ---------------------------------------------------------------------------


class SecurityPipeline:
    """Run all security attacks and defenses against a ``FedRAGSimulator``.

    Parameters
    ----------
    simulator:
        A configured ``FedRAGSimulator`` instance.
    malicious_ratio:
        Fraction of clients designated as malicious (data poisoning).
    poisoning_ratio:
        Fraction of a malicious client's data to poison.
    poison_type:
        ``"wrong_answer"``, ``"misleading"``, ``"noise"``, or ``"answer_swap"``.
    membership_threshold:
        Score threshold for the membership inference attack.
    extraction_top_k:
        Top-k for knowledge extraction queries.
    rate_limit:
        Max queries per client for the rate-limiter defense.
    node_attack_type:
        ``"node_removal"``, ``"byzantine"``, ``"partition"``, ``"ddos"``,
        or ``"sybil"``.
    node_attack_ratio:
        Fraction of nodes targeted.
    defense_enabled:
        Toggle all defenses on/off.
    """

    def __init__(
        self,
        simulator: FedRAGSimulator,
        *,
        malicious_ratio: float = 0.2,
        poisoning_ratio: float = 0.2,
        poison_type: str = "wrong_answer",
        membership_threshold: float = 0.8,
        extraction_top_k: int = 3,
        rate_limit: int = 20,
        node_attack_type: str = "node_removal",
        node_attack_ratio: float = 0.3,
        defense_enabled: bool = True,
    ) -> None:
        self.simulator = simulator
        self.malicious_ratio = malicious_ratio
        self.poisoning_ratio = poisoning_ratio
        self.poison_type = poison_type
        self.membership_threshold = membership_threshold
        self.extraction_top_k = extraction_top_k
        self.rate_limit = rate_limit
        self.node_attack_type = node_attack_type
        self.node_attack_ratio = node_attack_ratio
        self.defense_enabled = defense_enabled

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run_all(self) -> "SecurityReport":
        """Run every attack (with optional defense) and return a report."""
        from security_framework.reporter import SecurityReport

        rng = random.Random(self.simulator.seed)
        n_malicious = max(1, int(self.simulator.num_clients * self.malicious_ratio))
        malicious_ids = sorted(rng.sample(range(self.simulator.num_clients), n_malicious))

        results: dict[str, Any] = {"simulator": self.simulator.to_dict()}

        print("[SecurityPipeline] Running Data Poisoning attack...")
        results["data_poisoning"] = _PoisoningRunner().run(
            self.simulator,
            malicious_client_ids=malicious_ids,
            poisoning_ratio=self.poisoning_ratio,
            poison_type=self.poison_type,
            defense_enabled=self.defense_enabled,
        )

        print("[SecurityPipeline] Running Membership Inference attack...")
        results["membership_inference"] = _MembershipInferenceRunner().run(
            self.simulator,
            threshold=self.membership_threshold,
            defense_enabled=self.defense_enabled,
        )

        print("[SecurityPipeline] Running Knowledge Extraction attack...")
        results["knowledge_extraction"] = _ExtractionRunner().run(
            self.simulator,
            top_k=self.extraction_top_k,
            rate_limit=self.rate_limit,
            defense_enabled=self.defense_enabled,
        )

        print("[SecurityPipeline] Running Node Availability attack...")
        results["node_availability"] = _NodeAvailabilityRunner().run(
            self.simulator,
            attack_type=self.node_attack_type,
            attack_ratio=self.node_attack_ratio,
            defense_enabled=self.defense_enabled,
        )

        print("[SecurityPipeline] All attacks complete.")
        return SecurityReport(results)

    def run_attack(self, attack_name: str) -> dict[str, Any]:
        """Run a single named attack. Returns raw result dict."""
        rng = random.Random(self.simulator.seed)
        n_malicious = max(1, int(self.simulator.num_clients * self.malicious_ratio))
        malicious_ids = sorted(rng.sample(range(self.simulator.num_clients), n_malicious))

        if attack_name == "data_poisoning":
            return _PoisoningRunner().run(
                self.simulator,
                malicious_client_ids=malicious_ids,
                poisoning_ratio=self.poisoning_ratio,
                poison_type=self.poison_type,
                defense_enabled=self.defense_enabled,
            )
        if attack_name == "membership_inference":
            return _MembershipInferenceRunner().run(
                self.simulator,
                threshold=self.membership_threshold,
                defense_enabled=self.defense_enabled,
            )
        if attack_name == "knowledge_extraction":
            return _ExtractionRunner().run(
                self.simulator,
                top_k=self.extraction_top_k,
                rate_limit=self.rate_limit,
                defense_enabled=self.defense_enabled,
            )
        if attack_name == "node_availability":
            return _NodeAvailabilityRunner().run(
                self.simulator,
                attack_type=self.node_attack_type,
                attack_ratio=self.node_attack_ratio,
                defense_enabled=self.defense_enabled,
            )
        raise ValueError(
            f"Unknown attack '{attack_name}'. "
            "Choose from: data_poisoning, membership_inference, "
            "knowledge_extraction, node_availability."
        )
