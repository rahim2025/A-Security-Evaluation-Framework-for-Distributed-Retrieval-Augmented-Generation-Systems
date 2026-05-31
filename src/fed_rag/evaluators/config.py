"""Configuration helpers for FedRAG security evaluations.

Loads ``config/security.yaml`` (attack switches) and ``config/defense.yaml``
(defence switches) in the same style as DRAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SecurityConfig:
    enable_attack: bool = False
    poisoning_ratio: float = 0.3
    attack_strategy: str = "random"
    poison_type: str = "answer_swap"
    target_peer_ids: list[int] | None = None
    amplification_factor: int = 3
    question_variants: int = 2
    enable_membership_inference: bool = True
    membership_threshold: float = 0.8
    membership_top_k: int = 1
    enable_extraction: bool = False
    extraction_attacker_peer: int | None = None
    extraction_queries_per_topic: int = 5
    extraction_attack_type: str = "external"
    extraction_use_topic_inference: bool = False
    extraction_use_dataset_questions: bool = False
    enable_node_availability: bool = False
    node_attack_type: str = "node_removal"
    node_attack_ratio: float = 0.3


@dataclass(frozen=True)
class DefenseConfig:
    enabled: bool = False
    client_data_poisoning_defense_enabled: bool = False
    client_data_poisoning_defense_quarantine_threshold: float = 0.8
    score_masking_enabled: bool = False
    score_masking_public_score: float = 0.5
    cross_peer_validation_enabled: bool = False
    cross_peer_validation_min_agreement_ratio: float = 0.6
    cross_peer_validation_voting_method: str = "majority"
    cross_peer_validation_min_peers: int = 3
    cross_peer_validation_use_similarity: bool = True
    cross_peer_validation_similarity_threshold: float = 0.85
    query_rate_limiter_enabled: bool = False
    query_rate_limiter_max_queries: int = 18
    extraction_anomaly_detector_enabled: bool = False
    extraction_anomaly_detector_max_queries: int = 20
    extraction_anomaly_detector_max_unique_topics: int = 6
    extraction_anomaly_detector_topic_diversity_threshold: float = 0.6
    extraction_anomaly_detector_min_queries: int = 10
    response_perturbation_enabled: bool = False
    response_perturbation_level: float = 0.15
    response_perturbation_mode: str = "noise"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to read config files.") from exc
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload


def load_security_config(
    config_root: Path,
    security_path: Path | None = None,
) -> tuple[SecurityConfig, Path]:
    path = security_path or (config_root / "config" / "security.yaml")
    security = _load_yaml(path).get("security", {})

    return SecurityConfig(
        enable_attack=bool(security.get("enable_attack", False)),
        poisoning_ratio=float(security.get("poisoning_ratio", 0.3)),
        attack_strategy=str(security.get("attack_strategy", "random")),
        poison_type=str(security.get("poison_type", "answer_swap")),
        target_peer_ids=security.get("target_peer_ids"),
        amplification_factor=int(security.get("amplification_factor", 3)),
        question_variants=int(security.get("question_variants", 2)),
        enable_membership_inference=bool(
            security.get("enable_membership_inference", True)
        ),
        membership_threshold=float(
            security.get("membership_threshold", 0.8)
        ),
        membership_top_k=int(security.get("membership_top_k", 1)),
        enable_extraction=bool(
            security.get("enable_extraction", False)
        ),
        extraction_attacker_peer=security.get("extraction_attacker_peer"),
        extraction_queries_per_topic=int(
            security.get("extraction_queries_per_topic", 5)
        ),
        extraction_attack_type=str(
            security.get("extraction_attack_type", "external")
        ),
        extraction_use_topic_inference=bool(
            security.get("extraction_use_topic_inference", False)
        ),
        extraction_use_dataset_questions=bool(
            security.get("extraction_use_dataset_questions", False)
        ),
        enable_node_availability=bool(
            security.get("enable_node_availability", False)
        ),
        node_attack_type=str(
            security.get("node_attack_type", "node_removal")
        ),
        node_attack_ratio=float(
            security.get("node_attack_ratio", 0.3)
        ),
    ), path


def load_defense_config(
    config_root: Path,
    defense_path: Path | None = None,
) -> tuple[DefenseConfig, Path]:
    path = defense_path or (config_root / "config" / "defense.yaml")
    defense = _load_yaml(path).get("defense", {})

    cpv = defense.get("cross_peer_validation", {})
    qrl = defense.get("query_rate_limiter", {})
    ead = defense.get("extraction_anomaly_detector", {})
    rp = defense.get("response_perturbation", {})
    cdp = defense.get("client_data_poisoning_defense", {})
    sm = defense.get("score_masking", {})

    return DefenseConfig(
        enabled=bool(defense.get("enabled", False)),
        client_data_poisoning_defense_enabled=bool(
            cdp.get("enabled", False)
        ),
        client_data_poisoning_defense_quarantine_threshold=float(
            cdp.get("quarantine_threshold", 0.8)
        ),
        score_masking_enabled=bool(sm.get("enabled", False)),
        score_masking_public_score=float(sm.get("public_score", 0.5)),
        cross_peer_validation_enabled=bool(cpv.get("enabled", False)),
        cross_peer_validation_min_agreement_ratio=float(
            cpv.get("min_agreement_ratio", 0.6)
        ),
        cross_peer_validation_voting_method=str(
            cpv.get("voting_method", "majority")
        ),
        cross_peer_validation_min_peers=int(
            cpv.get("min_peers_for_validation", 3)
        ),
        cross_peer_validation_use_similarity=bool(
            cpv.get("use_similarity_matching", True)
        ),
        cross_peer_validation_similarity_threshold=float(
            cpv.get("similarity_threshold", 0.85)
        ),
        query_rate_limiter_enabled=bool(qrl.get("enabled", False)),
        query_rate_limiter_max_queries=int(
            qrl.get("max_queries_per_client", 18)
        ),
        extraction_anomaly_detector_enabled=bool(ead.get("enabled", False)),
        extraction_anomaly_detector_max_queries=int(
            ead.get("max_queries_threshold", 20)
        ),
        extraction_anomaly_detector_max_unique_topics=int(
            ead.get("max_unique_topics_threshold", 6)
        ),
        extraction_anomaly_detector_topic_diversity_threshold=float(
            ead.get("topic_diversity_threshold", 0.6)
        ),
        extraction_anomaly_detector_min_queries=int(
            ead.get("min_queries_for_detection", 10)
        ),
        response_perturbation_enabled=bool(rp.get("enabled", False)),
        response_perturbation_level=float(
            rp.get("perturbation_level", 0.15)
        ),
        response_perturbation_mode=str(rp.get("mode", "noise")),
    ), path
