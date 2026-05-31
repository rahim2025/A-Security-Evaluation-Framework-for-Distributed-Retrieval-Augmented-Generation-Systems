"""Training-based federated security evaluator for FedRAG.

Simulates the full federated-learning loop for a retriever model:

1.  A global SentenceTransformer is initialised.
2.  Data is split IID across clients.
3.  Each client trains locally for *E* epochs on its own data.
4.  The server aggregates client weights via FedAvg.
5.  Rounds repeat.
6.  After training, the global model is evaluated on a hold-out test set.

Training-time attacks (``TrainingDataPoisoningAttack``,
``GradientManipulationAttack``) are injected at the local-training phase so
that corrupted updates propagate through aggregation and degrade the global
retriever.  This mirrors the modular approach described in the training-time
attack design: attacks are applied *during* local training, not to the
knowledge-store after the fact.

The evaluator produces the same matrix output format as
``centralized.py``, ``federated.py``, and ``system.py`` so that results can
be compared directly.
"""

from __future__ import annotations

import copy
import json
import random
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fed_rag.attacks.training.data_poisoning import TrainingDataPoisoningAttack
from fed_rag.attacks.training.gradient_manipulation import (
    GradientManipulationAttack,
)
from fed_rag.base.retriever import BaseRetriever
from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType
from fed_rag.defenses import ClientDataPoisoningDefense
from fed_rag.evaluators.config import DefenseConfig, SecurityConfig
from fed_rag.evaluators.metrics import evaluate_system_rag_answers
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.utils.evaluation_matrix import (
    write_evaluation_matrix,
    write_json_results,
)

# ------------------------------------------------------------------
# Optional sentence-transformers import
# ------------------------------------------------------------------

_ST_AVAILABLE = False
try:
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    _ST_AVAILABLE = True
except Exception:  # pragma: no cover
    pass

# ------------------------------------------------------------------
# Retriever wrapper
# ------------------------------------------------------------------


class _SimpleSTRetriever(BaseRetriever):
    """Lightweight wrapper around a SentenceTransformer model."""

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, model: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model = model

    def encode_query(self, query: Any, **kwargs: Any) -> torch.Tensor:
        texts = (
            [str(query)]
            if isinstance(query, str)
            else [str(q) for q in query]
        )
        return self._model.encode(texts, convert_to_tensor=True)

    def encode_context(self, context: Any, **kwargs: Any) -> torch.Tensor:
        texts = (
            [str(context)]
            if isinstance(context, str)
            else [str(c) for c in context]
        )
        return self._model.encode(texts, convert_to_tensor=True)

    @property
    def encoder(self) -> torch.nn.Module | None:
        return self._model

    @property
    def query_encoder(self) -> torch.nn.Module | None:
        return self._model

    @property
    def context_encoder(self) -> torch.nn.Module | None:
        return self._model


# ------------------------------------------------------------------
# Public runner
# ------------------------------------------------------------------


def run(
    *,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    seed: int,
    model_name: str,
    dataset: list[dict[str, str]],
    output_dir: Path,
    security_cfg: SecurityConfig,
    defense_cfg: DefenseConfig | None = None,
    attack: str = "none",
    malicious_clients: int = 0,
    poisoning_ratio: float = 0.35,
    noise_std: float = 0.5,
    scale: float = 2.0,
    split_test_ratio: float = 0.2,
    device: str | None = None,
) -> dict[str, Any]:
    """Run training-based federated security evaluation.

    Args:
        num_clients: number of simulated clients.
        num_rounds: number of federated aggregation rounds.
        local_epochs: local training epochs per client per round.
        batch_size: batch size for local training.
        seed: random seed.
        model_name: HuggingFace model id for the base retriever.
        dataset: list of {"query", "response", "topic"} dicts.
        output_dir: where to write results.
        security_cfg: parsed ``config/security.yaml``.
        defense_cfg: parsed ``config/defense.yaml`` (optional).
        attack: attack type – ``none``, ``data_poisoning``,
            ``gradient_flip``, ``gradient_noise``, ``gradient_scale``.
        malicious_clients: number of malicious clients.
        poisoning_ratio: ratio of examples to poison per malicious client.
        noise_std: std-dev for ``gradient_noise``.
        scale: multiplier for ``gradient_scale``.
        split_test_ratio: fraction of data to hold out for testing.
        device: torch device override (``None`` = auto).

    Returns:
        dict with raw results for programmatic access.
    """
    if not _ST_AVAILABLE:
        raise RuntimeError(
            "sentence-transformers is required for training-based evaluation."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    defense_cfg = defense_cfg or DefenseConfig()

    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # -- hold-out test set --
    shuffled = [dict(row) for row in dataset]
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1.0 - split_test_ratio))
    train_rows = shuffled[:split_idx]
    test_rows = shuffled[split_idx:]

    clients_data = _iid_split(train_rows, num_clients, rng)

    # -- pick malicious clients --
    malicious_ids: set[int] = set()
    if attack != "none":
        malicious_ids = set(
            rng.sample(
                range(num_clients),
                min(malicious_clients, num_clients),
            )
        )

    print(
        f"Dataset: {len(dataset)} total, {len(train_rows)} train, "
        f"{len(test_rows)} test"
    )
    print(f"Clients: {num_clients}, Rounds: {num_rounds}")
    print(
        f"Attack: {attack}, Malicious clients: {sorted(malicious_ids)}"
    )
    sys.stdout.flush()

    # -- initialise global retriever model --
    print(f"Loading base model: {model_name} …")
    sys.stdout.flush()
    global_model: Any = SentenceTransformer(model_name)
    if device is not None:
        global_model = global_model.to(device)

    round_records: list[dict[str, Any]] = []

    # ================================================================
    # Federated rounds
    # ================================================================
    for round_idx in range(num_rounds):
        local_states: list[dict[str, np.ndarray]] = []
        local_sizes: list[int] = []

        for cid, client_rows in enumerate(clients_data):
            # 1. Fresh copy of global model
            local_model: Any = copy.deepcopy(global_model)
            if device is not None:
                local_model = local_model.to(device)

            # 2. Prepare training examples
            train_examples = [
                InputExample(texts=[row["query"], row["response"]])
                for row in client_rows
            ]

            # 3. TRAINING-TIME ATTACK :: data poisoning
            if attack == "data_poisoning" and cid in malicious_ids:
                train_examples = _apply_data_poisoning(
                    train_examples,
                    poisoning_ratio=poisoning_ratio,
                    seed=seed + cid + round_idx,
                )

            # 4. Local training
            _local_train(
                local_model,
                train_examples,
                epochs=local_epochs,
                batch_size=batch_size,
            )

            # 5. TRAINING-TIME ATTACK :: gradient / model manipulation
            if (
                attack in ("gradient_flip", "gradient_noise", "gradient_scale")
                and cid in malicious_ids
            ):
                local_model = _apply_gradient_manipulation(
                    local_model,
                    attack=attack,
                    scale=scale,
                    noise_std=noise_std,
                    seed=seed + cid + round_idx,
                )

            local_states.append(_state_dict_to_numpy(local_model))
            local_sizes.append(len(client_rows))

        # 6. Server aggregates (FedAvg)
        global_state = _fedavg(local_states, local_sizes)
        _load_numpy_state_dict(global_model, global_state)

        # 7. Evaluate global retriever on test set
        metrics = _evaluate(global_model, train_rows, test_rows)
        round_records.append(
            {
                "round": round_idx + 1,
                "metrics": metrics,
            }
        )
        print(
            f"Round {round_idx + 1}: "
            f"em={metrics['exact_match']:.3f} "
            f"f1={metrics['f1']:.3f} "
            f"bleu={metrics['bleu']:.3f} "
            f"semantic={metrics['semantic_similarity']:.3f}"
        )
        sys.stdout.flush()

    # ================================================================
    # Defense evaluation (post-training)
    # ================================================================
    defended_metrics: dict[str, float] = {}
    if (
        defense_cfg.enabled
        and defense_cfg.client_data_poisoning_defense_enabled
        and attack == "data_poisoning"
    ):
        defended_metrics = _evaluate_with_defense(
            global_model=global_model,
            clients_data=clients_data,
            train_rows=train_rows,
            test_rows=test_rows,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            attack=attack,
            malicious_ids=malicious_ids,
            poisoning_ratio=poisoning_ratio,
            noise_std=noise_std,
            scale=scale,
            seed=seed,
            defense_cfg=defense_cfg,
            device=device,
        )

    # ================================================================
    # Build matrix records
    # ================================================================
    baseline_metrics = round_records[0]["metrics"] if round_records else {}
    final_metrics = round_records[-1]["metrics"] if round_records else {}

    records: list[dict[str, Any]] = []

    # Per-round records
    for rec in round_records:
        m = rec["metrics"]
        records.append(
            _matrix_record(
                attack=f"Training ({attack})",
                attack_phase=f"round_{rec['round']}",
                baseline=baseline_metrics,
                attacked=m,
                defended=defended_metrics if defended_metrics else m,
                runtime=0.0,
                notes=(
                    f"num_clients={num_clients}; round={rec['round']}; "
                    f"malicious={sorted(malicious_ids)}; "
                    f"attack={attack}; "
                    f"poisoning_ratio={poisoning_ratio}"
                ),
            )
        )

    # Final degradation record
    records.append(
        _matrix_record(
            attack=f"Training ({attack})",
            attack_phase="final_degradation",
            baseline=baseline_metrics,
            attacked=final_metrics,
            defended=defended_metrics if defended_metrics else final_metrics,
            runtime=0.0,
            notes=(
                f"num_clients={num_clients}; rounds={num_rounds}; "
                f"malicious={sorted(malicious_ids)}; attack={attack}; "
                f"local_epochs={local_epochs}; batch_size={batch_size}; "
                f"defense_enabled={defense_cfg.enabled}"
            ),
        )
    )

    # ================================================================
    # Write outputs
    # ================================================================
    raw_results = {
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "seed": seed,
        "model_name": model_name,
        "attack": attack,
        "malicious_ids": sorted(malicious_ids),
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "defended_metrics": defended_metrics,
        "round_records": round_records,
        "records": records,
    }

    write_json_results(raw_results, output_dir / "training_federated.json")
    write_evaluation_matrix(
        records,
        markdown_path=output_dir / "TRAINING_FEDERATED_MATRIX.md",
        csv_path=output_dir / "training_federated_matrix.csv",
    )

    print(f"\nResults written to {output_dir}")
    _print_summary(baseline_metrics, final_metrics, defended_metrics)

    return raw_results


# ------------------------------------------------------------------
# Attack helpers
# ------------------------------------------------------------------


def _apply_data_poisoning(
    train_examples: list[Any],
    *,
    poisoning_ratio: float,
    seed: int,
) -> list[Any]:
    """Apply training-time data poisoning to a list of InputExamples."""
    rows = [
        {"query": ex.texts[0], "response": ex.texts[1]}
        for ex in train_examples
    ]
    attack = TrainingDataPoisoningAttack(
        poisoning_ratio=poisoning_ratio,
        seed=seed,
    )
    poisoned_rows = attack.on_before_local_training(None, rows)
    return [
        InputExample(texts=[r["query"], r["response"]])
        for r in poisoned_rows
    ]


def _apply_gradient_manipulation(
    model: Any,
    *,
    attack: str,
    scale: float,
    noise_std: float,
    seed: int,
) -> Any:
    """Apply gradient/model manipulation to a SentenceTransformer."""
    attack_obj = GradientManipulationAttack(
        scale=scale if attack == "gradient_scale" else 1.0,
        noise_std=noise_std if attack == "gradient_noise" else 0.0,
        flip_sign=attack == "gradient_flip",
        seed=seed,
    )
    state = _state_dict_to_numpy(model)
    poisoned_state = {
        k: attack_obj.on_weights_ready([v])[0]
        for k, v in state.items()
    }
    _load_numpy_state_dict(model, poisoned_state)
    return model


# ------------------------------------------------------------------
# Defense evaluation
# ------------------------------------------------------------------


def _evaluate_with_defense(
    global_model: Any,
    clients_data: list[list[dict[str, str]]],
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    *,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    attack: str,
    malicious_ids: set[int],
    poisoning_ratio: float,
    noise_std: float,
    scale: float,
    seed: int,
    defense_cfg: DefenseConfig,
    device: str | None = None,
) -> dict[str, float]:
    """Re-run FL with ClientDataPoisoningDefense enabled."""
    model: Any = copy.deepcopy(global_model)
    if device is not None:
        model = model.to(device)

    rng = random.Random(seed)

    for round_idx in range(num_rounds):
        local_states: list[dict[str, np.ndarray]] = []
        local_sizes: list[int] = []
        all_client_rows: list[list[dict[str, str]]] = []

        for cid, client_rows in enumerate(clients_data):
            local_model = copy.deepcopy(model)
            if device is not None:
                local_model = local_model.to(device)

            train_examples = [
                InputExample(texts=[row["query"], row["response"]])
                for row in client_rows
            ]

            if attack == "data_poisoning" and cid in malicious_ids:
                train_examples = _apply_data_poisoning(
                    train_examples,
                    poisoning_ratio=poisoning_ratio,
                    seed=seed + cid + round_idx,
                )

            _local_train(
                local_model,
                train_examples,
                epochs=local_epochs,
                batch_size=batch_size,
            )

            local_states.append(_state_dict_to_numpy(local_model))
            local_sizes.append(len(client_rows))
            all_client_rows.append(client_rows)

        # -- apply defense --
        if defense_cfg.client_data_poisoning_defense_enabled:
            defense = ClientDataPoisoningDefense(
                quarantine_threshold=defense_cfg.client_data_poisoning_defense_quarantine_threshold,
            )
            weights = [
                {
                    "client_id": cid,
                    "data_size": sz,
                }
                for cid, sz in enumerate(local_sizes)
            ]
            # ClientDataPoisoningDefense expects list[list[dict]] for sanitize
            # but it operates on data examples.  Since our attack poisons data
            # before training, the *weights* already contain poisoned info.
            # We simulate defense by dropping quarantined client weights.
            # This is a reasonable proxy: if a client's data looks suspicious,
            # we drop their update entirely.
            sanitized_weights, inspections = defense.sanitize(all_client_rows)
            quarantined = {
                r.client_id for r in inspections if r.quarantined
            }
            filtered_states = [
                s for i, s in enumerate(local_states) if i not in quarantined
            ]
            filtered_sizes = [
                s for i, s in enumerate(local_sizes) if i not in quarantined
            ]
            if filtered_states:
                global_state = _fedavg(filtered_states, filtered_sizes)
            else:
                global_state = _fedavg(local_states, local_sizes)
        else:
            global_state = _fedavg(local_states, local_sizes)

        _load_numpy_state_dict(model, global_state)

    return _evaluate(model, train_rows, test_rows)


# ------------------------------------------------------------------
# Training / FedAvg / Evaluation helpers
# ------------------------------------------------------------------


def _local_train(
    model: Any,
    examples: list[Any],
    *,
    epochs: int,
    batch_size: int,
) -> None:
    if not examples:
        return
    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=0,
        show_progress_bar=False,
    )


def _state_dict_to_numpy(
    model: Any,
) -> dict[str, np.ndarray]:
    return {k: v.cpu().numpy() for k, v in model.state_dict().items()}


def _load_numpy_state_dict(
    model: Any, state: dict[str, np.ndarray]
) -> None:
    model.load_state_dict(
        {k: torch.tensor(v) for k, v in state.items()}
    )


def _fedavg(
    local_states: list[dict[str, np.ndarray]],
    local_sizes: list[int],
) -> dict[str, np.ndarray]:
    total = sum(local_sizes)
    avg_state: dict[str, np.ndarray] = {}
    for key in local_states[0].keys():
        weighted_sum = sum(
            state[key] * size
            for state, size in zip(local_states, local_sizes)
        )
        avg_state[key] = weighted_sum / total
    return avg_state


def _evaluate(
    model: Any,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
) -> dict[str, float]:
    store = _build_store(train_rows, model)
    retriever = _SimpleSTRetriever(model)
    return evaluate_system_rag_answers(test_rows, store, retriever)


def _build_store(
    rows: list[dict[str, str]], model: Any
) -> InMemoryKnowledgeStore:
    nodes: list[KnowledgeNode] = []
    for row in rows:
        emb = model.encode(row["query"], convert_to_tensor=True).tolist()
        # SentenceTransformer may return (1, dim) when batched
        if emb and isinstance(emb[0], list):
            emb = emb[0]
        nodes.append(
            KnowledgeNode(
                node_type=NodeType.TEXT,
                text_content=row["query"],
                embedding=[float(v) for v in emb],
                metadata={
                    "answer": row["response"],
                    "topic": row.get("topic", "unknown"),
                },
            )
        )
    return InMemoryKnowledgeStore.from_nodes(nodes)


# ------------------------------------------------------------------
# IID split
# ------------------------------------------------------------------


def _iid_split(
    dataset: list[dict[str, str]],
    num_clients: int,
    rng: random.Random,
) -> list[list[dict[str, str]]]:
    shuffled = [dict(row) for row in dataset]
    rng.shuffle(shuffled)
    base, remainder = divmod(len(shuffled), num_clients)
    clients: list[list[dict[str, str]]] = []
    offset = 0
    for cid in range(num_clients):
        size = base + (1 if cid < remainder else 0)
        clients.append(shuffled[offset : offset + size])
        offset += size
    return clients


# ------------------------------------------------------------------
# Dataset helpers (public so CLI can use them)
# ------------------------------------------------------------------


def make_synthetic_dataset(num_examples: int) -> list[dict[str, str]]:
    """Build a synthetic dataset for quick experiments."""
    topics = [
        "cryptography",
        "medicine",
        "finance",
        "biology",
        "law",
        "networking",
        "history",
        "physics",
        "privacy",
        "safety",
    ]
    rows: list[dict[str, str]] = []
    for idx in range(num_examples):
        topic = topics[idx % len(topics)]
        rows.append(
            {
                "query": (
                    f"Fact {idx}: what is the verified answer for {topic}?"
                ),
                "response": f"verified-{topic}-answer-{idx}",
                "topic": topic,
            }
        )
    return rows


def load_mmlu_dataset(num_examples: int | None = None) -> list[dict[str, str]]:
    """Load a slice of the MMLU test split."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except Exception as exc:
        raise RuntimeError(
            "HuggingFace `datasets` is required for MMLU."
        ) from exc

    ds = load_dataset(
        "cais/mmlu", "all", split="test", streaming=True
    )
    rows: list[dict[str, str]] = []
    for idx, row in enumerate(ds):
        if num_examples is not None and idx >= num_examples:
            break
        choices = row.get("choices", [])
        answer_idx = int(row.get("answer", 0))
        answer = (
            str(choices[answer_idx])
            if 0 <= answer_idx < len(choices)
            else str(answer_idx)
        )
        rows.append(
            {
                "query": str(row["question"]),
                "response": answer,
                "topic": str(row.get("subject", "unknown")),
            }
        )
    return rows


# ------------------------------------------------------------------
# Matrix / summary helpers
# ------------------------------------------------------------------


def _matrix_record(
    *,
    attack: str,
    attack_phase: str,
    baseline: dict[str, float],
    attacked: dict[str, float],
    defended: dict[str, float],
    runtime: float,
    notes: str,
) -> dict[str, Any]:
    return {
        "Attack": attack,
        "Phase": attack_phase,
        "System": "FedRAG Training FL",
        "Dataset": "synthetic-client-rag",
        "Model": "SentenceTransformer",
        "Baseline EM": baseline.get("exact_match"),
        "Post-Attack EM": attacked.get("exact_match"),
        "Defended EM": defended.get("exact_match"),
        "Delta EM": attacked.get("exact_match", 0.0)
        - baseline.get("exact_match", 0.0),
        "Baseline F1": baseline.get("f1"),
        "Post-Attack F1": attacked.get("f1"),
        "Defended F1": defended.get("f1"),
        "Delta F1": attacked.get("f1", 0.0)
        - baseline.get("f1", 0.0),
        "Baseline BLEU": baseline.get("bleu"),
        "Post-Attack BLEU": attacked.get("bleu"),
        "Defended BLEU": defended.get("bleu"),
        "Delta BLEU": attacked.get("bleu", 0.0)
        - baseline.get("bleu", 0.0),
        "Baseline Semantic Sim": baseline.get("semantic_similarity"),
        "Post-Attack Semantic Sim": attacked.get("semantic_similarity"),
        "Defended Semantic Sim": defended.get("semantic_similarity"),
        "Delta Semantic Sim": attacked.get("semantic_similarity", 0.0)
        - baseline.get("semantic_similarity", 0.0),
        "Baseline Precision": baseline.get("precision"),
        "Post-Attack Precision": attacked.get("precision"),
        "Baseline Recall": baseline.get("recall"),
        "Post-Attack Recall": attacked.get("recall"),
        "Query Failure Rate Baseline": baseline.get("query_failure_rate"),
        "Query Failure Rate Post-Attack": attacked.get("query_failure_rate"),
        "Runtime (s)": runtime,
        "Memory Overhead MB": _current_memory_mb(),
        "Notes": notes,
    }


def _current_memory_mb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return max_rss / (1024.0 * 1024.0)
    return max_rss / 1024.0


def _print_summary(
    baseline: dict[str, float],
    final: dict[str, float],
    defended: dict[str, float],
) -> None:
    print("\n" + "=" * 60)
    print("  TRAINING FEDERATED EVALUATION SUMMARY")
    print("=" * 60)
    for key in [
        "exact_match",
        "f1",
        "bleu",
        "semantic_similarity",
    ]:
        b = baseline.get(key, 0.0)
        f = final.get(key, 0.0)
        d = defended.get(key, 0.0) if defended else f
        delta = f - b
        defended_delta = d - b
        print(
            f"  {key:>20s}: "
            f"baseline={b:.4f}  "
            f"final={f:.4f}  "
            f"delta={delta:+.4f}  "
            f"defended={defended_delta:+.4f}"
        )
    print("=" * 60)
    sys.stdout.flush()
