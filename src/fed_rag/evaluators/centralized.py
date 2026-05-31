"""Centralized security evaluation for FedRAG.

Runs data poisoning, membership inference, and knowledge extraction against a
single knowledge store backed by a real sentence-transformer retriever.
"""

from __future__ import annotations

import csv
import json
import resource
import time
from pathlib import Path
from typing import Any

from fed_rag.attacks import (
    DataPoisoningAttack,
    KnowledgeExtractionAttack,
    MembershipInferenceAttack,
)
from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType
from fed_rag.evaluators.config import SecurityConfig
from fed_rag.evaluators.metrics import (
    evaluate_lookup,
    jaccard,
    simple_bleu,
    token_f1,
    tokenize,
)
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.evaluators.metrics import HashingRetriever
from fed_rag.utils.evaluation_matrix import (
    write_evaluation_matrix,
    write_json_results,
)

DRAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def run(
    *,
    config_root: Path,
    llm_name: str,
    dataset_name: str,
    num_samples: int | None,
    seed: int,
    membership_threshold: float,
    top_k: int,
    output_dir: Path,
    use_ollama: bool,
    max_generation_examples: int | None,
    dataset_source: str,
    security_cfg: SecurityConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    drag_config = _load_drag_config(config_root, llm_name, dataset_name)
    dataset = _load_drag_dataset(
        config_root,
        drag_config,
        num_samples,
        seed,
        dataset_source,
    )
    dataset_path = output_dir / "drag_compatible_dataset.jsonl"
    _write_jsonl(dataset, dataset_path)

    retriever = HFSentenceTransformerRetriever(
        model_name=DRAG_EMBEDDING_MODEL
    )
    store = _build_knowledge_store(dataset, retriever)

    baseline_metrics = _evaluate_rag_answers(
        dataset=dataset,
        retriever=retriever,
        knowledge_store=store,
        llm_config=drag_config["llm"],
        use_ollama=use_ollama,
        max_examples=max_generation_examples,
    )
    records: list[dict[str, Any]] = []
    model = drag_config["llm"].get("name", "llama3.2:3b")
    ds_name = (
        f'{drag_config["data"]["load"]["path"]}'
        f'/{drag_config["data"]["load"]["name"]}'
        f':{drag_config["data"]["load"]["split"]}'
    )

    # --- Data Poisoning ---
    start = time.perf_counter()
    poisoned = DataPoisoningAttack(
        poisoning_ratio=0.1,
        poison_type="wrong_answer",
        mode="replace",
        seed=seed,
    ).execute(dataset)
    poisoning_runtime = time.perf_counter() - start
    poisoned_lookup = {
        row["query"]: row["response"] for row in poisoned.poisoned_examples
    }
    poisoned_store = _build_knowledge_store(
        poisoned.poisoned_examples, retriever
    )
    poisoned_metrics = _evaluate_rag_answers(
        dataset=dataset,
        retriever=retriever,
        knowledge_store=poisoned_store,
        llm_config=drag_config["llm"],
        use_ollama=use_ollama,
        fallback_lookup=poisoned_lookup,
        max_examples=max_generation_examples,
    )
    records.append(
        _make_matrix_record(
            attack="Data Poisoning",
            system="FedRAG",
            dataset=ds_name,
            model=model,
            baseline=baseline_metrics,
            attacked=poisoned_metrics,
            runtime=poisoning_runtime,
            memory_mb=current_memory_mb(),
            notes=f"DRAG-compatible dataset copied to {dataset_path}",
        )
    )

    # --- Membership Inference ---
    member_queries = [
        row["query"] for row in dataset[: max(1, len(dataset) // 2)]
    ]
    non_member_queries = [
        f"Out-of-distribution probe {idx}: unrelated private record"
        for idx in range(max(1, len(member_queries)))
    ]
    start = time.perf_counter()
    mia = MembershipInferenceAttack(
        threshold=membership_threshold, top_k=1
    ).execute(
        retriever=retriever,
        knowledge_store=store,
        member_queries=member_queries,
        non_member_queries=non_member_queries,
    )
    mia_runtime = time.perf_counter() - start
    records.append(
        _make_matrix_record(
            attack="Membership Inference",
            system="FedRAG",
            dataset=ds_name,
            model=model,
            baseline=baseline_metrics,
            attacked=baseline_metrics,
            membership_acc=mia.attack_accuracy,
            runtime=mia_runtime,
            memory_mb=current_memory_mb(),
            notes=f"threshold={membership_threshold}, top_k=1",
        )
    )

    # --- Knowledge Extraction ---
    topics = sorted({row["topic"] for row in dataset})
    extraction_attack = KnowledgeExtractionAttack(top_k=top_k)
    queries = [row["query"] for row in dataset] + extraction_attack.generate_queries(
        topics
    )
    start = time.perf_counter()
    extraction = extraction_attack.execute(
        retriever=retriever,
        knowledge_store=store,
        queries=queries,
        total_nodes=store.count,
    )
    extraction_runtime = time.perf_counter() - start
    records.append(
        _make_matrix_record(
            attack="Knowledge Extraction",
            system="FedRAG",
            dataset=ds_name,
            model=model,
            baseline=baseline_metrics,
            attacked=baseline_metrics,
            kb_recovery_pct=extraction.recovery_ratio * 100.0,
            runtime=extraction_runtime,
            memory_mb=current_memory_mb(),
            notes=f"queries={extraction.total_queries}, top_k={top_k}",
        )
    )

    records.extend(_load_drag_reference_rows(config_root, ds_name, model))
    write_json_results(
        records, output_dir / "benchmark_attack_results.json"
    )
    write_evaluation_matrix(
        records,
        markdown_path=output_dir / "EVALUATION_MATRIX.md",
        csv_path=output_dir / "evaluation_matrix.csv",
    )
    print(f"Wrote {output_dir / 'EVALUATION_MATRIX.md'}")
    print(f"Wrote {output_dir / 'benchmark_attack_results.json'}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _load_drag_config(
    drag_root: Path, llm_name: str, dataset_name: str
) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required.") from exc
    config: dict[str, Any] = {}
    for rel_path in (
        "config/rag.yaml",
        f"config/llm/{llm_name}.yaml",
        f"config/data/{dataset_name}.yaml",
    ):
        with (drag_root / rel_path).open(
            encoding="utf-8"
        ) as handle:
            config.update(yaml.safe_load(handle))
    return config


def _load_drag_dataset(
    drag_root: Path,
    config: dict[str, Any],
    num_samples: int | None,
    seed: int,
    source: str,
) -> list[dict[str, Any]]:
    if source == "drag-log":
        return _load_drag_log_dataset(drag_root)
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("`datasets` is required.") from exc

    data_cfg = config["data"]
    dataset = load_dataset(**data_cfg["load"])
    sample_count = (
        num_samples or data_cfg.get("num_samples") or len(dataset)
    )
    if config["rag"].get("test_mode", False):
        sample_count = min(sample_count, 20)
        dataset = dataset.select(range(sample_count))
    else:
        dataset = dataset.shuffle(seed=seed).select(
            range(min(sample_count, len(dataset)))
        )

    rows = []
    for item in dataset:
        topic = _get_nested_value(item, data_cfg["topic_path"])
        question = _get_nested_value(item, data_cfg["question_path"])
        answer = _get_nested_value(item, data_cfg["answer_path"])
        if data_cfg["task_type"] == "mcqa":
            choices = _get_nested_value(item, data_cfg["choices_path"])
            question = (
                f"{question} Select the best answer from the following "
                f"candidates, replying with 1, 2, 3, or 4: {choices}"
            )
        rows.append(
            {
                "query": str(question),
                "response": str(answer),
                "topic": str(topic),
            }
        )
    return rows


def _load_drag_log_dataset(drag_root: Path) -> list[dict[str, Any]]:
    path = drag_root / "logs/version_1/test_cases.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            knowledge = json.loads(row["relevant_knowledge"])
            rows.append(
                {
                    "query": knowledge["question"],
                    "response": knowledge["answer"],
                    "topic": knowledge["topic"],
                }
            )
    return rows


def _build_knowledge_store(
    dataset: list[dict[str, Any]],
    retriever: HFSentenceTransformerRetriever,
) -> InMemoryKnowledgeStore:
    nodes = []
    for row in dataset:
        embedding = (
            retriever.encode_context(row["query"]).squeeze(0).tolist()
        )
        nodes.append(
            KnowledgeNode(
                node_type=NodeType.TEXT,
                text_content=row["query"],
                embedding=[float(v) for v in embedding],
                metadata={
                    "topic": row["topic"],
                    "answer": row["response"],
                },
            )
        )
    return InMemoryKnowledgeStore.from_nodes(nodes)


def _evaluate_rag_answers(
    *,
    dataset: list[dict[str, Any]],
    retriever: HFSentenceTransformerRetriever,
    knowledge_store: InMemoryKnowledgeStore,
    llm_config: dict[str, Any],
    use_ollama: bool,
    fallback_lookup: dict[str, str] | None = None,
    max_examples: int | None = None,
) -> dict[str, float]:
    subset = dataset[:max_examples] if max_examples else dataset
    predictions: dict[str, str] = {}
    ollama_client = _make_ollama_client(llm_config) if use_ollama else None

    for row in subset:
        retrieved = knowledge_store.retrieve(
            query_emb=retriever.encode_query(row["query"])
            .squeeze(0)
            .tolist(),
            top_k=1,
        )
        if not retrieved:
            predictions[row["query"]] = ""
            continue
        _, node = retrieved[0]
        retrieved_answer = str(node.metadata.get("answer", ""))
        if ollama_client is None:
            predictions[row["query"]] = (
                fallback_lookup.get(row["query"], retrieved_answer)
                if fallback_lookup
                else retrieved_answer
            )
            continue
        prompt = (
            "Answer the multiple-choice question using only the retrieved "
            "context. Reply with the answer option only.\n\n"
            f"Question: {row['query']}\n"
            f"Retrieved context: {node.text_content}\n"
            f"Retrieved answer: {retrieved_answer}\n"
            "Answer:"
        )
        response = ollama_client.generate(
            model=llm_config["name"],
            prompt=prompt,
            options={
                "seed": 0,
                "num_ctx": int(llm_config.get("num_ctx", 4096)),
                "num_predict": 16,
            },
        )
        predictions[row["query"]] = str(
            response.get("response", "")
        ).strip()

    return evaluate_lookup(subset, predictions)


def _make_ollama_client(llm_config: dict[str, Any]) -> Any:
    try:
        from ollama import Client
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install the `ollama` package to use --use-ollama-generation."
        ) from exc
    return Client(host=llm_config.get("base_url", "http://localhost:11434"))


def _get_nested_value(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for key in path.split("."):
        value = value[key]
    return value


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def current_memory_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _make_matrix_record(
    *,
    attack: str,
    system: str,
    dataset: str,
    model: str,
    baseline: dict[str, float],
    attacked: dict[str, float],
    runtime: float,
    memory_mb: float,
    membership_acc: float | None = None,
    kb_recovery_pct: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "Attack": attack,
        "System": system,
        "Dataset": dataset,
        "Model": model,
        "Baseline BLEU": baseline.get("bleu"),
        "Post-Attack BLEU": attacked.get("bleu"),
        "Delta BLEU": attacked.get("bleu", 0.0)
        - baseline.get("bleu", 0.0),
        "Baseline EM": baseline.get("em"),
        "Post-Attack EM": attacked.get("em"),
        "Delta EM": attacked.get("em", 0.0)
        - baseline.get("em", 0.0),
        "Baseline F1": baseline.get("f1"),
        "Post-Attack F1": attacked.get("f1"),
        "Delta F1": attacked.get("f1", 0.0)
        - baseline.get("f1", 0.0),
        "Membership Acc": membership_acc,
        "KB Recovery %": kb_recovery_pct,
        "Runtime (s)": runtime,
        "Memory Overhead MB": memory_mb,
        "Notes": notes,
    }


def _load_drag_reference_rows(
    drag_root: Path, dataset_name: str, model_name: str
) -> list[dict[str, Any]]:
    rows = []
    extraction_path = drag_root / "logs/version_0/extraction_metrics.csv"
    if extraction_path.exists():
        with extraction_path.open(
            newline="", encoding="utf-8"
        ) as handle:
            row = next(csv.DictReader(handle), None)
        if row:
            rows.append(
                _make_matrix_record(
                    attack="Knowledge Extraction",
                    system="DRAG",
                    dataset=dataset_name,
                    model=model_name,
                    baseline={},
                    attacked={},
                    kb_recovery_pct=float(row["crr"]) * 100.0,
                    runtime=0.0,
                    memory_mb=0.0,
                    notes="Imported from DRAG logs",
                )
            )
    return rows
