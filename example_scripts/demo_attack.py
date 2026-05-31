from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from fed_rag import RAGSystem
from fed_rag.attacks import DataPoisoningAttack
from fed_rag.base.generator import BaseGenerator
from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.base.retriever import BaseRetriever
from fed_rag.base.tokenizer import BaseTokenizer
from fed_rag.data_structures import KnowledgeNode, RAGConfig
from fed_rag.data_structures.knowledge_node import NodeType
from fed_rag.evals.benchmarker import Benchmarker
from fed_rag.evals.benchmarks import HuggingFaceMMLU
from fed_rag.evals.metrics import ExactMatchEvaluationMetric


class DummyTokenizer(BaseTokenizer):
    def encode(self, input: str, **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": [ord(ch) for ch in input], "attention_mask": None}

    def decode(self, input_ids: list[int], **kwargs: Any) -> str:
        return "".join(chr(i) for i in input_ids)

    @property
    def unwrapped(self) -> Any:
        return None


class DummyGenerator(BaseGenerator):
    def generate(self, query: Any, context: Any, **kwargs: dict) -> str | list[str]:
        if isinstance(query, list):
            return ["A" for _ in query]
        return "A"

    def complete(self, prompt: Any, **kwargs: dict) -> str | list[str]:
        if isinstance(prompt, list):
            return ["A" for _ in prompt]
        return "A"

    @property
    def model(self) -> torch.nn.Module:
        return torch.nn.Identity()

    @property
    def tokenizer(self) -> BaseTokenizer:
        return DummyTokenizer()

    def compute_target_sequence_proba(self, prompt: Any, target: str) -> torch.Tensor:
        return torch.tensor(0.0)

    @property
    def prompt_template(self) -> str:
        return "{query}\n{context}"

    @prompt_template.setter
    def prompt_template(self, value: str) -> None:
        pass


class DummyRetriever(BaseRetriever):
    def encode_query(self, query: Any, **kwargs: Any) -> torch.Tensor:
        if isinstance(query, list):
            return torch.zeros((len(query), 1))
        return torch.zeros(1)

    def encode_context(self, context: Any, **kwargs: Any) -> torch.Tensor:
        if isinstance(context, list):
            return torch.zeros((len(context), 1))
        return torch.zeros(1)

    @property
    def encoder(self) -> torch.nn.Module | None:
        return None

    @property
    def query_encoder(self) -> torch.nn.Module | None:
        return None

    @property
    def context_encoder(self) -> torch.nn.Module | None:
        return None


class DummyKnowledgeStore(BaseKnowledgeStore):
    nodes: list[KnowledgeNode] = []

    def load_node(self, node: KnowledgeNode) -> None:
        self.nodes.append(node)

    def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        self.nodes.extend(nodes)

    def retrieve(self, query_emb: list[float], top_k: int) -> list[tuple[float, KnowledgeNode]]:
        if not self.nodes:
            return []
        return [(1.0, node) for node in self.nodes[:top_k]]

    def batch_retrieve(self, query_embs: list[list[float]], top_k: int) -> list[list[tuple[float, KnowledgeNode]]]:
        return [self.retrieve(query_emb, top_k) for query_emb in query_embs]

    def delete_node(self, node_id: str) -> bool:
        for i, node in enumerate(self.nodes):
            if node.node_id == node_id:
                del self.nodes[i]
                return True
        return False

    def clear(self) -> None:
        self.nodes = []

    @property
    def count(self) -> int:
        return len(self.nodes)

    def persist(self) -> None:
        return None

    def load(self) -> None:
        return None


# Setup mock RAG system
ks = DummyKnowledgeStore()
ks.load_node(
    KnowledgeNode(
        node_type=NodeType.TEXT,
        text_content="",
        embedding=[0.0],
    )
)
rag_system = RAGSystem(
    generator=DummyGenerator(),
    retriever=DummyRetriever(),
    knowledge_store=ks,
    rag_config=RAGConfig(top_k=1),
)

# MMLU benchmark (500 examples, streaming)
benchmarker = Benchmarker(rag_system=rag_system)
mmlu = HuggingFaceMMLU(streaming=True)
metric = ExactMatchEvaluationMetric()

logs_dir = Path("logs")
logs_dir.mkdir(parents=True, exist_ok=True)

benchmark_result = benchmarker.run(
    benchmark=mmlu,
    metric=metric,
    agg="avg",
    is_streaming=True,
    num_examples=500,
    save_evaluations=True,
    output_dir=logs_dir / "benchmark_results",
)

# Attack demo data
random.seed(7)
topics = ["math", "history", "biology", "physics", "literature"]

def make_example(i: int) -> dict[str, Any]:
    topic = topics[i % len(topics)]
    question = f"Q{i}: What is a fact about {topic}?"
    answer = f"Answer about {topic} #{i}"
    return {"query": question, "response": answer, "topic": topic}

clean_data = [make_example(i) for i in range(200)]

attack = DataPoisoningAttack(
    poisoning_ratio=0.1,
    poison_type="wrong_answer",
    mode="replace",
    seed=42,
)
poisoned_result = attack.execute(clean_data)
poisoned_data = poisoned_result.poisoned_examples

# Build lookup-based models
clean_lookup = {ex["query"]: ex["response"] for ex in clean_data}
poisoned_lookup = {ex["query"]: ex["response"] for ex in poisoned_data}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def exact_match(pred: str, actual: str) -> float:
    return float(pred.strip().lower() == actual.strip().lower())


def token_f1(pred: str, actual: str) -> float:
    pred_tokens = tokenize(pred)
    actual_tokens = tokenize(actual)
    if not pred_tokens and not actual_tokens:
        return 1.0
    if not pred_tokens or not actual_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(actual_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(actual_tokens)
    return 2 * precision * recall / (precision + recall)


def jaccard_sim(pred: str, actual: str) -> float:
    pred_set = set(tokenize(pred))
    actual_set = set(tokenize(actual))
    if not pred_set and not actual_set:
        return 1.0
    if not pred_set or not actual_set:
        return 0.0
    return len(pred_set & actual_set) / len(pred_set | actual_set)


def bleu_score(pred: str, actual: str) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except Exception:
        return 0.0
    pred_tokens = tokenize(pred)
    actual_tokens = tokenize(actual)
    if not pred_tokens or not actual_tokens:
        return 0.0
    smooth = SmoothingFunction().method1
    return float(sentence_bleu([actual_tokens], pred_tokens, smoothing_function=smooth))


def evaluate(dataset: list[dict[str, Any]], lookup: dict[str, str]) -> dict[str, float]:
    scores = {"em": [], "f1": [], "bleu": [], "jaccard": []}
    for ex in dataset:
        pred = lookup.get(ex["query"], "")
        actual = ex["response"]
        scores["em"].append(exact_match(pred, actual))
        scores["f1"].append(token_f1(pred, actual))
        scores["bleu"].append(bleu_score(pred, actual))
        scores["jaccard"].append(jaccard_sim(pred, actual))

    return {k: sum(v) / max(1, len(v)) for k, v in scores.items()}


clean_metrics = evaluate(clean_data, clean_lookup)
poisoned_metrics = evaluate(clean_data, poisoned_lookup)

attack_summary = {
    "poisoning_ratio": poisoned_result.poisoning_ratio,
    "num_original": poisoned_result.num_original,
    "num_poisoned": poisoned_result.num_poisoned,
    "mode": poisoned_result.mode,
    "poison_type": poisoned_result.poison_type,
}

output = {
    "mmlu": {
        "score": benchmark_result.score,
        "num_examples_used": benchmark_result.num_examples_used,
        "num_total_examples": benchmark_result.num_total_examples,
        "metric": benchmark_result.metric_name,
        "evaluations_file": benchmark_result.evaluations_file,
    },
    "attack_demo": {
        "summary": attack_summary,
        "metrics_clean": clean_metrics,
        "metrics_poisoned": poisoned_metrics,
    },
}

log_path = logs_dir / "benchmark_attack_demo.json"
with open(log_path, "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
