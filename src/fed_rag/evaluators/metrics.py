"""Shared evaluation metrics for FedRAG security benchmarking.

Mirrors the DRAG ``QAEvaluator`` metric suite so that both systems can be
compared on the same matrix.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

import numpy as np
import torch

from fed_rag.base.retriever import BaseRetriever
from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore

# ------------------------------------------------------------------
# Optional DRAG-level metrics (graceful fallback if libs missing)
# ------------------------------------------------------------------

_ROUGE_AVAILABLE = False
try:
    from rouge_score import rouge_scorer

    _ROUGE_AVAILABLE = True
except Exception:
    pass

_LEV_AVAILABLE = False
try:
    import Levenshtein

    _LEV_AVAILABLE = True
except Exception:
    pass

_NLTK_AVAILABLE = False
try:
    from nltk.tokenize import word_tokenize

    _NLTK_AVAILABLE = True
except Exception:
    pass

_SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    pass


# ------------------------------------------------------------------
# Tokenisation (NLTK when possible, else simple split)
# ------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    if _NLTK_AVAILABLE:
        try:
            return word_tokenize(text.lower())
        except Exception:
            pass
    return text.lower().split()


# ------------------------------------------------------------------
# Basic metrics
# ------------------------------------------------------------------


def token_precision(pred: str, actual: str) -> float:
    pred_tokens = tokenize(pred)
    actual_tokens = tokenize(actual)
    if not pred_tokens or not actual_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(actual_tokens)
    overlap = sum(common.values())
    return overlap / len(pred_tokens)


def token_recall(pred: str, actual: str) -> float:
    pred_tokens = tokenize(pred)
    actual_tokens = tokenize(actual)
    if not pred_tokens or not actual_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(actual_tokens)
    overlap = sum(common.values())
    return overlap / len(actual_tokens)


def token_f1(pred: str, actual: str) -> float:
    p = token_precision(pred, actual)
    r = token_recall(pred, actual)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def simple_bleu(pred: str, actual: str) -> float:
    pred_tokens = tokenize(pred)
    actual_tokens = tokenize(actual)
    if not pred_tokens or not actual_tokens:
        return 0.0
    overlap = sum(
        (Counter(pred_tokens) & Counter(actual_tokens)).values()
    )
    return overlap / len(pred_tokens)


def jaccard(pred: str, actual: str) -> float:
    pred_set = set(tokenize(pred))
    actual_set = set(tokenize(actual))
    return (
        len(pred_set & actual_set) / len(pred_set | actual_set)
        if pred_set or actual_set
        else 1.0
    )


def exact_match(pred: str, actual: str) -> float:
    return float(
        _normalize_text(pred) == _normalize_text(actual)
    )


def _normalize_text(text: str) -> str:
    """DRAG-style normalisation: lower-case, remove articles + punct."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"[\[\]]", "", text)
    text = " ".join(text.split())
    return text


# ------------------------------------------------------------------
# ROUGE metrics
# ------------------------------------------------------------------


def rouge_scores(pred: str, actual: str) -> dict[str, float]:
    if not _ROUGE_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(actual, pred)
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
    }


# ------------------------------------------------------------------
# Semantic similarity
# ------------------------------------------------------------------

_semantic_model: Any = None


def semantic_similarity(pred: str, actual: str) -> float:
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        return jaccard(pred, actual)
    global _semantic_model
    if _semantic_model is None:
        try:
            _semantic_model = SentenceTransformer(
                "paraphrase-MiniLM-L6-v2"
            )
        except Exception:
            return jaccard(pred, actual)
    if not pred or not actual:
        return 0.0
    try:
        e1 = _semantic_model.encode(pred)
        e2 = _semantic_model.encode(actual)
        sim = float(
            np.dot(e1, e2)
            / (np.linalg.norm(e1) * np.linalg.norm(e2))
        )
        return sim
    except Exception:
        return 0.0


# ------------------------------------------------------------------
# Edit distance
# ------------------------------------------------------------------


def edit_distance_metrics(pred: str, actual: str) -> dict[str, float]:
    if _LEV_AVAILABLE:
        distance = Levenshtein.distance(pred, actual)
    else:
        distance = _levenshtein_fallback(pred, actual)
    max_len = max(len(pred), len(actual))
    normalized = 1.0 - (distance / max_len) if max_len > 0 else 0.0
    return {
        "edit_distance": float(distance),
        "normalized_edit_distance": normalized,
    }


def _levenshtein_fallback(s1: str, s2: str) -> int:
    """Simple O(n*m) fallback when the C extension is absent."""
    if len(s1) < len(s2):
        return _levenshtein_fallback(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dels = curr[j] + 1
            subs = prev[j] + (0 if c1 == c2 else 1)
            curr.append(min(ins, dels, subs))
        prev = curr
    return prev[-1]


# ------------------------------------------------------------------
# N-gram overlap
# ------------------------------------------------------------------


def ngram_overlap(pred: str, actual: str, n: int = 2) -> float:
    pred_tokens = tokenize(pred)
    actual_tokens = tokenize(actual)
    if len(pred_tokens) < n or len(actual_tokens) < n:
        return 0.0

    def _get_ngrams(tokens: list[str], n: int) -> set[str]:
        return set(
            " ".join(tokens[i : i + n])
            for i in range(len(tokens) - n + 1)
        )

    pred_ngrams = _get_ngrams(pred_tokens, n)
    actual_ngrams = _get_ngrams(actual_tokens, n)
    overlap = len(pred_ngrams & actual_ngrams)
    total = len(pred_ngrams | actual_ngrams)
    return overlap / total if total > 0 else 0.0


# ------------------------------------------------------------------
# DRAG-compatible full metric suite
# ------------------------------------------------------------------


def compute_all_metrics(pred: str, actual: str) -> dict[str, float]:
    """Return a single dict with *all* metrics that DRAG computes."""
    rouge = rouge_scores(pred, actual)
    edit = edit_distance_metrics(pred, actual)
    return {
        "exact_match": exact_match(pred, actual),
        "precision": token_precision(pred, actual),
        "recall": token_recall(pred, actual),
        "f1": token_f1(pred, actual),
        "bleu": simple_bleu(pred, actual),
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "semantic_similarity": semantic_similarity(pred, actual),
        "jaccard": jaccard(pred, actual),
        "edit_distance": edit["edit_distance"],
        "normalized_edit_distance": edit["normalized_edit_distance"],
        "bigram_overlap": ngram_overlap(pred, actual, n=2),
        "trigram_overlap": ngram_overlap(pred, actual, n=3),
    }


# ------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ------------------------------------------------------------------


def evaluate_lookup(
    dataset: list[dict[str, Any]], lookup: dict[str, str]
) -> dict[str, float]:
    scores: dict[str, list[float]] = {
        "em": [],
        "f1": [],
        "bleu": [],
        "jaccard": [],
    }
    for row in dataset:
        pred = lookup.get(row["query"], "")
        actual = row["response"]
        scores["em"].append(exact_match(pred, actual))
        scores["f1"].append(token_f1(pred, actual))
        scores["bleu"].append(simple_bleu(pred, actual))
        scores["jaccard"].append(jaccard(pred, actual))
    return {
        key: sum(values) / max(1, len(values))
        for key, values in scores.items()
    }


def evaluate_rag_answers(
    dataset: list[dict[str, Any]],
    knowledge_store: InMemoryKnowledgeStore,
    retriever: BaseRetriever,
    dropped_clients: set[int] | None = None,
    byzantine_clients: set[int] | None = None,
) -> dict[str, float]:
    scores: dict[str, list[float]] = {
        "em": [],
        "f1": [],
        "bleu": [],
        "jaccard": [],
    }
    for row in dataset:
        client_id = row.get("client_id")
        if dropped_clients and client_id in dropped_clients:
            pred = "QUERY_FAILED_NODE_UNAVAILABLE"
        else:
            retrieved = knowledge_store.retrieve(
                query_emb=retriever.encode_query(row["query"]).tolist(),
                top_k=1,
            )
            pred = (
                str(retrieved[0][1].metadata.get("answer", ""))
                if retrieved
                else ""
            )
        if byzantine_clients and client_id in byzantine_clients:
            pred = "INCORRECT_BYZANTINE_RESPONSE"
        actual = row["response"]
        scores["em"].append(exact_match(pred, actual))
        scores["f1"].append(token_f1(pred, actual))
        scores["bleu"].append(simple_bleu(pred, actual))
        scores["jaccard"].append(jaccard(pred, actual))
    return {
        key: sum(values) / max(1, len(values))
        for key, values in scores.items()
    }


def evaluate_system_rag_answers(
    dataset: list[dict[str, Any]],
    knowledge_store: InMemoryKnowledgeStore,
    retriever: BaseRetriever,
    dropped_clients: set[int] | None = None,
    byzantine_clients: set[int] | None = None,
) -> dict[str, float]:
    """System-level evaluation matching DRAG's ``QAEvaluator`` output.

    Returns answer-quality metrics that are directly comparable across
    DRAG and FedRAG (EM, precision, recall, f1, bleu, rouge,
    semantic_similarity, edit_distance, n-gram overlap), plus
    query_failure_rate which measures system availability under attack.
    """
    metrics: dict[str, list[float]] = {
        "exact_match": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "bleu": [],
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "semantic_similarity": [],
        "jaccard": [],
        "edit_distance": [],
        "normalized_edit_distance": [],
        "bigram_overlap": [],
        "trigram_overlap": [],
    }
    failed_queries = 0
    total = len(dataset)

    for row in dataset:
        client_id = row.get("client_id")
        if dropped_clients and client_id in dropped_clients:
            pred = "QUERY_FAILED_NODE_UNAVAILABLE"
            failed_queries += 1
        else:
            retrieved = knowledge_store.retrieve(
                query_emb=retriever.encode_query(row["query"]).tolist(),
                top_k=1,
            )
            pred = (
                str(retrieved[0][1].metadata.get("answer", ""))
                if retrieved
                else ""
            )

        if byzantine_clients and client_id in byzantine_clients:
            pred = "INCORRECT_BYZANTINE_RESPONSE"

        actual = row["response"]
        all_metrics = compute_all_metrics(pred, actual)
        for k in metrics:
            metrics[k].append(all_metrics[k])

    results: dict[str, float] = {
        k: sum(v) / max(1, len(v)) for k, v in metrics.items()
    }
    results["query_failure_rate"] = (
        failed_queries / total if total else 0.0
    )
    results["successful_queries"] = total - failed_queries
    results["failed_queries"] = failed_queries
    return results


# ------------------------------------------------------------------
# Hashing retriever (for quick offline experiments)
# ------------------------------------------------------------------


class HashingRetriever(BaseRetriever):
    """Small deterministic retriever for offline experiments."""

    dimensions: int = 128

    def encode_query(self, query: Any, **kwargs: Any) -> torch.Tensor:
        return torch.tensor(
            self._encode(str(query)), dtype=torch.float32
        )

    def encode_context(self, context: Any, **kwargs: Any) -> torch.Tensor:
        return self.encode_query(context)

    @property
    def encoder(self) -> torch.nn.Module | None:
        return None

    @property
    def query_encoder(self) -> torch.nn.Module | None:
        return None

    @property
    def context_encoder(self) -> torch.nn.Module | None:
        return None

    def _encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else vector
