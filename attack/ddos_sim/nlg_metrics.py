"""
attack/ddos_sim/nlg_metrics.py

Generation-quality scoring for a real LLM answer vs. a ground-truth
answer: exact_match, token-level precision/recall/F1, BLEU, ROUGE-1/2/L,
semantic similarity, length ratio/difference, edit distance, and
bigram/trigram overlap.

No implementation of these exists anywhere else in this repo (confirmed
by search: no rouge_score/nltk/sacrebleu dependency is installed, and
every existing attack script -- attack/datapoisoning/run_attack.py,
attack/ssm_score/run_attack.py, drag_llm_service's own /query_analyze --
only does case-insensitive substring/exact-string matching for
"correctness", not partial-credit NLG scoring). This module is
pure-Python (no new dependencies) except for semantic_similarity, which
reuses the sentence-transformers model attack/Mia_attack/mia_attack.py
already loads (`all-MiniLM-L6-v2`) rather than adding a new one.

All functions operate on a single prediction against one or more
acceptable gold answers (`score_answer` takes the max/best-matching gold
for each metric, mirroring how SQuAD-style multi-reference scoring
picks the best-matching reference per example).
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional, Sequence

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    text = (text or "").lower().translate(_PUNCT_TABLE)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def _tokens(text: str) -> List[str]:
    return normalize_text(text).split()


# ── exact match ──────────────────────────────────────────────────────────
def exact_match(prediction: str, golds: Sequence[str]) -> float:
    pred_norm = normalize_text(prediction)
    return 1.0 if any(pred_norm == normalize_text(g) for g in golds) else 0.0


# ── token-level precision/recall/F1 (SQuAD-style, multiset overlap) ───────
def _token_prf1(pred_tokens: List[str], gold_tokens: List[str]) -> Dict[str, float]:
    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def best_token_prf1(prediction: str, golds: Sequence[str]) -> Dict[str, float]:
    pred_tokens = _tokens(prediction)
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for g in golds:
        scores = _token_prf1(pred_tokens, _tokens(g))
        if scores["f1"] > best["f1"]:
            best = scores
    return best


# ── BLEU (single-reference, up to 4-gram, with brevity penalty) ───────────
def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)) if len(tokens) >= n else Counter()


def bleu(prediction: str, gold: str, max_n: int = 4) -> float:
    pred_tokens, gold_tokens = _tokens(prediction), _tokens(gold)
    if not pred_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _ngrams(pred_tokens, n)
        gold_ngrams = _ngrams(gold_tokens, n)
        overlap = sum((pred_ngrams & gold_ngrams).values())
        total = max(1, sum(pred_ngrams.values()))
        # +1 additive smoothing so a single missing n-gram order doesn't zero the geometric mean
        precisions.append((overlap + 1) / (total + 1))
    import math
    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    geo_mean = math.exp(log_avg)
    bp = 1.0 if len(pred_tokens) >= len(gold_tokens) else math.exp(1 - len(gold_tokens) / max(1, len(pred_tokens)))
    return bp * geo_mean


# ── ROUGE-N (recall-oriented) and ROUGE-L (LCS-based F1) ──────────────────
def rouge_n(prediction: str, gold: str, n: int) -> float:
    gold_ngrams = _ngrams(_tokens(gold), n)
    if not gold_ngrams:
        return 0.0
    pred_ngrams = _ngrams(_tokens(prediction), n)
    overlap = sum((pred_ngrams & gold_ngrams).values())
    return overlap / sum(gold_ngrams.values())


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            curr[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def rouge_l(prediction: str, gold: str) -> float:
    pred_tokens, gold_tokens = _tokens(prediction), _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ── edit distance (word-level Levenshtein) ─────────────────────────────────
def word_edit_distance(prediction: str, gold: str) -> int:
    a, b = _tokens(prediction), _tokens(gold)
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, tb in enumerate(b, start=1):
            cost = 0 if ta == tb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len(b)]


def normalized_edit_distance_score(prediction: str, gold: str) -> float:
    """1.0 = identical, 0.0 = maximally different (word-level, length-normalized).
    Standalone convenience wrapper around word_edit_distance() for callers
    that just want the single-pair similarity score without the full
    score_answer() aggregate (e.g. attack/kb_extraction's CRR check)."""
    pred_tokens, gold_tokens = _tokens(prediction), _tokens(gold)
    max_len = max(len(pred_tokens), len(gold_tokens), 1)
    return 1.0 - (word_edit_distance(prediction, gold) / max_len)


# ── n-gram overlap (recall-style, same denominator convention as rouge_n) ──
def ngram_overlap(prediction: str, gold: str, n: int) -> float:
    gold_ngrams = _ngrams(_tokens(gold), n)
    if not gold_ngrams:
        return 0.0
    pred_ngrams = _ngrams(_tokens(prediction), n)
    overlap = sum((pred_ngrams & gold_ngrams).values())
    return overlap / sum(gold_ngrams.values())


# ── semantic similarity (sentence-transformers cosine similarity) ─────────
_ENCODER = None
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # same model attack/Mia_attack/mia_attack.py already loads


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer(EMBEDDING_MODEL)
    return _ENCODER


def semantic_similarity(prediction: str, gold: str) -> float:
    if not prediction.strip() or not gold.strip():
        return 0.0
    import numpy as np
    encoder = _get_encoder()
    emb = encoder.encode([prediction, gold], normalize_embeddings=True)
    cos = float(np.dot(emb[0], emb[1]))
    return max(0.0, min(1.0, cos))


# ── aggregate: score one prediction against one or more gold answers ──────
def score_answer(prediction: str, golds: Sequence[str], compute_semantic: bool = True) -> Dict[str, float]:
    """Best-matching-gold scoring: for each metric, pick the gold answer
    that yields the highest score for that metric (SQuAD-style
    multi-reference convention)."""
    golds = [g for g in golds if g] or [""]
    prf1 = best_token_prf1(prediction, golds)

    best_gold_for_lexical = max(golds, key=lambda g: rouge_l(prediction, g)) if prediction.strip() else golds[0]
    pred_tokens, gold_tokens = _tokens(prediction), _tokens(best_gold_for_lexical)
    lp, lg = len(pred_tokens), len(gold_tokens)
    max_len = max(lp, lg, 1)
    edit_dist = word_edit_distance(prediction, best_gold_for_lexical)

    result = {
        "exact_match": exact_match(prediction, golds),
        "precision": prf1["precision"],
        "recall": prf1["recall"],
        "f1": prf1["f1"],
        "bleu": max(bleu(prediction, g) for g in golds),
        "rouge1": max(rouge_n(prediction, g, 1) for g in golds),
        "rouge2": max(rouge_n(prediction, g, 2) for g in golds),
        "rougeL": max(rouge_l(prediction, g) for g in golds),
        "length_ratio": min(lp, lg) / max_len if max_len else 1.0,
        "length_difference": abs(lp - lg),
        "edit_distance": edit_dist,
        "normalized_edit_distance": 1.0 - (edit_dist / max_len),
        "bigram_overlap": max(ngram_overlap(prediction, g, 2) for g in golds),
        "trigram_overlap": max(ngram_overlap(prediction, g, 3) for g in golds),
    }
    if compute_semantic:
        result["semantic_similarity"] = max(semantic_similarity(prediction, g) for g in golds)
    return result


def average_metrics(per_question: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_question:
        return {}
    keys = per_question[0].keys()
    return {k: sum(m.get(k, 0.0) for m in per_question) / len(per_question) for k in keys}
