"""
Defense layers against data-poisoning attacks on candidate retrieval.

Two complementary, config-toggleable layers applied before final top_k selection:

  1. dedup_candidates   - collapses near-identical candidates (cosine similarity above
                          a threshold) within the same source down to one representative.
                          Blunts amplification/question_variants flooding, which always
                          duplicates within a single poisoned source.
  2. consensus_scores   - scores each candidate by how well its embedding aligns with
                          the centroid of candidates from OTHER sources. A source
                          poisoned with a fabricated answer is a semantic outlier
                          relative to independent retrieval from clean sources.

select_top_k_with_defense() ties these together with the existing Reranker fusion math
(dense/bm25/hybrid + orig_score_weight + reliability_weight) into a single selection path
shared by both server.py call sites.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def dedup_candidates(
    candidates: List[Dict[str, Any]],
    cand_embeddings: np.ndarray,
    similarity_threshold: float = 0.93,
    same_source_only: bool = True,
) -> Tuple[List[int], Dict[int, int]]:
    """Greedy clustering by cosine similarity (embeddings assumed L2-normalized).

    Processes candidates in descending `score` order so the best-scoring member of
    each near-duplicate cluster survives as representative.

    Returns
    -------
    kept_indices : indices into the ORIGINAL candidates/embeddings arrays to keep.
    dup_count    : {rep_idx: cluster_size}, keyed by original index.
    """
    n = len(candidates)
    order = sorted(range(n), key=lambda i: -float(candidates[i].get("score", 0.0)))
    assigned = [False] * n
    kept: List[int] = []
    dup_count: Dict[int, int] = {}
    for i in order:
        if assigned[i]:
            continue
        assigned[i] = True
        dup_count[i] = 1
        for j in order:
            if assigned[j] or j == i:
                continue
            if same_source_only and candidates[j]["meta"].get("source") != candidates[i]["meta"].get("source"):
                continue
            sim = float(np.dot(cand_embeddings[i], cand_embeddings[j]))
            if sim >= similarity_threshold:
                assigned[j] = True
                dup_count[i] += 1
        kept.append(i)
    return kept, dup_count


def consensus_scores(
    candidates: List[Dict[str, Any]],
    cand_embeddings: np.ndarray,
    source_key: str = "source",
    min_sources: int = 2,
) -> np.ndarray:
    """Per-candidate cosine similarity to the centroid of OTHER sources' candidates.

    Returns a flat 0.5 array (neutral, no discounting after blending) when fewer than
    min_sources distinct sources are present in the candidate pool.
    """
    n = len(candidates)
    sources = [c["meta"].get(source_key) for c in candidates]
    if len(set(sources)) < min_sources or cand_embeddings.size == 0:
        return np.full(n, 0.5, dtype=np.float32)

    raw = np.zeros(n, dtype=np.float32)
    for i, src in enumerate(sources):
        mask = [s != src for s in sources]
        if not any(mask):
            raw[i] = 0.5
            continue
        centroid = cand_embeddings[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        raw[i] = float(np.dot(cand_embeddings[i], centroid))
    return _minmax(raw)


def select_top_k_with_defense(
    reranker,
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
    reliability_scores: Optional[Dict[str, float]],
    reliability_weight: float,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Single shared selection path for both server.py call sites (query() and
    _query_analyze()). Reuses Reranker's existing hybrid_alpha/orig_score_weight fusion,
    then applies dedup + consensus on top.
    """
    texts = [str(c.get("text") or "") for c in candidates]
    orig_scores = np.asarray([float(c.get("score", 0.0)) for c in candidates], dtype=np.float32)

    q_emb, cand_embs = reranker.embed_query_and_candidates(query, texts)
    dense = (cand_embs @ q_emb) if q_emb.size and cand_embs.size else np.zeros(len(candidates), dtype=np.float32)
    bm25 = reranker._bm25_scores(query, texts) if reranker.method in {"bm25", "hybrid"} else None

    if reranker.method == "hybrid":
        fused = (1.0 - reranker.hybrid_alpha) * _minmax(dense) + reranker.hybrid_alpha * _minmax(bm25)
    elif reranker.method == "bm25":
        fused = _minmax(bm25) if bm25 is not None else np.zeros_like(orig_scores)
    else:
        fused = _minmax(dense)

    if reranker.orig_score_weight > 0.0:
        fused = (1.0 - reranker.orig_score_weight) * fused + reranker.orig_score_weight * _minmax(orig_scores)

    if reliability_scores:
        rel_raw = np.array(
            [reliability_scores.get(c["meta"]["source"], 0.0) for c in candidates],
            dtype=np.float32,
        )
        fused = (1.0 - reliability_weight) * fused + reliability_weight * _minmax(rel_raw)

    kept_idx = list(range(len(candidates)))
    dup_count: Dict[int, int] = {}
    if cfg.get("dedup_enabled", True):
        kept_idx, dup_count = dedup_candidates(
            candidates,
            cand_embs,
            similarity_threshold=cfg.get("dedup_similarity_threshold", 0.93),
            same_source_only=cfg.get("dedup_same_source_only", True),
        )

    d_candidates = [candidates[i] for i in kept_idx]
    d_embs = cand_embs[kept_idx]
    d_fused = fused[kept_idx]
    d_orig = orig_scores[kept_idx]

    consensus_w = float(cfg.get("consensus_weight", 0.0))
    if consensus_w > 0.0:
        cons = consensus_scores(
            d_candidates, d_embs, min_sources=cfg.get("consensus_min_sources", 2)
        )
        combined = (1.0 - consensus_w) * d_fused + consensus_w * cons
    else:
        cons = np.full(len(d_candidates), 0.5, dtype=np.float32)
        combined = d_fused

    k = max(1, min(top_k, len(d_candidates)))
    order = np.argsort(-combined)[:k]

    results: List[Dict[str, Any]] = []
    for pos in order:
        pos = int(pos)
        orig_i = kept_idx[pos]
        c = dict(d_candidates[pos])
        c["orig_score"] = float(d_orig[pos])
        c["rerank_score"] = float(combined[pos])
        c["consensus_score"] = float(cons[pos])
        c["dup_count"] = dup_count.get(orig_i, 1)
        results.append(c)
    return results
