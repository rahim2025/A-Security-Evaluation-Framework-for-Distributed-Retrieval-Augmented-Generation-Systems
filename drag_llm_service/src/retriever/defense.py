"""
Defense layers against data-poisoning attacks on candidate retrieval, plus
one defense against a different threat model: Membership Inference Attacks
(MIA) against retrieval-corpus membership (attack/Mia_attack/).

Three complementary, config-toggleable layers applied before final top_k selection:

  1. dedup_candidates       - collapses near-identical candidates (cosine similarity
                              above a threshold) within the same source down to one
                              representative. Blunts amplification/question_variants
                              flooding, which always duplicates within a single
                              poisoned source.
  2. consensus_scores       - scores each candidate by how well its embedding aligns
                              with the centroid of candidates from OTHER sources. A
                              source poisoned with a fabricated answer is a semantic
                              outlier relative to independent retrieval from clean
                              sources.
  3. add_laplace_noise_to_similarity - (NEW) perturbs each candidate's raw dense
                              cosine-similarity score with calibrated Laplace noise
                              before ranking. Targets a different attacker: an MIA
                              probe exploits the fact that a member document's real
                              backing passage lets it be retrieved (and hence
                              answered) with a distinctively higher similarity score
                              than a non-member's absence of any real passage --
                              see reports/MIA_Security_Analysis_Report.md and
                              attack/Mia_attack/mia_attack.py's cosine-similarity
                              signal. Adding noise to the similarity score itself
                              (not to reliability, not to the LLM's output text)
                              blunts exactly that channel without touching anything
                              else in the fusion/ranking pipeline. See that
                              function's docstring for the full design and
                              reports/updated_reports_safin/ for the writeup.

select_top_k_with_defense() ties these together with the existing Reranker fusion math
(dense/bm25/hybrid + orig_score_weight + reliability_weight) into a single selection path
shared by both server.py call sites. Layer 3 is applied to `dense` before any of the
existing fusion math (_minmax, hybrid blending, orig_score_weight, reliability_weight,
dedup, consensus) runs -- every one of those steps is unchanged code operating on a
perturbed input, not new fusion math.
"""

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# ── Similarity-only Laplace noise defense (MIA countermeasure) ──────────────

# Reranker._embed_texts() L2-normalizes embeddings by default
# (normalize_dense=True, the config default -- see reranker.py), so the raw
# dense score `cand_embs @ q_emb` this module perturbs is a plain cosine
# similarity, range [-1, 1] by construction. Global sensitivity of a single
# bounded scalar is its full range: 1 - (-1) = 2.0. This is the "actual
# similarity representation/range in this repository" the sensitivity
# parameter is derived from -- not an assumption about a different score
# representation (e.g. an already-minmax'd-to-[0,1] score, whose effective
# per-query range is data-dependent, not a fixed constant suitable for a
# global sensitivity figure). Overridable via config for a deployment that
# changes normalize_dense or uses an unnormalized/differently-scaled
# similarity.
DEFAULT_SIMILARITY_SENSITIVITY = 2.0
DEFAULT_SIMILARITY_EPSILON = 1.0
DEFAULT_SIMILARITY_NOISE_SEED = 42


def _derive_seed(base_seed: int, query: str) -> int:
    """
    Deterministically derive a per-query seed from the configured base seed
    (the repository's existing seed-config convention -- see
    drag_llm_service/configs/config.yaml's `model.seed`) and the query text.

    Two properties this is designed to give, together:
      (a) Reproducibility: the SAME query with the SAME base seed always
          derives the SAME sub-seed, so the SAME noise draw -- required so
          tests and offline multi-run comparisons are deterministic, and so
          `select_top_k_with_defense(..., cfg={"similarity_noise_seed": N})`
          called twice with identical inputs returns bit-identical output.
      (b) Not one frozen noise vector for the life of the process: different
          queries derive different sub-seeds, so noise is not a single fixed
          offset applied identically to every query the service ever
          receives.

    Uses hashlib (stdlib, already a transitive dependency of this project)
    rather than Python's built-in hash(), which is salted per-process
    (PYTHONHASHSEED) and would silently break property (a) across separate
    runs/processes -- not introducing a new random/hashing framework, just
    using the stdlib deterministically instead of a non-deterministic
    built-in.
    """
    digest = hashlib.sha256(f"{int(base_seed)}:{query}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def add_laplace_noise_to_similarity(
    similarity_scores: np.ndarray,
    query: str,
    enabled: bool = False,
    epsilon: float = DEFAULT_SIMILARITY_EPSILON,
    sensitivity: float = DEFAULT_SIMILARITY_SENSITIVITY,
    seed: int = DEFAULT_SIMILARITY_NOISE_SEED,
) -> np.ndarray:
    """
    Similarity-only Laplace noise defense against Membership Inference
    Attacks (attack/Mia_attack/). For each candidate's raw similarity score,
    adds i.i.d. Laplace(0, b) noise, b = sensitivity / epsilon:

        noisy_similarity = similarity + Laplace(0, sensitivity / epsilon)

    Parameters
    ----------
    similarity_scores : the RAW (pre-_minmax) dense cosine-similarity score
        per candidate -- see module docstring for why this specific stage
        of the pipeline, and DEFAULT_SIMILARITY_SENSITIVITY's comment for
        why 2.0 (the [-1,1] cosine range) is the right sensitivity constant
        for this representation.
    enabled : when False, returns `similarity_scores` completely unchanged
        (same object's values, no new array semantics beyond a passthrough)
        -- this is what makes the defense a true no-op when disabled,
        satisfying "retrieval behavior must remain exactly unchanged."
    epsilon : Laplace privacy parameter. Smaller epsilon -> larger scale ->
        more noise -> stronger defense, weaker task-quality signal. Must be
        > 0 (epsilon<=0 is not a valid privacy budget for this mechanism
        and raises ValueError rather than silently producing inf/nan
        noise).
    sensitivity : global sensitivity of the perturbed score. Defaults to
        2.0 (see DEFAULT_SIMILARITY_SENSITIVITY) but is exposed so a
        deployment with a different similarity representation/range can
        override it correctly rather than inheriting an assumption that
        doesn't hold for it.
    seed : base seed for reproducible noise -- see `_derive_seed()`.

    Only ever touches the similarity score passed in. Never reads or
    writes reliability scores, utility scores, blockchain/smart-contract
    values, SSM logic, or LLM output text -- none of those are visible to
    or reachable from this function by construction (it takes and returns
    only a plain numpy array of similarity scores).
    """
    if not enabled:
        return similarity_scores
    if epsilon <= 0:
        raise ValueError(f"similarity_noise epsilon must be > 0, got {epsilon}")
    if sensitivity <= 0:
        raise ValueError(f"similarity_noise sensitivity must be > 0, got {sensitivity}")

    scale = float(sensitivity) / float(epsilon)
    rng = np.random.default_rng(_derive_seed(seed, query))
    noise = rng.laplace(loc=0.0, scale=scale, size=np.asarray(similarity_scores).shape)
    return (np.asarray(similarity_scores, dtype=np.float64) + noise).astype(np.float32)


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

    # Similarity-only Laplace noise defense (MIA countermeasure) -- applied to
    # the RAW dense cosine-similarity score, before _minmax/fusion, so every
    # downstream step (hybrid blending, orig_score_weight, reliability
    # blending, dedup, consensus, top-k) is unchanged code operating on a
    # perturbed input, not new fusion math. `enabled` defaults to False, so
    # when the config key is absent or false this call is a pure passthrough
    # and `dense` is bit-identical to before this defense existed -- see
    # add_laplace_noise_to_similarity()'s docstring. BM25 (`bm25`, below) is
    # deliberately untouched: this is a similarity-only defense, not a
    # blanket noise layer over every retrieval signal.
    dense = add_laplace_noise_to_similarity(
        dense,
        query=query,
        enabled=bool(cfg.get("similarity_noise_enabled", False)),
        epsilon=float(cfg.get("similarity_noise_epsilon", DEFAULT_SIMILARITY_EPSILON)),
        sensitivity=float(cfg.get("similarity_noise_sensitivity", DEFAULT_SIMILARITY_SENSITIVITY)),
        seed=int(cfg.get("similarity_noise_seed", DEFAULT_SIMILARITY_NOISE_SEED)),
    )

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
