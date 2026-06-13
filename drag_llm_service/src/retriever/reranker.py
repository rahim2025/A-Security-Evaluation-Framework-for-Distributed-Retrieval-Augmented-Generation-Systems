"""
Reranker: re-ranks retrieved documents across multiple sources and returns top-k.

Features:
- Dense similarity reranking using SentenceTransformers (cosine)
- Optional BM25 reranking when rank_bm25 is installed
- Hybrid fusion of dense and BM25 with configurable alpha
- Can optionally fuse original retriever scores as a prior
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Optional imports with graceful fallback
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False
    SentenceTransformer = None  # type: ignore

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _BM25_AVAILABLE = True
except Exception:
    _BM25_AVAILABLE = False
    BM25Okapi = None  # type: ignore


WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class Reranker:
    """
    Re-rank candidate documents with one of the following strategies:

    - method="dense": cosine similarity with a SentenceTransformer model
    - method="bm25": BM25Okapi term-based scoring
    - method="hybrid": linear fusion of dense and BM25 scores

    Optionally includes a prior from the original retriever score (orig_score_weight).

    Candidate schema: a dict per doc with at least {"text"}. Optionally include
    {"id", "score" (original retriever score), "meta"}.
    """

    def __init__(
        self,
        method: str = "dense",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_dense: bool = True,
        hybrid_alpha: float = 0.3,
        orig_score_weight: float = 0.0,
        batch_size: int = 64,
    ) -> None:
        self.method = method.lower().strip()
        self.model_name = model_name
        self.device = device
        self.normalize_dense = normalize_dense
        self.hybrid_alpha = float(max(0.0, min(1.0, hybrid_alpha)))
        self.orig_score_weight = float(max(0.0, min(1.0, orig_score_weight)))
        self.batch_size = batch_size

        self.model: Optional[SentenceTransformer] = None
        if self.method in {"dense", "hybrid"}:
            if not _ST_AVAILABLE:
                raise SystemExit(
                    "Dense/hybrid reranking requires 'sentence-transformers'. Install with: \n"
                    "  pip install sentence-transformers"
                )
            self.model = SentenceTransformer(self.model_name, device=self.device)

        # BM25 is instantiated per-query on candidate texts for simplicity

    def _embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        assert self.model is not None
        embs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            vecs = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            embs.append(vecs.astype(np.float32))
        arr = np.vstack(embs) if embs else np.zeros((0, 384), dtype=np.float32)
        return _l2_normalize_rows(arr) if self.normalize_dense and arr.size else arr

    def _dense_scores(self, query: str, cand_texts: Sequence[str]) -> np.ndarray:
        q = self._embed_texts([query])
        d = self._embed_texts(cand_texts)
        if q.size == 0 or d.size == 0:
            return np.zeros(len(cand_texts), dtype=np.float32)
        return (d @ q[0]).astype(np.float32)

    def _bm25_scores(self, query: str, cand_texts: Sequence[str]) -> np.ndarray:
        if not _BM25_AVAILABLE:
            return np.zeros(len(cand_texts), dtype=np.float32)
        corpus_tokens = [_tokenize(t) for t in cand_texts]
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(_tokenize(query))
        return np.asarray(scores, dtype=np.float32)

    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank a list of candidate documents.

        - query: question text
        - candidates: list of dicts with at least {"text"}; may include {"id","score","meta"}
        - top_k: number of documents to return
        Returns: candidates sorted by fused rerank score with fields preserved and
                 added keys: {"rerank_score", "orig_score"}
        """

        texts = [str(c.get("text") or "") for c in candidates]
        orig_scores = np.asarray([float(c.get("score", 0.0)) for c in candidates], dtype=np.float32)

        dense_scores = None
        bm25_scores = None

        if self.method in {"dense", "hybrid"}:
            dense_scores = self._dense_scores(query, texts)
        if self.method in {"bm25", "hybrid"}:
            bm25_scores = self._bm25_scores(query, texts)

        # Normalize components to [0,1] to stabilize fusion
        fused: np.ndarray
        if self.method == "dense":
            ds = _minmax(dense_scores) if dense_scores is not None else np.zeros_like(orig_scores)
            fused = ds
        elif self.method == "bm25":
            bs = _minmax(bm25_scores) if bm25_scores is not None else np.zeros_like(orig_scores)
            fused = bs
        elif self.method == "hybrid":
            ds = _minmax(dense_scores) if dense_scores is not None else np.zeros_like(orig_scores)
            bs = _minmax(bm25_scores) if bm25_scores is not None else np.zeros_like(orig_scores)
            fused = (1.0 - self.hybrid_alpha) * ds + self.hybrid_alpha * bs
        else:
            raise ValueError(f"Unknown rerank method: {self.method}")

        if self.orig_score_weight > 0.0:
            prior = _minmax(orig_scores)
            fused = (1.0 - self.orig_score_weight) * fused + self.orig_score_weight * prior

        # Sort and slice
        k = max(1, min(top_k, len(candidates)))
        order = np.argsort(-fused)[:k]

        results: List[Dict[str, Any]] = []
        for idx in order:
            c = dict(candidates[int(idx)])
            c["orig_score"] = float(orig_scores[int(idx)])
            c["rerank_score"] = float(fused[int(idx)])
            results.append(c)
        return results

    def rerank_with_reliability(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        top_k: int = 10,
        reliability_weight: float = 0.2,
        reliability_meta_key: str = "source",
        reliability_scores: Dict[str, float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank candidates while incorporating per-source reliability.

        - reliability_scores: mapping from source_id -> reliability score (arbitrary scale)
        - reliability_weight: weight for reliability in [0,1]. 0 uses only model scores; 1 uses only reliability
        - reliability_meta_key: key inside candidate["meta"] that stores the source id
        """

        texts = [c['text'] for c in candidates]
        orig_scores = np.asarray([c["score"] for c in candidates], dtype=np.float32) # final retrieval score

        dense_scores = None
        bm25_scores = None

        if self.method in {"dense", "hybrid"}:
            dense_scores = self._dense_scores(query, texts)
        if self.method in {"bm25", "hybrid"}:
            bm25_scores = self._bm25_scores(query, texts)

        if self.method == "dense":
            ds = _minmax(dense_scores) if dense_scores is not None else np.zeros_like(orig_scores)
            fused = ds
        elif self.method == "bm25":
            bs = _minmax(bm25_scores) if bm25_scores is not None else np.zeros_like(orig_scores)
            fused = bs
        elif self.method == "hybrid":
            ds = _minmax(dense_scores) if dense_scores is not None else np.zeros_like(orig_scores)
            bs = _minmax(bm25_scores) if bm25_scores is not None else np.zeros_like(orig_scores)
            fused = (1.0 - self.hybrid_alpha) * ds + self.hybrid_alpha * bs
        else:
            raise ValueError(f"Unknown rerank method: {self.method}")

        if self.orig_score_weight > 0.0:
            prior = _minmax(orig_scores)
            fused = (1.0 - self.orig_score_weight) * fused + self.orig_score_weight * prior

        reliability_raw = []
        for c in candidates:
            src_id = c["meta"][reliability_meta_key]
            r = float(reliability_scores[src_id])
            reliability_raw.append(r)
        reliability_raw_arr = np.asarray(reliability_raw, dtype=np.float32)

        reliability_norm = _minmax(reliability_raw_arr)

        w = float(max(0.0, min(1.0, reliability_weight)))
        combined = (1.0 - w) * fused + w * reliability_norm

        k = max(1, min(top_k, len(candidates)))
        order = np.argsort(-combined)[:k]

        results: List[Dict[str, Any]] = []
        for idx in order:
            i = int(idx)
            c = dict(candidates[i])
            c["orig_score"] = float(orig_scores[i])
            c["base_rerank_score"] = float(fused[i])
            c["reliability_score"] = float(reliability_norm[i])
            c["rerank_score"] = float(combined[i])
            results.append(c)

        return results
