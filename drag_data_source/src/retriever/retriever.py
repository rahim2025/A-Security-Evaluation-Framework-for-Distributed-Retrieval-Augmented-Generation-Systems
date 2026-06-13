import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

import numpy as np
from tqdm import tqdm

# Optional imports with graceful fallback
try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False
    faiss = None

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except Exception:
    _BM25_AVAILABLE = False
    BM25Okapi = None 

try:
    from sentence_transformers import SentenceTransformer 
except Exception as e:
    raise SystemExit(
        "Missing dependency 'sentence-transformers'. Install with:\n"
        "  pip install sentence-transformers"
    )


WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


@dataclass
class Document:
    id: str
    text: str
    meta: Optional[Dict[str, Any]] = None


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


class FastRetriever:
    """
    A compact, fast retriever that supports:
      - Dense vector search with SentenceTransformers + FAISS (or NumPy fallback)
      - Optional BM25 hybrid retrieval via linear score fusion
      - Simple on-disk persistence (JSONL + NPY + optional FAISS index)
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        use_faiss: Optional[bool] = None,
        device: Optional[str] = None,
        batch_size: int = 64
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.batch_size = batch_size
        self.use_faiss = _FAISS_AVAILABLE if use_faiss is None else bool(use_faiss)

        # In-memory state
        self.docs: List[Document] = []
        self.emb: Optional[np.ndarray] = None  # (N, D)
        self.faiss_index = None
        self.bm25 = None

    # ------------------------ Index building ------------------------ #
    def fit(self, docs: Iterable[Document]) -> None:
        """Build the index from an iterable of Document."""
        self.docs = list(docs)
        if not self.docs:
            raise ValueError("No documents provided to index.")

        texts = [d.text for d in self.docs]
        self.emb = self._embed_texts(texts, show_progress=True)

        if self.normalize:
            self.emb = _normalize_rows(self.emb)

        if self.use_faiss:
            self._build_faiss(self.emb)

        if _BM25_AVAILABLE:
            tokenized = [_tokenize(t) for t in texts]
            self.bm25 = BM25Okapi(tokenized)

    def _embed_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        embs: List[np.ndarray] = []
        it = range(0, len(texts), self.batch_size)
        if show_progress:
            it = tqdm(it, desc="Embedding", unit="batch")
        for i in it:
            batch = texts[i:i + self.batch_size]
            vecs = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
            embs.append(vecs.astype(np.float32))
        return np.vstack(embs)

    def _build_faiss(self, emb: np.ndarray) -> None:
        d = emb.shape[1]
        index = faiss.IndexFlatIP(d)  # cosine via normalized vectors
        index.add(emb)
        self.faiss_index = index

    # ------------------------ Querying ------------------------ #
    def search(self, query: str, k: int = 10, hybrid_alpha: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents.
        - Dense search always runs.
        - If hybrid_alpha provided (0..1) and BM25 available, perform score fusion:
            score = (1 - alpha) * dense + alpha * bm25
        Returns: list of {"id","score","text","meta"}
        """
        if self.emb is None or not self.docs:
            raise RuntimeError("Index is empty. Call fit(...) first or load an index.")

        # Dense score
        q = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)[0]
        if self.normalize:
            q = q / (np.linalg.norm(q) + 1e-12)

        if self.use_faiss and self.faiss_index is not None:
            scores, idxs = self.faiss_index.search(q.reshape(1, -1), min(k, len(self.docs)))
            dense_scores = scores[0]
            dense_idxs = idxs[0]
        else:
            dense_scores = self.emb @ q  # cosine if normalized
            dense_idxs = np.argpartition(-dense_scores, kth=min(k, len(dense_scores)-1))[:k]
            order = np.argsort(-dense_scores[dense_idxs])
            dense_idxs = dense_idxs[order]
            dense_scores = dense_scores[dense_idxs]

        # Hybrid fusion if requested and bm25 available
        if hybrid_alpha is not None and _BM25_AVAILABLE and self.bm25 is not None:
            alpha = float(hybrid_alpha)
            alpha = min(max(alpha, 0.0), 1.0)

            # Compute BM25 for all docs
            bm25_scores = np.array(self.bm25.get_scores(_tokenize(query)), dtype=np.float32)

            # Normalize to [0,1] for stable fusion
            def _minmax(x: np.ndarray) -> np.ndarray:
                lo, hi = float(np.min(x)), float(np.max(x))
                if hi == lo:
                    return np.zeros_like(x)
                return (x - lo) / (hi - lo)

            # If FAISS only gave us top-k dense scores, compute full dense for better fusion if no FAISS
            if len(dense_scores) < len(self.docs) and not (self.use_faiss and self.faiss_index is not None):
                dense_full = self.emb @ q
            else:
                dense_full = np.full(len(self.docs), -np.inf, dtype=np.float32)
                dense_full[dense_idxs] = dense_scores

            dense_n = _minmax(dense_full)
            bm25_n = _minmax(bm25_scores)
            fused = (1.0 - alpha) * dense_n + alpha * bm25_n

            topk = np.argpartition(-fused, kth=min(k, len(fused)-1))[:k]
            order = np.argsort(-fused[topk])
            final_idxs = topk[order]
            final_scores = fused[final_idxs]
        else:
            final_idxs = dense_idxs
            final_scores = dense_scores

        results: List[Dict[str, Any]] = []
        for rank, (idx, sc) in enumerate(zip(final_idxs, final_scores), start=1):
            d = self.docs[int(idx)]
            results.append({
                "rank": rank,
                "id": d.id,
                "score": float(sc),
                "text": d.text,
                "meta": d.meta or {}
            })
        return results

    def save(self, out_dir: str) -> None:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)

        # Config
        cfg = {
            "model_name": self.model_name,
            "normalize": self.normalize,
            "use_faiss": bool(self.use_faiss and _FAISS_AVAILABLE),
            "count": len(self.docs),
            "dim": int(self.emb.shape[1]) if self.emb is not None else None,
        }
        (p / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        # Store
        with (p / "store.jsonl").open("w", encoding="utf-8") as f:
            for d in self.docs:
                rec = {"id": d.id, "text": d.text, "meta": d.meta}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Embeddings
        if self.emb is None:
            raise RuntimeError("No embeddings to save. Did you call fit(...)?")
        np.save(p / "emb.npy", self.emb.astype(np.float32))

        # FAISS
        if self.use_faiss and _FAISS_AVAILABLE and self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(p / "faiss.index"))


    @classmethod
    def load(cls, in_dir: str) -> "FastRetriever":
        p = Path(in_dir)
        cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
        obj = cls(
            model_name=cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            normalize=bool(cfg.get("normalize", True)),
            use_faiss=bool(cfg.get("use_faiss", True))
        )

        # Load store
        docs: List[Document] = []
        with (p / "store.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                docs.append(Document(id=str(rec.get("id")), text=str(rec.get("text")), meta=rec.get("meta")))
        obj.docs = docs

        # Load embeddings
        obj.emb = np.load(p / "emb.npy")
        if obj.normalize:
            obj.emb = _normalize_rows(obj.emb.astype(np.float32))

        # Rebuild or load FAISS
        if obj.use_faiss and _FAISS_AVAILABLE and (p / "faiss.index").exists():
            obj.faiss_index = faiss.read_index(str(p / "faiss.index"))
        elif obj.use_faiss and _FAISS_AVAILABLE:
            obj._build_faiss(obj.emb)

        # Rebuild BM25 if available
        if _BM25_AVAILABLE:
            tokenized = [_tokenize(d.text) for d in obj.docs]
            obj.bm25 = BM25Okapi(tokenized)

        return obj
