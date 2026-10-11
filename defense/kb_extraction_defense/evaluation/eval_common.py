"""
defense/kb_extraction_defense/evaluation/eval_common.py

Shared plumbing for the offline (in-process) KB-extraction evaluation:
corpus loading, cached embeddings, the same dense top-k search the data source
runs, and the question -> gold-document map used for utility measurement.

"In-process" means the real corpus (data/polluted_token/sources_*.jsonl), the real
embedding model (all-MiniLM-L6-v2) and the real retrieval rule (cosine top-k over
normalised embeddings, exactly drag_data_source/src/retriever/retriever.py's dense
path) -- only the HTTP hop is removed, which makes multi-seed sweeps cheap and
deterministic. live_eval.py repeats the headline results over real HTTP.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CACHE_DIR = os.path.join(ROOT, "defense_logs", "kb_extraction_defense", ".cache")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SOURCE_FILES = {0: "sources_0.jsonl", 1: "sources_20.jsonl", 2: "sources_100.jsonl"}


class Embedder:
    """Cached sentence-embedding wrapper (L2-normalised, same model as the retriever)."""

    def __init__(self, cache_name: str = "embeddings"):
        self._model = None
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
        self._cache: Dict[str, np.ndarray] = {}
        if os.path.exists(self._path):
            with open(self._path, "rb") as f:
                self._cache = pickle.load(f)
        self._dirty = False

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME, device="cpu")
        return self._model

    @staticmethod
    def _key(t: str) -> str:
        return hashlib.md5(t.encode("utf-8")).hexdigest()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._cache]
        if missing:
            uniq = {}
            for k, t in missing:
                uniq.setdefault(k, t)
            ks, ts = list(uniq.keys()), list(uniq.values())
            vecs = self.model.encode(ts, convert_to_numpy=True, show_progress_bar=False,
                                     batch_size=64).astype(np.float32)
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
            for k, v in zip(ks, vecs):
                self._cache[k] = v
            self._dirty = True
        return np.stack([self._cache[k] for k in keys]) if keys else np.zeros((0, 384), np.float32)

    def save(self) -> None:
        if self._dirty:
            with open(self._path, "wb") as f:
                pickle.dump(self._cache, f)
            self._dirty = False


@dataclass
class QA:
    question: str
    answers: List[str]
    doc_id: str
    topic: str


@dataclass
class Corpus:
    source: int
    name: str
    ids: List[str]
    texts: List[str]
    emb: np.ndarray
    topics: List[str]                      # per-document topic (SQuAD title / embedding cluster)
    qa: List[QA] = field(default_factory=list)
    id_index: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, q_emb: np.ndarray, k: int = 5) -> List[Dict]:
        """Identical to FastRetriever.search's dense path (normalised cosine, top-k)."""
        scores = self.emb @ q_emb
        k = min(k, len(scores))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [{"id": self.ids[i], "score": float(scores[i]), "text": self.texts[i]} for i in idx]


def _read_jsonl(path: str):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_corpus(source: int, embedder: Embedder, n_pseudo_topics: int = 10, seed: int = 0,
                load_qa: bool = True) -> Corpus:
    recs = _read_jsonl(os.path.join(ROOT, "data", "polluted_token", SOURCE_FILES[source]))
    ids, texts = [], []
    for r in recs:
        ids.append(str(r.get("id") or r.get("htmlid")))
        texts.append(r.get("documents") or r.get("html", ""))
    emb = embedder.embed(texts)
    text_to_idx = {t: i for i, t in enumerate(texts)}
    topics = [""] * len(ids)
    qa: List[QA] = []

    if source in (1, 2):
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad", split="train")
        seen_q = set()
        for item in ds:
            ctx = item["context"]
            i = text_to_idx.get(ctx)
            if i is None:
                continue
            topics[i] = item["title"]
            q = item["question"].strip()
            if load_qa and q not in seen_q and item["answers"]["text"]:
                seen_q.add(q)
                qa.append(QA(q, list(item["answers"]["text"]), ids[i], item["title"]))
        name = "squad"
    else:
        from datasets import load_dataset
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        seen_q = set()
        for item in ds:
            ctx = " ".join(c.strip() for c in item["context"]["contexts"] if c and c.strip())
            i = text_to_idx.get(ctx)
            if i is None:
                continue
            q = item["question"].strip()
            if load_qa and q not in seen_q:
                seen_q.add(q)
                qa.append(QA(q, [item.get("final_decision", "")], ids[i], ""))
        name = "pubmedqa"

    # Documents without a native topic label. SQuAD-backed sources: token-polluted docs no
    # longer exact-match their SQuAD context, so they inherit the title of the nearest
    # labelled topic centroid. PubMedQA has no native topic field, so documents get a
    # deterministic embedding-cluster pseudo-topic.
    if any(t == "" for t in topics):
        labelled = [i for i, t in enumerate(topics) if t]
        if source in (1, 2) and labelled:
            titles = sorted({topics[i] for i in labelled})
            cents = np.stack([emb[[i for i in labelled if topics[i] == t]].mean(axis=0) for t in titles])
            cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)
            for i, t in enumerate(topics):
                if t == "":
                    topics[i] = titles[int(np.argmax(cents @ emb[i]))]
        else:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=n_pseudo_topics, n_init=10, random_state=seed).fit(emb)
            for i, t in enumerate(topics):
                if t == "":
                    topics[i] = f"cluster_{int(km.labels_[i])}"
        for q in qa:
            if q.topic == "":
                q.topic = topics[ids.index(q.doc_id)]
    c = Corpus(source=source, name=name, ids=ids, texts=texts, emb=emb, topics=topics, qa=qa)
    c.id_index = {d: i for i, d in enumerate(ids)}
    return c


def text_to_idx_id(ids: List[str], doc_id: str) -> int:
    return ids.index(doc_id)


def stopwords() -> set:
    return {
        "the", "and", "for", "that", "with", "from", "this", "were", "was", "are", "has", "have",
        "had", "his", "her", "its", "their", "which", "also", "been", "into", "than", "then",
        "them", "these", "those", "such", "only", "over", "after", "before", "during", "about",
        "between", "other", "more", "most", "some", "many", "much", "when", "where", "while",
        "what", "who", "whom", "whose", "will", "would", "could", "should", "being", "both",
        "each", "not", "but", "can", "may", "one", "two", "three", "first", "second", "new",
        "used", "using", "use", "per", "within", "without", "under", "against", "among",
    }


def union_corpus(corpora: Sequence["Corpus"]) -> "Corpus":
    """Federated view: the concatenation of several real data sources (ids prefixed by
    source, topics kept distinct). Used to test how the leakage bound B/N scales with N
    while a legitimate user's working set (bounded by interest, not by corpus size) is
    unchanged."""
    ids, texts, topics, qa = [], [], [], []
    embs = []
    for c in corpora:
        p = f"s{c.source}:"
        ids += [p + i for i in c.ids]
        texts += c.texts
        topics += [f"{c.name}{c.source}/{t}" for t in c.topics]
        embs.append(c.emb)
        qa += [QA(q.question, q.answers, p + q.doc_id, f"{c.name}{c.source}/{q.topic}") for q in c.qa]
    u = Corpus(source=-1, name="union" + "".join(str(c.source) for c in corpora), ids=ids, texts=texts,
               emb=np.concatenate(embs, axis=0), topics=topics, qa=qa)
    u.id_index = {d: i for i, d in enumerate(ids)}
    return u
