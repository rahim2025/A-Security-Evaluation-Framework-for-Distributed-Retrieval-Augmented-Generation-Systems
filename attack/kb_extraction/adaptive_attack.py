"""
attack/kb_extraction/adaptive_attack.py

Adaptive KB-extraction attackers, written against an abstract `Transport` so the
same attacker runs (a) in-process against a simulated server for fast multi-seed
sweeps and (b) over real HTTP against the Docker data sources.

Why this exists: run_attack.py sends a fixed list of 54 dataset questions. That
measures what a naive scraper recovers, not what a competent adversary can do,
and a defense validated only against it proves little. These attackers use ONLY
what a real caller sees (the returned documents) to steer the next query.

CLAUDE.md constraint honoured: no attacker ever sends stored document text as a
probe. Every query is a short bag of keywords (3-4 words) sampled from retrieved
documents, or a generic public seed question.

Attackers
---------
SnowballCrawler(selection=...)  harvest keywords from every newly retrieved
    document, queue keyword-bag queries, repeat. `selection` picks the next query:
      "random"    random walk over the candidate queue.
      "frontier"  novelty-seeking: choose the candidate LEAST similar to anything
                  already retrieved (white-box worst case: the attacker is assumed
                  to know the public embedding model, so it can estimate where
                  unexplored territory is). Maximises documents discovered/query.
      "local"     stealth: stay close to the previous query. Low query dispersion,
                  mimicking a focused legitimate user.
      "dwell"     white-box adaptive: knows a breadth-style detector exists, so it keeps a
                  FIXED anchor and exhausts one region of the corpus (queries nearest the
                  anchor) before jumping to the next, keeping the number of distinct
                  regions per window small while still harvesting everything.
    `dilution=r` interleaves r repeats of already-asked queries after every real
    query -- an attempt to dilute the discovery-velocity signal (costs throughput).
StaticProbes  the original behaviour: send a fixed list of questions in order.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

_WORD = re.compile(r"[A-Za-z][A-Za-z\-']{3,}")
_STOP = {
    "that", "with", "from", "this", "were", "which", "their", "also", "been", "into", "than",
    "then", "them", "these", "those", "such", "only", "over", "after", "before", "during",
    "about", "between", "other", "more", "most", "some", "many", "much", "when", "where",
    "while", "what", "whom", "whose", "will", "would", "could", "should", "being", "both",
    "each", "have", "has", "had", "within", "without", "under", "against", "among", "first",
    "second", "three", "used", "using", "there", "because", "however", "while", "since",
}


@dataclass
class QueryResult:
    status: str                      # "ok" | "blocked" | "ratelimited" | "error"
    docs: List[Dict] = field(default_factory=list)   # [{"id", "text"}]


class Transport(Protocol):
    def query(self, text: str, k: int = 5) -> QueryResult: ...


@dataclass
class TraceRecord:
    index: int
    text: str
    status: str
    doc_ids: List[str]
    new_docs: int
    decoy: bool = False


@dataclass
class Trace:
    attacker: str
    records: List[TraceRecord] = field(default_factory=list)
    known: Dict[str, str] = field(default_factory=dict)

    def extracted_ids(self, upto: Optional[int] = None) -> set:
        out = set()
        for r in self.records[: upto if upto is not None else len(self.records)]:
            out.update(r.doc_ids)
        return out


def _keywords(text: str, df: Counter, n_docs: int, rng: random.Random, top: int = 12) -> List[str]:
    # sorted(): a set's iteration order varies per process (hash randomisation) and would make
    # a seeded crawl non-reproducible
    words = sorted({w.lower() for w in _WORD.findall(text)})
    words = [w for w in words if w not in _STOP]
    if not words:
        return []
    # rarer-in-what-I-already-know = more informative / more likely at the frontier
    scored = sorted(words, key=lambda w: (-(math.log((1 + n_docs) / (1 + df.get(w, 0)))) + rng.random() * 0.5))
    return scored[:top]


class SnowballCrawler:
    def __init__(self, selection: str = "frontier", seed_queries: Sequence[str] = (),
                 kw_per_query: int = 3, cands_per_doc: int = 4, dilution: int = 0, stall: int = 8,
                 embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
                 k: int = 5, name: Optional[str] = None):
        assert selection in ("random", "frontier", "local", "dwell")
        if selection != "random" and embed_fn is None:
            raise ValueError("embed_fn required for frontier/local selection")
        self.selection = selection
        self.seed_queries = list(seed_queries)
        self.kw = kw_per_query
        self.cands_per_doc = cands_per_doc
        self.dilution = dilution
        self.stall = stall
        self.embed_fn = embed_fn
        self.k = k
        self.name = name or f"snowball_{selection}" + (f"_dil{dilution}" if dilution else "")

    def run(self, transport: Transport, max_queries: int, seed: int = 0) -> Trace:
        rng = random.Random(seed)
        trace = Trace(self.name)
        known: Dict[str, str] = trace.known
        df: Counter = Counter()
        queue: List[str] = []
        asked: List[str] = []
        asked_set = set()
        seeds = list(self.seed_queries)
        rng.shuffle(seeds)
        known_emb: List[np.ndarray] = []
        last_q_emb: Optional[np.ndarray] = None
        anchor: Optional[np.ndarray] = None
        stalled = 0
        idx = 0

        def issue(text: str, decoy: bool = False):
            nonlocal idx
            res = transport.query(text, self.k)
            ids = [d["id"] for d in res.docs]
            new = [d for d in res.docs if d["id"] not in known]
            trace.records.append(TraceRecord(idx, text, res.status, ids, len(new), decoy))
            idx += 1
            return res, new

        while idx < max_queries:
            if not queue:
                if not seeds:
                    break
                queue.append(seeds.pop())
            # ---- pick next query
            if self.selection == "random" or len(queue) == 1 or not known_emb:
                q = queue.pop(rng.randrange(len(queue)))
                q_emb = None
            else:
                embs = self.embed_fn(queue)
                if self.selection == "frontier":
                    kn = np.stack(known_emb)
                    score = (embs @ kn.T).max(axis=1)           # similarity to nearest known doc
                    best = np.flatnonzero(score <= score.min() + 1e-6)
                elif self.selection == "dwell":
                    kn = np.stack(known_emb)
                    if anchor is None or stalled >= self.stall:
                        # jump: re-anchor at the frontier (candidate least similar to known docs)
                        far = (embs @ kn.T).max(axis=1)
                        anchor = embs[int(np.argmin(far))]
                        stalled = 0
                    score = embs @ anchor
                    best = np.flatnonzero(score >= score.max() - 1e-6)
                else:  # local
                    score = (embs @ last_q_emb) if last_q_emb is not None else np.zeros(len(queue))
                    best = np.flatnonzero(score >= score.max() - 1e-6)
                j = int(rng.choice(list(best)))
                q = queue.pop(j)
                q_emb = embs[j]
            if q in asked_set:
                continue
            asked.append(q)
            asked_set.add(q)

            res, new = issue(q)
            if self.selection == "local":
                last_q_emb = q_emb if q_emb is not None else (self.embed_fn([q])[0] if self.embed_fn else None)

            stalled = 0 if new else stalled + 1
            # ---- learn from the response
            if new:
                new_embs = self.embed_fn([d["text"] for d in new]) if self.embed_fn else None
                for n_i, d in enumerate(new):
                    known[d["id"]] = d["text"]
                    for w in {w.lower() for w in _WORD.findall(d["text"])}:
                        df[w] += 1
                    if new_embs is not None:
                        known_emb.append(new_embs[n_i])
                    kws = _keywords(d["text"], df, len(known), rng)
                    for _ in range(self.cands_per_doc):
                        if len(kws) >= self.kw:
                            cand = " ".join(rng.sample(kws[: max(self.kw + 2, 8)], self.kw))
                            if cand not in asked_set:
                                queue.append(cand)

            # ---- dilution: repeat already-asked queries to lower measured discovery velocity
            for _ in range(self.dilution):
                if idx >= max_queries or not asked:
                    break
                issue(rng.choice(asked), decoy=True)
        return trace


class StaticProbes:
    def __init__(self, questions: Sequence[str], k: int = 5, name: str = "static_probes"):
        self.questions = list(questions)
        self.k = k
        self.name = name

    def run(self, transport: Transport, max_queries: int, seed: int = 0) -> Trace:
        trace = Trace(self.name)
        for i, q in enumerate(self.questions[:max_queries]):
            res = transport.query(q, self.k)
            ids = [d["id"] for d in res.docs]
            new = [d for d in res.docs if d["id"] not in trace.known]
            for d in res.docs:
                trace.known.setdefault(d["id"], d["text"])
            trace.records.append(TraceRecord(i, q, res.status, ids, len(new)))
        return trace


def extraction_curve(trace: Trace, corpus_size: int, checkpoints: Sequence[int]) -> Dict[int, float]:
    """Fraction of the corpus recovered after each number of query ATTEMPTS."""
    return {c: round(len(trace.extracted_ids(c)) / corpus_size, 4) for c in checkpoints}
