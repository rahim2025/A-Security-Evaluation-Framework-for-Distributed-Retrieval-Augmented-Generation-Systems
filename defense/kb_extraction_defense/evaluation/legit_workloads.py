"""
defense/kb_extraction_defense/evaluation/legit_workloads.py

Modelled legitimate-caller sessions. No real user telemetry exists for this
system, so "legitimate" is modelled explicitly and swept over a grid rather than
assumed once. Queries are real dataset questions (natural language, what a user
actually types) with a known gold document, so utility is measurable.

Workloads
---------
focused      a user working on 1-3 topics, with some exact re-asks. The canonical
             legitimate pattern the defense must not disturb.
browsing     a reader who drifts between topics (Markov topic persistence).
aggregate    iid questions across all topics. NOT a single user: this is what an
             aggregating gateway such as the orchestrator looks like to a data
             source. Included to show why such callers must be exempted (trusted).
analyst      a legitimate user who deliberately surveys the WHOLE corpus topic by
             topic. By content this is indistinguishable from extraction; the only
             defense is authorisation. Reported as the honest stress case.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from defense.kb_extraction_defense.evaluation.eval_common import Corpus, QA


@dataclass
class Session:
    kind: str
    params: Dict
    queries: List[QA]


def _by_topic(c: Corpus) -> Dict[str, List[QA]]:
    d: Dict[str, List[QA]] = {}
    for q in c.qa:
        d.setdefault(q.topic, []).append(q)
    return d


def focused(c: Corpus, rng: random.Random, n_topics: int, length: int, repeat_p: float = 0.15) -> Session:
    bt = _by_topic(c)
    topics = rng.sample(sorted(bt), min(n_topics, len(bt)))
    pool = [q for t in topics for q in bt[t]]
    qs: List[QA] = []
    for _ in range(length):
        if qs and rng.random() < repeat_p:
            qs.append(rng.choice(qs))
        else:
            qs.append(rng.choice(pool))
    return Session("focused", {"n_topics": n_topics, "length": length, "repeat_p": repeat_p}, qs)


def browsing(c: Corpus, rng: random.Random, length: int, stay_p: float = 0.8) -> Session:
    bt = _by_topic(c)
    topics = sorted(bt)
    cur = rng.choice(topics)
    qs: List[QA] = []
    for _ in range(length):
        if rng.random() > stay_p:
            cur = rng.choice(topics)
        qs.append(rng.choice(bt[cur]))
    return Session("browsing", {"length": length, "stay_p": stay_p}, qs)


def aggregate(c: Corpus, rng: random.Random, length: int) -> Session:
    return Session("aggregate", {"length": length}, [rng.choice(c.qa) for _ in range(length)])


def analyst(c: Corpus, rng: random.Random, per_topic: int = 12) -> Session:
    bt = _by_topic(c)
    qs: List[QA] = []
    for t in rng.sample(sorted(bt), len(bt)):
        pool = bt[t]
        qs.extend(rng.sample(pool, min(per_topic, len(pool))))
    return Session("analyst", {"per_topic": per_topic, "length": len(qs)}, qs)


def build_suite(c: Corpus, seed: int, n_per_cell: int = 20) -> List[Session]:
    """A grid of legitimate sessions (seeded; calibration and test use different seeds)."""
    rng = random.Random(seed)
    out: List[Session] = []
    for n_topics in (1, 2, 3, 5):
        for length in (20, 50, 100, 200):
            for _ in range(n_per_cell):
                out.append(focused(c, rng, n_topics, length))
    for length in (20, 50, 100, 200):
        for stay in (0.7, 0.9):
            for _ in range(n_per_cell):
                out.append(browsing(c, rng, length, stay))
    return out
