"""
Ingestion-time content-shape detector for SSM grounding-farming filler.

Why this exists (and why the first design attempt here was abandoned)
------------------------------------------------------------------------
The first design for this defense gated the *grounding decision* in
drag_llm_service/app/server.py on relevance signals the retrieval pipeline
already computes (rerank_score, consensus_score) -- the hypothesis being
that broad, generic filler should score as a weaker, more marginal
retrieval hit than genuinely on-topic content.

Offline validation against the real retrieval/reranking pipeline and real
grounding-farming filler (defense/ssm_defense/evaluation/offline_validation.py,
200 real SQuAD questions) disproved this: filler documents that survive
retrieval and trip the naive substring grounding check score, if anything,
*higher* on consensus_score (median 1.00 vs genuine content's 0.67) and
indistinguishably on rerank_score (0.61-0.82, squarely inside genuine
content's own 0.47-0.93 range). The intuition that broad/generic content
is a "weak, marginal" retrieval hit turned out to be wrong for this
specific attack once it actually survives the funnel: content built to be
topically broad can look like an unremarkable, averagely-relevant
candidate, not an obvious outlier, on these particular signals. Shipping
that gate anyway would have cost ~19% of genuinely grounded sources their
deserved credit while blocking approximately none of the actual attack --
worse than doing nothing. (The gate itself is left in server.py, disabled
by default, with this finding documented in the README -- a negative
result worth keeping on record, not hidden.)

This module targets a different, more reliable property instead: *what the
filler text itself is shaped like*, independent of how it scores at
retrieval time. Grounding-farming's content (attack/ssm_score/
grounding_farming_attack.py's SUBTOPIC_TEMPLATES / FRAGMENT_TEMPLATES) is
built by stitching together several short, independent factual claims,
each introduced by a vague, non-committal attribution phrase ("sometimes
cited", "occasionally reference", "in various contexts", "has been
associated with") -- because the attacker does not actually know which
specific fact a given query will need, so the content hedges across many
unrelated facts rather than committing to one coherent claim the way a
genuine encyclopedic passage does. That hedging density is the signal here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# Generic vague-attribution / hedging constructions used to loosely link a
# topic to an unrelated, discondeded factual fragment, rather than making a
# direct, committed claim. Phrased around the *rhetorical pattern* (hedged,
# non-committal attribution of a fact to an unspecified "some"/"other"
# source) rather than lifted verbatim from grounding_farming_attack.py's
# exact templates, so this generalizes to other broad-coverage filler
# strategies that share the same underlying need to hedge across many
# disconnected facts.
HEDGE_ATTRIBUTION_PATTERNS = [
    r"\bsometimes (cited|mentioned|referenced|discussed|associated)\b",
    r"\boccasionally\b",
    r"\bin various contexts\b",
    r"\bhas(?:\s+\w+){0,2}\s+been associated with\b",
    r"\bother accounts\b",
    r"\bsome records\b",
    r"\brelated information\b",
    r"\brecurring theme\b",
    r"\bfrequent attention\b",
    r"\bamong other details\b",
    r"\balongside these details\b",
    r"\bin connection with\b",
    r"\brecords mention\b",
    r"\baccounts (?:of|reference)\b",
    r"\banalyses of\b.{0,20}\bfocus\b",
    r"\bcoverage of\b.{0,20}\boften touches\b",
]


@dataclass
class ShapeVerdict:
    flagged: bool
    hedge_match_count: int
    distinct_sentences_with_hedge: int
    matches: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "flagged": self.flagged,
            "hedge_match_count": self.hedge_match_count,
            "distinct_sentences_with_hedge": self.distinct_sentences_with_hedge,
            "matches": self.matches,
        }


class GroundingFarmDetector:
    """
    Flags documents built by stitching together several short factual claims
    behind vague, hedged attribution language -- the structural hallmark of
    broad-coverage grounding-farming filler (see module docstring).

    Parameters
    ----------
    min_hedge_sentences : flag a document once at least this many of its
                         sentences contain a hedge-attribution construction.
                         Genuine single-topic prose very rarely needs this
                         phrasing more than once, if at all; farming filler
                         needs it in nearly every sentence by design (each
                         sentence hedges a different, otherwise-unrelated
                         fact onto the topic anchor).
    """

    def __init__(self, min_hedge_sentences: int = 2):
        self.min_hedge_sentences = min_hedge_sentences
        self._patterns = [re.compile(p, re.IGNORECASE) for p in HEDGE_ATTRIBUTION_PATTERNS]
        self.stats = {"total_checked": 0, "total_flagged": 0}

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def classify(self, text: str) -> ShapeVerdict:
        self.stats["total_checked"] += 1
        sentences = self._split_sentences(text)

        matches: List[str] = []
        hedged_sentence_count = 0
        for sentence in sentences:
            hit = False
            for pattern in self._patterns:
                m = pattern.search(sentence)
                if m:
                    matches.append(m.group(0))
                    hit = True
            if hit:
                hedged_sentence_count += 1

        flagged = hedged_sentence_count >= self.min_hedge_sentences
        if flagged:
            self.stats["total_flagged"] += 1

        return ShapeVerdict(
            flagged=flagged,
            hedge_match_count=len(matches),
            distinct_sentences_with_hedge=hedged_sentence_count,
            matches=matches,
        )

    def classify_batch(self, texts: List[str]) -> List[ShapeVerdict]:
        return [self.classify(t) for t in texts]

    def get_stats(self) -> dict:
        return dict(self.stats)
