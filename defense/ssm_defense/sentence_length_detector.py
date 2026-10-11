"""
Phrasing-independent complement to content_shape_detector.py.

Why this exists
-----------------
content_shape_detector.py's GroundingFarmDetector flags documents that use
specific hedged-attribution phrasing ("sometimes cited", "in various
contexts", ...). That is real, measured signal (100% detection, 0% false
positives against the actual attack -- see that module's README) -- but it
is a surface-lexical signal. An attacker aware of it could reword the
connective phrases while keeping the exploit intact (e.g. "{t} is connected
to {f}." instead of "Related information about {t} sometimes cited in this
context includes {f}.") and evade it outright: measured at 0/40 (0.0%)
detection against a rewritten filler set using neutral, non-hedging
connectives (see offline_validation.py's adaptive-attacker check).

This module targets a property the rewording does NOT change: the attack's
own stated design goal (.claude/ssm_grounding_farming_plan.md §4) requires
cramming several independent, topically disconnected facts into one
document to maximize broad answer coverage. That forces short, simple
"{topic} {verb} {fact}." sentences almost by construction, regardless of
which verb or connective phrase is used -- there's no room for the
elaboration, subordinate clauses, and explanatory context that make real
encyclopedic prose run long. Measured directly against the real corpus
(see offline_validation.py):

  - Real clean documents: median 23.0 words/sentence (range 10.6-102.0)
  - Original (hedged) filler:  median 12.4 words/sentence
  - Reworded (adaptive) filler: median 6.8 words/sentence

At a 12-word average-sentence-length threshold: 92.5% of adaptive filler is
flagged, vs. 1.3% of real clean documents (2/150 -- both genuinely terse,
list-like real content, inspected manually). This is intentionally a
*complementary* signal, not a replacement: it catches the adaptive case the
hedge-phrase detector misses, while the hedge-phrase detector still covers
the direct, unmodified attack at 100%. Defeating both at once would require
the attacker to write longer, more elaborated sentences while still
cramming in a disconnected fact each time -- raising the cost/sophistication
bar, not a claim of unconditional prevention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class LengthVerdict:
    flagged: bool
    avg_sentence_length: float
    num_sentences: int

    def to_dict(self) -> dict:
        return {
            "flagged": self.flagged,
            "avg_sentence_length": round(self.avg_sentence_length, 1),
            "num_sentences": self.num_sentences,
        }


class SentenceLengthDetector:
    """
    Flags documents whose average sentence length (in words) is
    anomalously short -- the structural shape grounding-farming filler
    needs regardless of connective phrasing (see module docstring).

    Parameters
    ----------
    max_avg_words   : flag if the document's average words-per-sentence is
                      at or below this. Calibrated against real clean
                      corpus documents and real (including adaptively
                      reworded) attack filler -- see offline_validation.py.
    min_sentences   : documents with fewer sentences than this aren't
                      evaluated (a single long sentence isn't comparable
                      to the attack's many-short-sentences shape) and pass
                      through unflagged.
    """

    def __init__(self, max_avg_words: float = 12.0, min_sentences: int = 2):
        self.max_avg_words = max_avg_words
        self.min_sentences = min_sentences
        self.stats = {"total_checked": 0, "total_flagged": 0, "total_skipped_too_short": 0}

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def classify(self, text: str) -> LengthVerdict:
        self.stats["total_checked"] += 1
        sentences = self._split_sentences(text)

        if len(sentences) < self.min_sentences:
            self.stats["total_skipped_too_short"] += 1
            return LengthVerdict(flagged=False, avg_sentence_length=0.0, num_sentences=len(sentences))

        lengths = [len(WORD_RE.findall(s)) for s in sentences]
        avg = sum(lengths) / len(lengths)
        flagged = avg <= self.max_avg_words

        if flagged:
            self.stats["total_flagged"] += 1

        return LengthVerdict(flagged=flagged, avg_sentence_length=avg, num_sentences=len(sentences))

    def classify_batch(self, texts: List[str]) -> List[LengthVerdict]:
        return [self.classify(t) for t in texts]

    def get_stats(self) -> dict:
        return dict(self.stats)
