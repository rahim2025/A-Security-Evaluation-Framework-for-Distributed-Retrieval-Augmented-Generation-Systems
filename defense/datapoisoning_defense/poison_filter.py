"""
Ingestion-time content filter for attack/datapoisoning.

Defense principle
------------------
attack/datapoisoning's DataPoisoningAttack injects documents through each
data source's /poison endpoint (drag_data_source/app/server.py), which
accepts any {"id", "text"} pair and indexes it unconditionally -- there is
no check on content, volume, or novelty before a document becomes
retrievable. This module adds that missing content-integrity gate, mirroring
the same "enforce at the boundary the attack actually breaches" principle
defense/ssm_defense used on-chain for the score-manipulation attack: the
check lives where the malicious write happens, not only in an offline
report.

Four independent, individually-weak signals are combined (any one firing is
enough to reject -- see Honesty Note in the README for why thresholds are
deliberately permissive rather than tuned for maximum catch rate):

1. Lexical red flags       -- generic self-referential override/correction
                               language ("disregard this context", "has
                               been deliberately altered", "the verified and
                               official answer is", bracketed meta-markers
                               like "[CORRUPTED]"). Generalizes beyond this
                               repo's own attack script's exact phrasing to
                               the broader class of prompt-injection-style
                               override language real poisoning payloads use.
2. Gibberish tokens        -- fraction of long alphabetic tokens with no
                               vowel at all, OR a contiguous run of several
                               such tokens in a row. Catches the "noise"
                               poison_type's random lowercase strings without
                               needing a dictionary; the run-length check
                               also catches a short noise block appended to an
                               otherwise long, legitimate-looking document,
                               where the overall ratio alone would be diluted
                               below threshold.
3. Near-duplicate / template reuse -- character-shingle Jaccard similarity
                               against other documents in the same batch and
                               a bounded recent-history window. Catches
                               question_variants/amplification_factor copies
                               and the "targeted" poison type's fixed-sentence
                               template reused across many different queries.
4. Burst-volume cap        -- a single /poison call may not inject more than
                               a fraction of the source's current corpus size
                               (nor more than an absolute per-request cap).
                               Forces an attacker who wants real volume into
                               many slow, rate-limited requests instead of one
                               large one.

What this does NOT catch (by design, not oversight)
-----------------------------------------------------
"answer_swap" poisoning reuses a real, well-formed document from elsewhere in
the corpus attached to the wrong record -- its text is indistinguishable from
legitimate content by any of the four signals above. No content-only filter
can catch this; it would need provenance/identity binding between a document
and the source that is authorized to serve it, which is out of scope here.
See the README's Honesty Note.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, FrozenSet, List, Optional, Tuple

WORD_RE = re.compile(r"[a-z]+", re.IGNORECASE)
VOWEL_RE = re.compile(r"[aeiou]", re.IGNORECASE)

# Generic self-referential / override / correction language. Deliberately
# phrased around the *behavior* (claiming to override or correct context,
# labeling content as corrupted/authoritative) rather than lifted verbatim
# from attack/datapoisoning/data_poisoning_attack.py's exact strings, so this
# generalizes to poisoning payloads that don't share this repo's wording.
DEFAULT_LEXICAL_PATTERNS = [
    r"\bdisregard\b.{0,30}\b(context|information|content|answer)\b",
    r"\bignore\b.{0,30}\b(context|instructions|information|previous)\b",
    r"\bintentionally\b.{0,20}\b(corrupt|corrupted)\b",
    r"\bdeliberately\b.{0,20}\b(alter|altered|changed)\b",
    r"\bpoisoned\b",
    r"\bcorrupted\b",
    r"\[corrupted\]",
    r"\bthe (verified|official|authoritative|definitive)\b.{0,20}\banswer\b",
    r"\bdisregard all\b",
]


@dataclass
class PoisonVerdict:
    accepted: bool
    reasons: List[str] = field(default_factory=list)
    gibberish_ratio: float = 0.0
    max_jaccard: float = 0.0

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "reasons": self.reasons,
            "gibberish_ratio": round(self.gibberish_ratio, 3),
            "max_jaccard": round(self.max_jaccard, 3),
        }


class PoisonFilter:
    """
    Stateful content-integrity gate for one data source's ingestion path.

    Call `classify(text, batch_texts_so_far=...)` once per candidate document
    *before* it is added to the retriever index; only index it if
    `verdict.accepted` is True. Call `check_batch_volume(...)` once per
    /poison request to get the max number of documents that request may add.

    Parameters
    ----------
    gibberish_token_ratio : reject if more than this fraction of long (>=4
                            char) alphabetic tokens contain no vowel.
    near_dup_jaccard       : reject if shingle-Jaccard similarity to any other
                            document in the same batch or recent-history
                            window reaches/exceeds this.
    shingle_k              : character shingle length used for near-duplicate
                            detection.
    history_size           : how many recently-accepted documents' shingle
                            signatures are kept for cross-batch near-dup
                            checks (bounded so memory doesn't grow unbounded
                            over a long-running process).
    max_injection_fraction  : a single /poison request may add at most this
                            fraction of the source's current total_docs.
    max_docs_per_request    : absolute per-request cap, applied in addition to
                            max_injection_fraction (whichever is smaller).
    lexical_patterns        : override the default red-flag regex list.
    """

    def __init__(
        self,
        gibberish_token_ratio: float = 0.12,
        gibberish_run_length: int = 5,
        near_dup_jaccard: float = 0.55,
        shingle_k: int = 5,
        history_size: int = 2000,
        max_injection_fraction: float = 0.15,
        max_docs_per_request: int = 200,
        lexical_patterns: Optional[List[str]] = None,
    ):
        self.gibberish_token_ratio = gibberish_token_ratio
        self.gibberish_run_length = gibberish_run_length
        self.near_dup_jaccard = near_dup_jaccard
        self.shingle_k = shingle_k
        self.max_injection_fraction = max_injection_fraction
        self.max_docs_per_request = max_docs_per_request
        self._patterns = [
            re.compile(p, re.IGNORECASE) for p in (lexical_patterns or DEFAULT_LEXICAL_PATTERNS)
        ]
        self._history: Deque[FrozenSet[str]] = deque(maxlen=history_size)

        self.stats = {
            "total_checked": 0,
            "total_rejected": 0,
            "rejected_by_reason": {
                "lexical": 0,
                "gibberish": 0,
                "near_duplicate": 0,
            },
            "volume_capped_requests": 0,
            "volume_capped_docs_dropped": 0,
        }

    # ------------------------------------------------------------------
    # Signal 1 -- lexical red flags
    # ------------------------------------------------------------------
    def _lexical_hit(self, text: str) -> Optional[str]:
        for pattern in self._patterns:
            if pattern.search(text):
                return pattern.pattern
        return None

    # ------------------------------------------------------------------
    # Signal 2 -- gibberish token ratio
    # ------------------------------------------------------------------
    def _gibberish_ratio(self, text: str) -> float:
        tokens = [t for t in WORD_RE.findall(text) if len(t) >= 4]
        if not tokens:
            return 0.0
        gibberish = sum(1 for t in tokens if not VOWEL_RE.search(t))
        return gibberish / len(tokens)

    def _max_gibberish_run(self, text: str) -> int:
        """Longest run of consecutive long (>=4 char) vowel-less tokens.

        The ratio signal above is diluted when a short block of noise tokens
        is appended to a long legitimate document (the noise still makes up
        a small fraction of total tokens even though it's obviously nonsense
        in isolation). A contiguous run catches exactly that shape -- natural
        English essentially never produces several consecutive no-vowel
        "words" -- independent of how long the surrounding document is.
        """
        tokens = [t for t in WORD_RE.findall(text) if len(t) >= 4]
        best = run = 0
        for t in tokens:
            if not VOWEL_RE.search(t):
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best

    # ------------------------------------------------------------------
    # Signal 3 -- near-duplicate / template reuse
    # ------------------------------------------------------------------
    def _shingles(self, text: str) -> FrozenSet[str]:
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        k = self.shingle_k
        if len(normalized) < k:
            return frozenset({normalized}) if normalized else frozenset()
        return frozenset(normalized[i:i + k] for i in range(len(normalized) - k + 1))

    @staticmethod
    def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def _max_jaccard(self, sig: FrozenSet[str], batch_sigs: List[FrozenSet[str]]) -> float:
        best = 0.0
        for other in batch_sigs:
            j = self._jaccard(sig, other)
            if j > best:
                best = j
        for other in self._history:
            j = self._jaccard(sig, other)
            if j > best:
                best = j
        return best

    # ------------------------------------------------------------------
    # Signal 4 -- burst-volume cap
    # ------------------------------------------------------------------
    def check_batch_volume(self, requested_count: int, current_total_docs: int) -> Tuple[int, Optional[str]]:
        """
        Returns (allowed_count, reason). allowed_count < requested_count means
        the tail of the batch should be dropped before per-document checks.
        """
        fraction_cap = int(current_total_docs * self.max_injection_fraction)
        cap = max(1, min(self.max_docs_per_request, fraction_cap) if fraction_cap > 0 else self.max_docs_per_request)
        if requested_count > cap:
            self.stats["volume_capped_requests"] += 1
            self.stats["volume_capped_docs_dropped"] += requested_count - cap
            return cap, f"burst_volume_cap(requested={requested_count}, cap={cap}, current_total={current_total_docs})"
        return requested_count, None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, text: str, batch_sigs_so_far: Optional[List[FrozenSet[str]]] = None) -> PoisonVerdict:
        self.stats["total_checked"] += 1
        reasons: List[str] = []

        lex = self._lexical_hit(text)
        if lex:
            reasons.append(f"lexical:{lex}")

        gratio = self._gibberish_ratio(text)
        grun = self._max_gibberish_run(text)
        is_gibberish = gratio > self.gibberish_token_ratio or grun >= self.gibberish_run_length
        if is_gibberish:
            reasons.append(f"gibberish(ratio={gratio:.2f}, max_run={grun})")

        sig = self._shingles(text)
        max_j = self._max_jaccard(sig, batch_sigs_so_far or [])
        if max_j >= self.near_dup_jaccard:
            reasons.append(f"near_duplicate(jaccard={max_j:.2f})")

        accepted = len(reasons) == 0
        if accepted:
            self._history.append(sig)
        else:
            self.stats["total_rejected"] += 1
            if lex:
                self.stats["rejected_by_reason"]["lexical"] += 1
            if is_gibberish:
                self.stats["rejected_by_reason"]["gibberish"] += 1
            if max_j >= self.near_dup_jaccard:
                self.stats["rejected_by_reason"]["near_duplicate"] += 1

        return PoisonVerdict(accepted=accepted, reasons=reasons, gibberish_ratio=gratio, max_jaccard=max_j)

    def classify_batch(self, texts: List[str]) -> List[PoisonVerdict]:
        """Classify a whole /poison request body, with within-batch near-dup
        detection accumulating as the batch is walked (so document 5 is
        checked against documents 1-4 of the same request, not just history
        from prior requests)."""
        verdicts: List[PoisonVerdict] = []
        batch_sigs: List[FrozenSet[str]] = []
        for text in texts:
            verdict = self.classify(text, batch_sigs_so_far=batch_sigs)
            if verdict.accepted:
                batch_sigs.append(self._shingles(text))
            verdicts.append(verdict)
        return verdicts

    def get_stats(self) -> dict:
        return {
            **self.stats,
            "rejected_by_reason": dict(self.stats["rejected_by_reason"]),
            "history_size": len(self._history),
        }
