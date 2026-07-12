"""
MIA (Membership Inference Attack) Defense for Reliable-dRAG.

Attack recap (attack/Mia_attack/mia_attack.py)
-----------------------------------------------
A real PubMedQA question is sent to /query. Member questions have their
context passage actually loaded into the corpus (the first 500
`qiaojin/PubMedQA` `pqa_labeled` rows, written to sources_0.jsonl by
data/build_pubmedqa_corpus.py); non-member questions' contexts are the
remaining ~500 rows of the same dataset/split, genuinely never loaded --
same-domain, disjoint documents, not a different corpus/domain entirely
(avoids a topic-confound that would make any AUC lift ambiguous). PubMedQA
replaced an earlier SQuAD-based version of this attack after SQuAD's
Wikipedia-derived content produced a composite AUC-ROC consistently *below*
0.50 across four seeds -- the deployed LLM (Qwen2.5-1.5B-Instruct) had
almost certainly memorized much of that content during pretraining,
independent of retrieval. For each document the attacker probes the
available question(s) and, on the probe with the highest cosine similarity
between the response and the context passage, computes a **gated**
composite membership score (3rd revision):

    membership_score = DECISION_WEIGHT * decision_match
                      + SIM_WEIGHT * normalized_similarity
                      + CERTAINTY_WEIGHT * certainty
                      + LEN_WEIGHT * length_ratio

with `DECISION_WEIGHT (0.55) > SIM_WEIGHT+CERTAINTY_WEIGHT+LEN_WEIGHT
(0.20+0.10+0.15=0.45)`, so `decision_match` (does the response commit to the
correct gold yes/no/maybe judgment) *gates* the ranking -- any document with
a correct decision-match scores strictly higher than one without, regardless
of the other three signals, which only re-rank within a gate tier. Earlier
revisions used a plain linear blend where `length_ratio`'s occasional
sign-inversion at larger sample sizes could drag a correctly-gated document
back down (see reports/MIA_Security_Analysis_Report.md §12 for the full
before/after evidence, including the seed that regressed under the linear
blend and recovered under the gate). AUC-ROC over this composite score
across many probes is the leakage metric (see thesis_progress_tracker PDF:
"MIA — '50% neutralised' claim is weak" -> fixed by reporting AUC-ROC
instead of a single threshold accuracy). This defense module must be
evaluated against the *same* composite score the attack now uses -- scoring
plain cosine similarity instead would compare against a metric the attacker
no longer relies on.

Defense principle -- two complementary defenses, two different leak channels
-------------------------------------------------------------------------------
Reliable-dRAG's `/query` only ever returns `{"response": text}` -- there's no
numeric score field to add calibrated noise to, unlike the original DRAG
system's confidence-score noise defense. Two independent text-level defenses
are implemented, each targeting a different channel the attack exploits:

  1. **`sanitize_response()`** -- caps response length and redacts long
     verbatim overlaps with the retrieved context. Targets the *content*
     leak channel (does the response quote source text at length). Does not
     touch retrieval or generation quality for legitimate use -- only
     responses quoting a specific passage at length get capped.
  2. **`obfuscate_decision()`** (new) -- prepends a fixed hedging phrase to
     *every* response, pushing any yes/no/maybe commitment out of the
     leading-words window `decision_match` (the attack's strongest, most
     consistent signal -- see §12/§13 of the MIA report) inspects. Targets
     the *positional* leak channel specifically, and is disclosed as scoped
     to that: it does not remove the correct decision from the response
     text, and provides no protection against an attacker who scans the
     whole response instead of only its opening words.

This module evaluates both, applied together (`sanitize_response()` then
`obfuscate_decision()`), against the *live* LLM service without requiring a
redeploy: it sends the exact same probes the attack does, then scores raw,
sanitized-only, and sanitized+obfuscated responses side by side. See
README.md for the drop-in patch to actually wire these into
drag_llm_service/app/server.py.
"""
from __future__ import annotations

import os
import random
import re
import statistics
import sys
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.Mia_attack.mia_attack import (  # noqa: E402
    CORPUS_JSONL,
    DEFAULT_MEMBERS,
    DEFAULT_NONMEMBERS,
    EMBEDDING_MODEL,
    LLM_SERVICE_URL,
    CERTAINTY_WEIGHT,
    DECISION_WEIGHT,
    LEN_WEIGHT,
    SIM_WEIGHT,
    _DECISION_SYNONYMS,
    _answer_length_ratio,
    _certainty_score,
    _cosine_similarity,
    _decision_match,
    _decision_match_adaptive,
    _decision_match_semantic,
    _normalize_similarity,
    _query_llm,
    load_membership_documents,
)

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# Mirrors the values the drop-in server.py patch (README.md) would use.
MAX_RESPONSE_WORDS = 40
MAX_OVERLAP_WORDS = 8  # longest verbatim run allowed before redaction


def sanitize_response(
    response_text: str,
    context_texts: List[str],
    max_response_words: int = MAX_RESPONSE_WORDS,
    max_overlap_words: int = MAX_OVERLAP_WORDS,
) -> str:
    """
    Cap verbatim leakage of retrieved context into the returned response.
    Safe to call on every response -- a no-op unless the response is long or
    quotes a retrieved passage at length.
    """
    words = response_text.split()
    if len(words) > max_response_words:
        words = words[:max_response_words]

    for ctx in context_texts:
        ctx_words = ctx.split()
        match = SequenceMatcher(
            None,
            [w.lower() for w in words],
            [w.lower() for w in ctx_words],
            autojunk=False,
        ).find_longest_match(0, len(words), 0, len(ctx_words))
        if match.size > max_overlap_words:
            words = words[:match.a] + ["..."] + words[match.a + match.size:]

    return " ".join(words)


# Fixed, unconditionally-applied hedge prefix -- see obfuscate_decision() docstring
# for why it must be applied to EVERY response, not just ones that look confident.
DECISION_HEDGE_PREFIX = "Based on the available information, it is hard to say for certain, but "


def obfuscate_decision(response_text: str) -> str:
    """
    Defends against the `decision_match` leak channel (§12.2/§13 of
    reports/MIA_Security_Analysis_Report.md): the attack's strongest,
    most consistent signal checks whether the response's first few words
    state the correct yes/no/maybe judgment. This prepends a fixed hedging
    phrase to *every* response -- member and non-member alike -- pushing
    any yes/no/maybe commitment out of the leading-words window a
    positional decision-match detector inspects.

    Applied unconditionally, not conditionally on whether the response
    "looks confident": if it were only applied to confident-looking
    responses, that conditional application would itself become a new
    distinguishing signal (confident responses get hedged, hedged ones
    don't -- an attacker could just check for the presence of the hedge).

    Honesty note (important, do not remove): this defeats a *positional*
    check (first N words) specifically. It does not remove the correct
    decision from the response, and provides **no protection** against an
    attacker who scans the *entire* response text for a decision token
    instead of only its opening words -- see README.md for this scope
    limitation and why it is disclosed rather than implied away.
    """
    if not response_text.strip():
        return response_text
    return DECISION_HEDGE_PREFIX + response_text


# Pool sampled from at random (Revision 6), not cycled in a fixed order
# (Revision 5). Deliberately generic/uninformative content so padding never
# accidentally states or hints at the correct decision -- each word was
# checked against `_DECISION_SYNONYMS` (attack/Mia_attack/mia_attack.py) to
# confirm no overlap with yes/no/maybe or their synonyms.
LENGTH_FILLER_WORDS = [
    "additional", "clinical", "context", "and", "further", "review", "of",
    "the", "available", "evidence", "would", "be", "needed", "to", "fully",
    "confirm", "this", "specific", "determination", "in", "practice",
]


def calibrate_target_length(sample_responses: List[str], fallback: int = 150, floor: int = 100) -> int:
    """
    Revision 6: measures a `normalize_length()` target from the corpus's
    actual response lengths instead of guessing a round number. Revision 5's
    `target_chars=150` was chosen by inspection, not measurement (see
    reports/MIA_Security_Analysis_Report.md §13.3, risk 1) -- if member and
    non-member responses have systematically different natural lengths, an
    unmeasured fixed target could truncate one group more than the other,
    reintroducing the very length-based signal this defense exists to
    remove.

    `sample_responses` must be drawn from BOTH members and non-members
    together, not one group alone -- calibrating on only one group's typical
    length would bias the target toward that group and could *create* a
    length-based signal rather than remove one. Returns the median character
    length across the sample, or `fallback` if no non-empty sample is given.

    `floor` (Revision 6 addition, added after a live run exposed the failure
    mode): the deployed LLM was observed, during this project's own live
    evaluation, to sometimes return extremely terse responses (as short as
    10 characters, e.g. a bare "yes"/"no" with a leaked chat-template role
    token) alongside much longer ones for other queries -- an unrelated
    server-side formatting issue in drag_llm_service, not something this
    defense module can or should fix. Naively calibrating to the *median*
    of such a sample can collapse to a degenerate tiny target (measured:
    10 chars across all three tested seeds), at which point
    `normalize_length()` truncates nearly every substantive response down
    to a fragment -- destroying response content rather than neutralizing
    a length signal, and producing uninterpretable AUC swings in every
    other defended world as a side effect (measured: some "defended" worlds
    scored a *higher* AUC than undefended). `floor` guarantees the target
    never drops below a value that preserves enough content for a normal
    response, regardless of what a given sample's median happens to be --
    `max(measured_median, floor)`, not `measured_median` alone.
    """
    lengths = [len(r) for r in sample_responses if r and r.strip()]
    if not lengths:
        return fallback
    return max(int(statistics.median(lengths)), floor)


def normalize_length(
    response_text: str,
    target_chars: int = 150,
    _rng: Optional[random.Random] = None,
) -> str:
    """
    Defends against the `length_ratio` leak channel (measured the second-
    strongest individual signal in this project, AUC 0.59-0.65, and the only
    one of the four composite signals with no defense built against it prior
    to Revision 5).

    Pads or truncates to an exact **character** count -- not word count.
    `_answer_length_ratio()` computes `len(response)/len(gold_answer)` using
    Python's `len()` on the raw strings, i.e. character length, not word
    count (Revision 4 normalized to a fixed *word* count, which only
    approximately constrained character length and left residual signal --
    see the report's revision history).

    `target_chars` defaults to 150 for standalone callers, but production
    use should pass `calibrate_target_length()`'s measured value instead of
    relying on the default (see that function's docstring).

    Padding (Revision 6 change): each padding word is drawn independently at
    random from `LENGTH_FILLER_WORDS`, not cycled through in a fixed
    repeating sequence. Revision 5's fixed cycle always started at
    "additional" and repeated identically every time, which an attacker who
    noticed the pattern could strip as a known suffix before recomputing
    `length_ratio` -- fully defeating that version of the defense. Random
    per-call word choice removes that specific exploit, though it does not
    guarantee an attacker cannot statistically distinguish "generic padding"
    from organic text by other means (not tested here -- see report
    limitations). Pass `_rng` with a seeded `random.Random` for reproducible
    evaluation runs; production calls should omit it so padding differs
    across responses and can't be fingerprinted as a single fixed string.

    Should be applied *last*, after `sanitize_response()` and a decision
    defense, so the final returned length is the one actually fixed --
    applying it earlier and then appending more text would undo the
    normalization.
    """
    if len(response_text) >= target_chars:
        return response_text[:target_chars].rstrip()
    rng = _rng if _rng is not None else random
    pad_needed = target_chars - len(response_text)
    pieces: List[str] = []
    total = 0
    while total < pad_needed:
        piece = " " + rng.choice(LENGTH_FILLER_WORDS)
        pieces.append(piece)
        total += len(piece)
    return (response_text + "".join(pieces))[:target_chars]


# Vague, decision-neutral paraphrases the content-level defense substitutes
# for a detected decision token. Multiple options per category, chosen at
# random per call, so the substitution itself isn't a fixed fingerprintable
# string either.
#
# Every template below was checked against ALL THREE _DECISION_SYNONYMS sets
# (not just its own category) -- an earlier draft failed this check: the
# "yes" templates included "...being true" and the "maybe" templates included
# "...is mixed", both of which are themselves synonyms in _DECISION_SYNONYMS,
# so _decision_match_semantic() (which also matches synonyms, not just the
# literal word) still caught a residual signal even though the literal
# yes/no/maybe token had been removed -- measured as auc_roc_decision_match_semantic
# staying at 0.52-0.58 for the content-obfuscated world instead of dropping to
# 0.50 like the adaptive (literal-token-only) check did. Fixed by rewriting
# both templates to avoid every synonym in every category.
_VAGUE_REPLACEMENTS: Dict[str, List[str]] = {
    "yes": ["the evidence available points in that direction",
            "there is some support for this in what was reviewed"],
    "no": ["the evidence available does not point that way",
           "this does not appear to be well supported by what was reviewed"],
    "maybe": ["the available information on this does not settle the question",
              "this remains an open question based on what was reviewed"],
}


def _detect_decision_token(response_text: str) -> Optional[str]:
    """
    Scans `response_text` for a yes/no/maybe commitment (or a known synonym,
    via `_DECISION_SYNONYMS` imported from attack/Mia_attack/mia_attack.py)
    and returns the canonical category found, or None.

    Deliberately does NOT take the gold/ground-truth decision as an input --
    a real server-side deployment of this defense has no access to which
    document(s) were retrieved for a query or what the "correct" answer is
    supposed to be; it can only see the text it is about to return. Detecting
    the token from the response itself, the same way an attacker would, is
    what makes this usable as an actual production defense rather than an
    oracle-assisted simulation.
    """
    words = {w.strip(".,;:!?").lower() for w in response_text.split()}
    for canonical, synonyms in _DECISION_SYNONYMS.items():
        if words & synonyms:
            return canonical
    return None


def obfuscate_decision_content(response_text: str, _rng: Optional[random.Random] = None) -> str:
    """
    Content-level decision defense (Revision 6), targeting the gap
    `obfuscate_decision()` (Revision 4) was always disclosed not to close:
    that defense only relocates the decision token past a positional window,
    so an attacker who scans the whole response
    (`_decision_match_adaptive`/`_decision_match_semantic`) sees it
    unchanged -- confirmed empirically in Revision 5 (§12.7 of the report).

    This defense instead detects which canonical decision the response
    commits to (`_detect_decision_token`, using only the response text
    itself, never a gold label) and replaces the first matching token or
    synonym with a randomly-chosen vague paraphrase for that category. If no
    decision token is detected, the response is returned unchanged.

    Whether this actually reduces `_decision_match_semantic`'s AUC (as
    opposed to `obfuscate_decision()`, which measurably does not) is an
    empirical question answered by live evaluation, not assumed here -- see
    the report for the measured result before treating this as a proven fix.
    """
    detected = _detect_decision_token(response_text)
    if detected is None:
        return response_text
    rng = _rng if _rng is not None else random
    replacement = rng.choice(_VAGUE_REPLACEMENTS[detected])

    candidates = [detected] + sorted(_DECISION_SYNONYMS.get(detected, []))
    for token in candidates:
        pattern = re.compile(r"\b" + re.escape(token) + r"\b", re.IGNORECASE)
        new_text, n = pattern.subn(replacement, response_text, count=1)
        if n:
            return new_text
    return response_text


def _auc_or_half(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) > 1 and len(np.unique(scores)) > 1:
        return float(roc_auc_score(y_true, scores))
    return 0.5


def _compute_metrics(
    y_true:        np.ndarray,
    composite:     np.ndarray,
    sims:          np.ndarray,
    len_ratios:    np.ndarray,
    match_rates:   np.ndarray,
    certs:         np.ndarray,
    threshold_pct: int,
) -> Dict[str, Any]:
    """
    Mirrors attack/Mia_attack/mia_attack.py's _compute_metrics: `composite`
    (similarity*SIM_WEIGHT + certainty*CERTAINTY_WEIGHT + length_ratio*LEN_WEIGHT)
    drives AUC-ROC, threshold and classification (the same score the attacker
    actually uses); `sims`, `len_ratios`, `match_rates`, `certs` are reported
    alongside as diagnostics, exactly as the attack itself reports them, so
    undefended-vs-defended is an apples-to-apples comparison field for field.
    """
    threshold = float(np.percentile(composite, threshold_pct))
    y_pred = (composite >= threshold).astype(int)

    return {
        "auc_roc": round(_auc_or_half(y_true, composite), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "mean_member_similarity": round(float(np.mean(sims[y_true == 1])), 4) if np.any(y_true == 1) else 0.0,
        "mean_non_member_similarity": round(float(np.mean(sims[y_true == 0])), 4) if np.any(y_true == 0) else 0.0,
        "auc_roc_answer_match": round(_auc_or_half(y_true, match_rates), 4),
        "mean_member_answer_match_rate": round(float(np.mean(match_rates[y_true == 1])), 4) if np.any(y_true == 1) else 0.0,
        "mean_non_member_answer_match_rate": round(float(np.mean(match_rates[y_true == 0])), 4) if np.any(y_true == 0) else 0.0,
        "auc_roc_length_ratio": round(_auc_or_half(y_true, len_ratios), 4),
        "mean_member_length_ratio": round(float(np.mean(len_ratios[y_true == 1])), 4) if np.any(y_true == 1) else 0.0,
        "mean_non_member_length_ratio": round(float(np.mean(len_ratios[y_true == 0])), 4) if np.any(y_true == 0) else 0.0,
        "auc_roc_certainty": round(_auc_or_half(y_true, certs), 4),
        "mean_member_certainty": round(float(np.mean(certs[y_true == 1])), 4) if np.any(y_true == 1) else 0.0,
        "mean_non_member_certainty": round(float(np.mean(certs[y_true == 0])), 4) if np.any(y_true == 0) else 0.0,
    }


class MIADefenseEvaluator:
    """
    Runs the MIA probe set once against the live LLM service and scores both
    the raw (undefended) response and the sanitized (defended) response, so
    the two AUC-ROC figures are directly comparable on identical probes.
    """

    def __init__(
        self,
        llm_service_url: str = LLM_SERVICE_URL,
        corpus_jsonl: str = CORPUS_JSONL,
        n_members: int = DEFAULT_MEMBERS,
        n_nonmembers: int = DEFAULT_NONMEMBERS,
        probes_per_doc: int = 4,
        threshold_percentile: int = 50,
        api_key: str = "",
        random_seed: int = 42,
        max_response_words: int = MAX_RESPONSE_WORDS,
        max_overlap_words: int = MAX_OVERLAP_WORDS,
    ):
        if not _ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required.\nInstall: pip install sentence-transformers"
            )
        self.llm_service_url = llm_service_url
        self.corpus_jsonl = corpus_jsonl
        self.n_members = n_members
        self.n_nonmembers = n_nonmembers
        self.probes_per_doc = probes_per_doc
        self.threshold_pct = threshold_percentile
        self.api_key = api_key
        self.random_seed = random_seed
        self.max_response_words = max_response_words
        self.max_overlap_words = max_overlap_words
        print(f"  [MIA-Defense] Loading sentence-transformer: {EMBEDDING_MODEL}")
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)

    def run(self) -> Dict[str, Any]:
        members, non_members = load_membership_documents(
            self.n_members, self.n_nonmembers, self.random_seed,
            self.corpus_jsonl, self.probes_per_doc,
        )

        print(f"  [MIA-Defense] Member docs: {len(members)}   Non-member docs: {len(non_members)}")

        def _sanitize(raw: str, ctx: str) -> str:
            return sanitize_response(
                raw, [ctx],
                max_response_words=self.max_response_words,
                max_overlap_words=self.max_overlap_words,
            ) if raw.strip() else raw

        # Pass 1: query the LLM exactly once per probe and cache the raw
        # response. Needed before WORLDS can be defined, because the "full"/
        # "full_content" worlds' normalize_length() target is calibrated from
        # the actual observed response lengths (calibrate_target_length(),
        # Revision 6) rather than an unmeasured constant (Revision 5's 150) --
        # calibration requires seeing all raw responses first. Caching here
        # means the second pass re-processes already-fetched text only, no
        # extra network round-trips.
        all_records: List[Dict[str, Any]] = []
        for label, documents in (("MEMBER", members), ("NON-MEMBER", non_members)):
            print(f"\n  === Fetching {label} responses ===")
            for i, qa_list in enumerate(documents, 1):
                doc_records = []
                for qa in qa_list:
                    raw_response = _query_llm(qa["question"], self.llm_service_url, self.api_key)
                    doc_records.append({
                        "raw": raw_response,
                        "context": qa["context"],
                        "gold": qa["answer"],
                        "decision": qa.get("decision", ""),
                    })
                all_records.append({"label": label, "doc_idx": i, "n_docs": len(documents), "probes": doc_records})
                preview = qa_list[0]["question"][:40] if qa_list else ""
                print(f"    [{label}] doc {i:>2}/{len(documents)}  probes={len(qa_list)}  q0='{preview}'")

        target_chars = calibrate_target_length(
            [p["raw"] for rec in all_records for p in rec["probes"]]
        )
        print(f"\n  [MIA-Defense] Calibrated length-normalization target: {target_chars} chars "
              f"(median of {sum(len(rec['probes']) for rec in all_records)} observed responses, "
              "member+non-member combined -- see calibrate_target_length() docstring)")

        pad_rng = random.Random(self.random_seed)

        def _content_defended(raw: str, ctx: str) -> str:
            return obfuscate_decision_content(_sanitize(raw, ctx), _rng=pad_rng)

        # Six worlds. The first four match Revision 5 exactly (field names kept
        # for backward compatibility); "content_obfuscated"/"full_content" (NEW,
        # Revision 6) swap the positional obfuscate_decision() for the
        # content-level obfuscate_decision_content(), to test whether that
        # closes the gap _decision_match_adaptive/_decision_match_semantic
        # showed obfuscate_decision() leaves open (see reports/
        # MIA_Security_Analysis_Report.md §12.7).
        WORLDS: List[Tuple[str, Any]] = [
            ("undefended", lambda raw, ctx: raw),
            ("sanitized", lambda raw, ctx: _sanitize(raw, ctx)),
            ("obfuscated", lambda raw, ctx: obfuscate_decision(_sanitize(raw, ctx))),
            ("full", lambda raw, ctx: normalize_length(
                obfuscate_decision(_sanitize(raw, ctx)), target_chars=target_chars, _rng=pad_rng)),
            ("content_obfuscated", _content_defended),
            ("full_content", lambda raw, ctx: normalize_length(
                _content_defended(raw, ctx), target_chars=target_chars, _rng=pad_rng)),
        ]

        composite = {name: [] for name, _ in WORLDS}
        sim_diag = {name: [] for name, _ in WORLDS}
        len_diag = {name: [] for name, _ in WORLDS}
        match_diag = {name: [] for name, _ in WORLDS}
        cert_diag = {name: [] for name, _ in WORLDS}
        # Adaptive/semantic-attacker diagnostics (full-response scan; semantic
        # also accepts synonyms) -- not part of any composite score, evaluate
        # whether each defense's protection survives a more sophisticated
        # attacker. See _decision_match_adaptive, _decision_match_semantic.
        adapt_diag = {name: [] for name, _ in WORLDS}
        semantic_diag = {name: [] for name, _ in WORLDS}

        for rec in all_records:
            label, i, n_docs = rec["label"], rec["doc_idx"], rec["n_docs"]
            per_probe_sims = {name: [] for name, _ in WORLDS}
            per_probe_lens = {name: [] for name, _ in WORLDS}
            per_probe_matches = {name: [] for name, _ in WORLDS}
            per_probe_certs = {name: [] for name, _ in WORLDS}
            per_probe_adapts = {name: [] for name, _ in WORLDS}
            per_probe_semantic = {name: [] for name, _ in WORLDS}

            for p in rec["probes"]:
                raw_response, context, gold, decision = p["raw"], p["context"], p["gold"], p["decision"]
                emb_context = self._encoder.encode([context], convert_to_numpy=True)[0]

                for name, transform in WORLDS:
                    resp = transform(raw_response, context)
                    if resp.strip():
                        emb = self._encoder.encode([resp], convert_to_numpy=True)[0]
                        sim = _cosine_similarity(emb, emb_context)
                    else:
                        sim = 0.0
                    per_probe_sims[name].append(sim)
                    per_probe_lens[name].append(_answer_length_ratio(resp, gold))
                    per_probe_matches[name].append(1.0 if _decision_match(resp, decision) else 0.0)
                    per_probe_certs[name].append(_certainty_score(resp))
                    per_probe_adapts[name].append(1.0 if _decision_match_adaptive(resp, decision) else 0.0)
                    per_probe_semantic[name].append(_decision_match_semantic(resp, decision))

            # Each world's attacker only ever observes its own responses, so the
            # "best try across probes" is selected independently per world -- an
            # undefended attacker and a defended-world attacker can pick different
            # probes as their strongest, exactly mirroring the live attack's logic.
            for name, _ in WORLDS:
                sims, lens_, matches = per_probe_sims[name], per_probe_lens[name], per_probe_matches[name]
                certs, adapts, sems = per_probe_certs[name], per_probe_adapts[name], per_probe_semantic[name]
                best = int(np.argmax(sims)) if sims else 0
                sim = sims[best] if sims else 0.0
                length = lens_[best] if lens_ else 0.0
                cert = certs[best] if certs else 0.0
                match_rate = (sum(matches) / len(matches)) if matches else 0.0
                adapt_rate = (sum(adapts) / len(adapts)) if adapts else 0.0
                sem_rate = (sum(sems) / len(sems)) if sems else 0.0

                composite[name].append(
                    DECISION_WEIGHT * match_rate + SIM_WEIGHT * _normalize_similarity(sim)
                    + CERTAINTY_WEIGHT * cert + LEN_WEIGHT * length
                )
                sim_diag[name].append(sim)
                len_diag[name].append(length)
                match_diag[name].append(match_rate)
                cert_diag[name].append(cert)
                adapt_diag[name].append(adapt_rate)
                semantic_diag[name].append(sem_rate)

            scores_str = "  ".join(f"{name}={composite[name][-1]:.4f}" for name, _ in WORLDS)
            print(f"    [{label}] doc {i:>2}/{n_docs}  {scores_str}")

        y_true = np.array([1] * len(members) + [0] * len(non_members))
        metrics = {
            name: _compute_metrics(
                y_true, np.array(composite[name]), np.array(sim_diag[name]),
                np.array(len_diag[name]), np.array(match_diag[name]), np.array(cert_diag[name]),
                self.threshold_pct,
            )
            for name, _ in WORLDS
        }
        # Adaptive/semantic-attacker AUC per world -- not part of any composite
        # score, reported purely as a scope-limitation check.
        adaptive_aucs = {name: round(_auc_or_half(y_true, np.array(adapt_diag[name])), 4) for name, _ in WORLDS}
        semantic_aucs = {name: round(_auc_or_half(y_true, np.array(semantic_diag[name])), 4) for name, _ in WORLDS}

        result = {
            "n_members_tested": len(members),
            "n_non_members_tested": len(non_members),
            "calibrated_target_chars": target_chars,
            "score_weights": {"similarity": SIM_WEIGHT, "certainty": CERTAINTY_WEIGHT,
                              "length_ratio": LEN_WEIGHT, "decision_match": DECISION_WEIGHT},
            "undefended": metrics["undefended"],
            "defended": metrics["sanitized"],
            "decision_defended": metrics["obfuscated"],
            "fully_defended": metrics["full"],
            "content_decision_defended": metrics["content_obfuscated"],
            "fully_content_defended": metrics["full_content"],
            "auc_roc_reduction": round(metrics["undefended"]["auc_roc"] - metrics["sanitized"]["auc_roc"], 4),
            "auc_roc_reduction_decision_defended": round(metrics["undefended"]["auc_roc"] - metrics["obfuscated"]["auc_roc"], 4),
            "auc_roc_reduction_fully_defended": round(metrics["undefended"]["auc_roc"] - metrics["full"]["auc_roc"], 4),
            "auc_roc_reduction_content_decision_defended": round(metrics["undefended"]["auc_roc"] - metrics["content_obfuscated"]["auc_roc"], 4),
            "auc_roc_reduction_fully_content_defended": round(metrics["undefended"]["auc_roc"] - metrics["full_content"]["auc_roc"], 4),
            "auc_roc_decision_match_adaptive": {name: adaptive_aucs[name] for name, _ in WORLDS},
            "auc_roc_decision_match_semantic": {name: semantic_aucs[name] for name, _ in WORLDS},
        }
        self._print_summary(result)
        return result

    @staticmethod
    def _print_summary(r: Dict[str, Any]) -> None:
        print("\n" + "=" * 100)
        print("  MIA Defense — Undefended vs. 5 Defense Layers (Revision 6: + content-level decision defense)")
        print("=" * 100)
        w = r["score_weights"]
        print(f"  Gated composite: decision_match*{w['decision_match']} dominates; "
              f"similarity*{w['similarity']} + certainty*{w['certainty']} + "
              f"length_ratio*{w['length_ratio']} only re-rank within a gate tier")
        print(f"  Calibrated length-normalization target (Revision 6): {r['calibrated_target_chars']} chars "
              "(measured median of observed responses -- see calibrate_target_length())")
        u = r["undefended"]
        cols = [
            ("Undef.", u),
            ("Sanit.", r["defended"]),
            ("+Obfusc.", r["decision_defended"]),
            ("+LenNorm", r["fully_defended"]),
            ("+ContentObf.", r["content_decision_defended"]),
            ("+Content+Len", r["fully_content_defended"]),
        ]
        header = "  " + f"{'Metric':<28}" + "".join(f"{name:>14}" for name, _ in cols)
        print(header)
        for key in ("auc_roc", "accuracy", "precision", "recall", "f1_score",
                    "mean_member_similarity", "mean_non_member_similarity",
                    "auc_roc_answer_match", "mean_member_answer_match_rate",
                    "mean_non_member_answer_match_rate", "auc_roc_length_ratio",
                    "mean_member_length_ratio", "mean_non_member_length_ratio",
                    "auc_roc_certainty", "mean_member_certainty", "mean_non_member_certainty"):
            row = "  " + f"{key:<28}" + "".join(f"{m[key]:>14.4f}" for _, m in cols)
            print(row)
        print(f"\n  AUC-ROC reduction, sanitized only                    : {r['auc_roc_reduction']:+.4f}")
        print(f"  AUC-ROC reduction, sanitized+obfuscated              : {r['auc_roc_reduction_decision_defended']:+.4f}")
        print(f"  AUC-ROC reduction, sanitized+obfuscated+lennorm      : {r['auc_roc_reduction_fully_defended']:+.4f}")
        print(f"  AUC-ROC reduction, sanitized+content_obfuscated      : {r['auc_roc_reduction_content_decision_defended']:+.4f}")
        print(f"  AUC-ROC reduction, sanitized+content_obfuscated+len  : {r['auc_roc_reduction_fully_content_defended']:+.4f}")
        print(f"  (positive = defense reduced leakage; check distance-from-0.5 too if undefended AUC is near random)")
        a = r["auc_roc_decision_match_adaptive"]
        s = r["auc_roc_decision_match_semantic"]
        _labels = (
            ("undef", "undefended"), ("sanit", "sanitized"), ("+obfusc", "obfuscated"),
            ("+lennorm", "full"), ("+contentObf", "content_obfuscated"),
            ("+content+len", "full_content"),
        )
        print(f"\n  [adaptive attacker check -- scans WHOLE response, not just first 5 words]")
        print(f"  decision_match_adaptive AUC : " + "  ".join(f"{name}={a[key]:.4f}" for name, key in _labels))
        print(f"\n  [semantic attacker check -- full response + yes/no/maybe synonyms]")
        print(f"  decision_match_semantic AUC : " + "  ".join(f"{name}={s[key]:.4f}" for name, key in _labels))
        print(f"  (if +ContentObf./+Content+Len drop noticeably below undefended here, unlike +Obfusc./+LenNorm,")
        print(f"   the content-level defense is closing the gap the positional one was shown not to close)")
        print("=" * 100)
