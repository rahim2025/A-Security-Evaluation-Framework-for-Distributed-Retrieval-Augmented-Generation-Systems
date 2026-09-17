"""
attack/Mia_attack/mia_attack.py

Membership Inference Attack on Reliable-dRAG.

Why PubMedQA instead of SQuAD
------------------------------
An earlier version of this attack used SQuAD (Wikipedia-derived) as both the
corpus and the probe-question source, and measured composite-score AUC-ROC
*below* 0.50 across four independent seeds (mean 0.356) -- a systematic
inversion, not just "no signal". The most likely cause: the deployed LLM
(Qwen2.5-1.5B-Instruct) was almost certainly pretrained on Wikipedia, so it
can answer *non-member* SQuAD questions from memory nearly as well as
*member* ones, making the two groups look similar or even backwards. PubMedQA
(qiaojin/PubMedQA, `pqa_labeled` config) is a biomedical-research-abstract
QA dataset the base LLM is far less likely to have memorized verbatim, which
gives the retrieval-grounding signal a fairer test, decoupled from
pretraining-knowledge overlap.

Member/non-member split (same-domain, not cross-domain)
---------------------------------------------------------
`pqa_labeled` has exactly 1,000 rows in a single "train" split.
`data/build_pubmedqa_corpus.py` writes the first `PUBMEDQA_N_CORPUS` (500)
rows into `data/polluted_token/sources_0.jsonl` as the loaded corpus; the
remaining ~500 rows are never written anywhere and serve as a genuinely
disjoint, same-distribution held-out non-member pool -- deliberately NOT a
different dataset/domain (e.g. legal contracts), because a cross-domain
split would let a detector succeed merely by recognizing topic/domain rather
than genuine corpus membership, which would make any AUC lift ambiguous
evidence. This mirrors how the original SQuAD non-member bug was fixed
(non-members must be unseen-but-same-distribution, not a different corpus
entirely).

Caveat: PubMedQA has ~1 question per abstract (unlike SQuAD's ~4.5), so the
`probes_per_doc` multi-probe "best-of-N" mechanism effectively degrades to
best-of-1 for this dataset -- documented here rather than silently assumed
to still provide the same variance reduction it did for SQuAD.

The attack -- empirically re-tuned composite, 7th revision
------------------------------------------------------------
Revisions 2-4 hand-tuned a four-signal composite (decision_match, similarity,
certainty, length_ratio) by observing 3 seeds (0, 1, 42), settling on a
GATED formula where decision_match's weight (0.55) exceeded the other three
combined (0.45). Revisions 5-6 measured that this composite's held-out
generalization was NOT statistically distinguishable from chance (10-seed
held-out AUC 0.522, 95% CI [0.472, 0.572]) -- the weights had never been
validated against data withheld during the tuning process itself.

Revision 7 fixes this properly rather than diagnosing it again: a grid
search over the full weight simplex (`attack/Mia_attack/tune_weights.py
search`), maximizing mean per-seed AUC on 650 real per-document signal rows
pooled across all 13 seeds this project has ever observed (the "dev set" --
not held-out in any meaningful sense, since every one of their outcomes had
already been examined in Revisions 5-6). The winning weights were then
LOCKED and evaluated exactly once against 5 seeds generated fresh for this
purpose and never queried before that evaluation ran.

    membership_score = DECISION_WEIGHT * decision_match
                      + SIM_WEIGHT * normalized_similarity
                      + CERTAINTY_WEIGHT * certainty
                      + LEN_WEIGHT * length_ratio

Winning weights: `DECISION_WEIGHT=1.0, SIM_WEIGHT=0.0, CERTAINTY_WEIGHT=0.0,
LEN_WEIGHT=0.0` -- the composite is now, empirically, just `decision_match`
alone. The grid search found that `similarity`, `certainty`, and (contrary
to Revision 6's ablation diagnostic, which only tested dropping
similarity/certainty) `length_ratio` all add net noise rather than net
signal on the dev data -- not just the two signals previously suspected.

Fresh-seed result (the number that actually matters): mean AUC **0.620**,
95% CI **[0.546, 0.695]** -- excludes chance, and is *higher* than the
dev-set search score (0.572), indicating the search did not overfit to the
13 dev seeds. This is the first revision where the production composite's
generalization to genuinely unseen data is both measured and confirmed,
rather than assumed, hoped for, or found wanting.

`similarity`, `certainty`, and `length_ratio` remain computed and reported
as diagnostics (unchanged functions below) -- they are not deleted from the
codebase, only zeroed out of the score that decides membership, in case a
future revision with more dev data finds a role for them again.

  1. decision_match (1.0): fraction of probes where the response's first few
     words include the correct gold `final_decision` (yes/no/maybe) token --
     measured the strongest, most consistent signal in this project across
     all seven revisions (AUC 0.50-0.66 across all 13 dev seeds, never
     inverted; held-out mean 0.568, 95% CI excluding chance, Revision 6 §2.5).
  2. similarity (0.0, diagnostic only): cosine similarity, response vs. true context, remapped to [0,1].
  3. certainty (0.0, diagnostic only): yes/no/maybe-commitment vs. hedge detection (see `_certainty_score`).
  4. length_ratio (0.0, diagnostic only): min(len(response)/len(gold_answer), 1.0).

Revision 10 -- pretraining-knowledge calibration and rate-limit hardening
--------------------------------------------------------------------------
Two correctness problems, both raised directly by the thesis team and both
traced to concrete evidence rather than assumed (see
reports/updated_reports_safin/MIA_SCORE_MECHANISM_FIX.md for the full
writeup, and the source material: `RAGLeak: Membership Inference Attacks on
RAG-Based Large Language Models` (Feng et al., ACISP 2025), a paper on
membership-inference against RAG systems, generally).

(a) Pretraining-knowledge confound, previously undiagnosed by the score
    itself. `decision_match` (below) is a coarse 3-way categorical judgment
    (yes/no/maybe). PubMedQA's answers skew toward "yes" (~55%), and the
    base LLM has some non-zero chance of already knowing a given biomedical
    fact from pretraining -- both mean a *non-member* document's response
    can commit to the correct decision token by chance or by prior
    knowledge, with nothing in the RAG pipeline having grounded it. Before
    this revision, that was visible only as a footnote in the printed
    diagnostic ("high here means the LLM already knew the fact from
    pretraining", line ~856 in prior revisions) -- the composite score
    itself did nothing to detect or correct for it, so if pretraining
    knowledge inflated the non-member match rate enough, the member/
    non-member gap the whole attack depends on could shrink toward (or, in
    a bad enough draw, collapse to) chance, with no signal in the output
    that this had happened. RAGLeak's own methodology (paper §5.3,
    "Exclude LLM Training Data") diagnoses exactly this failure mode: it
    validates that a *cropped-continuation* probe design (Eq. 2-3 of the
    paper) keeps the before-RAG member/non-member similarity distributions
    overlapping (i.e. pretraining alone carries no signal), specifically
    because a coarse categorical target like a yes/no/maybe judgment would
    not have that property. This module cannot switch PubMedQA's ground
    truth to a continuation-style probe without changing the dataset
    (out of scope here), so instead of relying on dataset design alone to
    avoid the confound, Revision 10 measures and calibrates against it
    directly, every run: drag_llm_service's `/query` endpoint now accepts
    an additive `no_retrieval: true` request field (skips retrieval/
    reranking entirely, answers from an empty context through the exact
    same prompt template and model) -- see
    `drag_llm_service/app/server.py`. For each probe this module now also
    queries that no-RAG path and computes `_decision_match()` against it,
    giving a genuine "what would the LLM have said anyway" baseline through
    the identical model/prompt path, not a separately-reasoned guess.
    `_calibrated_decision_score()` combines the two: a RAG-grounded match
    that the no-RAG baseline does NOT also produce keeps full credit (1.0,
    clean evidence of retrieval-grounding); a RAG-grounded match that the
    baseline ALSO produces is now scored as ambiguous (0.5, since it could
    be genuine grounding or the model already knowing the fact) instead of
    silently counted as full-strength member evidence; no RAG-grounded
    match stays 0.0 either way. This calibrated score, not the raw
    `decision_match`, now drives the primary composite -- the raw,
    uncalibrated `decision_match` is kept and reported unchanged as
    `auc_roc_answer_match` for continuity with prior revisions' reports.
    The no-RAG baseline's own match rate is reported separately
    (`mean_*_pretraining_baseline_match`, `auc_roc_pretraining_baseline`) --
    this is the module's own live version of the paper's Fig. 3 "before
    RAG vs after RAG" validation, now measured on every run instead of a
    one-off historical check. **Disclosed, not yet re-validated**: Revision
    7's weights (`DECISION_WEIGHT=1.0` etc.) were grid-searched against the
    *uncalibrated* `decision_match`; calibration changes what that signal
    measures, so a fresh grid search against the calibrated signal is
    recommended follow-up work, not assumed equivalent, before citing a new
    AUC number as validated the way Revision 7's was.

(b) Rate-limit-induced retrieval corruption, a second, structurally
    distinct problem with the same symptom (a document that IS a member
    looking like a non-member). `drag_data_source`'s three containers each
    enforce a 60-requests/minute-per-IP cap by default
    (`drag_data_source/app/server.py`, `RATE_LIMIT_DEFAULT`); every single
    `/query` call to `drag_llm_service` fans out server-side to all three
    data sources (`query_data_sources()`) -- one HTTP request to *each*
    distinct source per external call (not three requests piled onto one
    source), so each source's own rate counter tracks this module's
    external call rate directly. Measured live during a 100+100-sample MIA
    run with no client-side pacing at all: 64% of data-source requests were
    rejected with HTTP 429 over a 30-minute window (see
    `problems/safin_faced_problems/` and `problems/mia_attack_gaps.md`) --
    this module, unlike its sibling `attack/selective_forward_sim` and
    `attack/ssm_score/run_attack.py`, had no pacing whatsoever between
    calls. Before this revision, a 429/5xx response from any data source
    was silently treated identically to "this source has nothing relevant"
    by `drag_llm_service` -- when a member document's real content lived on
    the rate-limited source, the model answered without it, and that
    document's response looked exactly like a genuine non-member's. This
    directly corrupts the membership signal regardless of pretraining, and
    (a) and (b) can compound: a rate-limited member document forced to
    answer blind is now *also* exactly the scenario (a) calibrates for, so
    without both fixes a rate-limited member could be scored as a
    confident, uncalibrated non-member miss. Fixed at the source
    (`drag_llm_service/app/server.py`'s new `_post_data_source_with_retry()`:
    retry-with-backoff on 429/5xx, honoring `Retry-After`) and at this
    module's client (`_query_llm()` below is now self-throttled to
    `MIN_QUERY_INTERVAL_S`, matching the pacing already validated live for
    this exact rate limit by `attack/selective_forward_sim/live_network.py`
    (1.1s) and `attack/ssm_score/run_attack.py` (1.3s) -- this module's own
    prior total absence of any pacing, not an unusually tight cap, is the
    confirmed root cause of the measured 64% failure rate). `/query`'s
    response now additionally reports `sources_used`/`degraded` so this
    module can tell a fully-grounded answer apart from a degraded one
    instead of treating them identically; degraded probes are now counted
    and reported (`degraded_probe_rate`) rather than silently folded into
    the same signal as clean probes.

Hypothesis
----------
For a *member* document the retriever can surface the real context, so its
best-try answer should ground in and resemble that passage, with a
confident, direct commitment to the correct yes/no/maybe judgment. For a
*non-member* document, no source holds that passage, so an honest answer
should fail to commit correctly -- unless the base LLM already knows the
fact from pretraining, in which case decision_match may be high for
non-members too, capping the available signal regardless of scoring
strategy. Revision 10's calibration (above) is a direct, measured response
to this "unless" clause, rather than leaving it as an unaddressed caveat.
AUC-ROC ~= 0.50 -> no privacy leakage (attack fails).
AUC-ROC > 0.70  -> genuine privacy vulnerability.

Prerequisites
-------------
  pip install sentence-transformers scikit-learn numpy requests datasets
  python data/build_pubmedqa_corpus.py     (regenerates sources_0.jsonl from PubMedQA)
  docker compose up -d          (in drag_data_source/ and drag_llm_service/)
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# ── Default endpoints (override with env vars or constructor args) ──────────
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:9000")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

# Used only to determine which SQuAD contexts are actually loaded into the
# running corpus -- all three sources share the same document indices, so
# sources_0.jsonl's context set is a system-wide "is this loaded" oracle.
CORPUS_JSONL = os.path.normpath(
    os.path.join(_ROOT, "data", "polluted_token", "sources_0.jsonl")
)
# Kept as an alias for backward compatibility with anything importing the
# old name.
MEMBER_JSONL = CORPUS_JSONL

PUBMEDQA_DATASET = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"
PUBMEDQA_SPLIT = "train"
# Must match data/build_pubmedqa_corpus.py's --n_corpus default exactly --
# rows [0, PUBMEDQA_N_CORPUS) are written to sources_0.jsonl (members);
# rows [PUBMEDQA_N_CORPUS, 1000) are the held-out non-member pool.
PUBMEDQA_N_CORPUS = 500

EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
DEFAULT_MEMBERS    = 25
DEFAULT_NONMEMBERS = 25

# Composite membership score, Revision 7 -- empirically re-tuned under a genuine
# train/test split (see reports/MIA_Security_Analysis_Report.md §2.9 for full
# methodology and reports/../attack_logs/tune_weights_*.json for raw results):
#   DECISION_WEIGHT * decision_match
#   + SIM_WEIGHT * normalized_similarity + CERTAINTY_WEIGHT * certainty + LEN_WEIGHT * length_ratio
#
# History of re-calibration:
#   1st revision (linear, DRAG-inspired prior): SIM=0.85, LEN=0.15.
#   2nd revision (linear, empirically re-weighted twice): SIM=0.20, CERTAINTY=0.10,
#     LEN=0.30, DECISION=0.40.
#   3rd revision (GATED, not linearly blended): DECISION=0.55 set to exceed
#     SIM+CERTAINTY+LEN (0.45) combined, so decision_match=1 always outranks
#     decision_match=0 regardless of the other signals. These weights (through
#     Revision 6) were chosen by human judgment and pattern-matching against 3
#     tuning seeds (0, 1, 42), never validated against data withheld during the
#     choice -- Revisions 5-6 both measured that the resulting composite's
#     held-out generalization was NOT statistically distinguishable from chance
#     (10-seed held-out AUC 0.522, 95% CI [0.472, 0.572], includes 0.50).
#   7th revision (this one, EMPIRICALLY RE-TUNED): weights chosen by a grid
#     search over the 4-weight simplex, maximizing mean per-seed AUC on 650
#     real per-document signal rows across all 13 seeds this project has ever
#     observed (`attack/Mia_attack/tune_weights.py search`) -- then evaluated
#     EXACTLY ONCE, with no further changes, against 5 seeds generated fresh
#     and never queried before that evaluation (865, 659, 693, 783, 154).
#     Winner: DECISION_WEIGHT=1.0, everything else=0.0 -- similarity, certainty,
#     AND length_ratio all measured to add net noise rather than net signal on
#     the dev data, not just similarity/certainty as Revision 6's ablation
#     diagnostic suggested. Fresh-seed result: mean AUC 0.620, 95% CI [0.546,
#     0.695] -- EXCLUDES chance, and is *higher* than the dev-set score (0.572),
#     indicating no overfitting to the search. This is the first revision in
#     this project's history where the production composite's generalization
#     to genuinely unseen data is both measured and statistically confirmed.
DECISION_WEIGHT  = 1.00
SIM_WEIGHT       = 0.00
CERTAINTY_WEIGHT = 0.00
LEN_WEIGHT       = 0.00
assert DECISION_WEIGHT > SIM_WEIGHT + CERTAINTY_WEIGHT + LEN_WEIGHT, (
    "DECISION_WEIGHT must strictly exceed the other three combined for the gate "
    "to be unconditional -- see module docstring."
)

# Ablation weights (Revision 6, diagnostic only -- NOT the production composite,
# and NOT imported by defense/mia_defense/mia_defense.py). Tests the hypothesis
# that SIM_WEIGHT and CERTAINTY_WEIGHT are adding noise rather than signal on
# held-out seeds (Revision 5, reports/MIA_Security_Analysis_Report.md §12.10):
# certainty was never shown to help in any revision, and similarity is weaker
# and noisier than decision_match/length_ratio. Reported alongside the primary
# composite's AUC as a diagnostic comparison -- does NOT replace DECISION_WEIGHT
# etc. above, since changing the production weights based on the same held-out
# seeds used to evaluate them would repeat the exact overfitting mistake this
# ablation is meant to detect.
ABLATION_DECISION_WEIGHT  = 0.70
ABLATION_SIM_WEIGHT       = 0.00
ABLATION_CERTAINTY_WEIGHT = 0.00
ABLATION_LEN_WEIGHT       = 0.30
assert abs((ABLATION_DECISION_WEIGHT + ABLATION_SIM_WEIGHT
            + ABLATION_CERTAINTY_WEIGHT + ABLATION_LEN_WEIGHT) - 1.0) < 1e-9

HEDGE_WORDS = frozenset({
    "may", "might", "could", "possibly", "generally", "often",
    "typically", "usually", "perhaps", "probably", "likely", "seem",
    "seems", "appear", "appears", "suggest", "suggests", "unclear",
    "uncertain",
})

# PubMedQA answers are fundamentally yes/no/maybe judgments (see `final_decision`
# field). A response that commits to one of these near its start is a more direct,
# dataset-appropriate signal of "answering with confidence" than generic hedge-word
# absence (which measured as degenerate -- see CERTAINTY_WEIGHT comment above).
DECISIVE_TOKENS = frozenset({"yes", "no", "maybe"})


# ── I/O helpers ─────────────────────────────────────────────────────────────

def _load_corpus_contexts(path: str = CORPUS_JSONL) -> set:
    """Return the set of context passages actually loaded into the corpus."""
    contexts = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            html = rec.get("html", "")
            if html:
                contexts.add(html)
    return contexts


def _join_pubmedqa_context(contexts) -> str:
    """
    Must match data/build_pubmedqa_corpus.py's join_context() exactly --
    membership is determined by string equality against sources_0.jsonl's
    "html" field, which was written using this same join logic.
    """
    return " ".join(c.strip() for c in contexts if c and c.strip())


def load_membership_documents(
    n_members: int,
    n_nonmembers: int,
    seed: int,
    corpus_jsonl: str = CORPUS_JSONL,
    probes_per_doc: int = 4,
) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]]]:
    """
    Return (members, non_members), each a list of *documents*, where each
    document is itself a list of up to `probes_per_doc`
    {"question", "context", "answer"} dicts. For PubMedQA this list is
    almost always length 1 (~1 question per abstract, unlike SQuAD's ~4.5),
    so multi-probing degrades to best-of-1 for this dataset -- see module
    docstring.

    members     : context IS one of the documents loaded into the corpus
                  (i.e. one of the first PUBMEDQA_N_CORPUS pqa_labeled rows,
                  written to sources_0.jsonl by build_pubmedqa_corpus.py).
    non_members : context is NOT loaded into any source -- the remaining
                  pqa_labeled rows, genuinely unseen but same dataset/
                  distribution (not a different domain/corpus).

    Sampling is seeded with `random.Random(seed)`, so different seeds
    produce genuinely different documents.
    """
    from datasets import load_dataset  # local import: heavy, only needed here

    print(f"  [MIA] Loading {PUBMEDQA_DATASET} ({PUBMEDQA_CONFIG}, {PUBMEDQA_SPLIT} split) ...")
    ds = load_dataset(PUBMEDQA_DATASET, PUBMEDQA_CONFIG, split=PUBMEDQA_SPLIT)
    corpus_contexts = _load_corpus_contexts(corpus_jsonl)
    print(f"  [MIA] {len(corpus_contexts)} contexts loaded into the running corpus")

    member_docs: Dict[str, List[Dict[str, str]]] = {}
    nonmember_docs: Dict[str, List[Dict[str, str]]] = {}
    seen_questions = set()

    for item in ds:
        q = item["question"].strip()
        if not q or q in seen_questions:
            continue
        long_answer = (item.get("long_answer") or "").strip()
        if not long_answer:
            continue
        context = _join_pubmedqa_context(item["context"]["contexts"])
        if not context:
            continue
        decision = (item.get("final_decision") or "").strip().lower()
        rec = {"question": q, "context": context, "answer": long_answer, "decision": decision}
        seen_questions.add(q)
        bucket = member_docs if context in corpus_contexts else nonmember_docs
        bucket.setdefault(context, []).append(rec)

    if not member_docs:
        raise RuntimeError(
            f"No PubMedQA questions matched contexts loaded in {corpus_jsonl}. "
            "Run `python data/build_pubmedqa_corpus.py` to (re)generate the corpus "
            "from PubMedQA before running this attack."
        )
    if not nonmember_docs:
        raise RuntimeError(
            "No genuinely unseen PubMedQA contexts found -- every context in "
            f"{PUBMEDQA_CONFIG} appears to be loaded into the corpus. Check "
            "PUBMEDQA_N_CORPUS matches build_pubmedqa_corpus.py's --n_corpus."
        )

    rng = random.Random(seed)
    member_contexts = rng.sample(list(member_docs), min(n_members, len(member_docs)))
    nonmember_contexts = rng.sample(list(nonmember_docs), min(n_nonmembers, len(nonmember_docs)))

    members = [member_docs[c][:probes_per_doc] for c in member_contexts]
    non_members = [nonmember_docs[c][:probes_per_doc] for c in nonmember_contexts]

    n_probes_m = sum(len(d) for d in members)
    n_probes_nm = sum(len(d) for d in non_members)
    print(f"  [MIA] Member documents: {len(member_docs)}  Non-member documents: {len(nonmember_docs)}")
    print(f"  [MIA] Sampled {len(members)} member docs ({n_probes_m} probes), "
          f"{len(non_members)} non-member docs ({n_probes_nm} probes) (seed={seed})")
    return members, non_members


def _parse_llm_response(data: Any) -> str:
    """
    Extract the answer string from whatever the LLM service returns.
    Tries a wide range of field names so the attack works regardless of
    which version of drag_llm_service the user is running.
    """
    if isinstance(data, str):
        return data.strip()
    if not isinstance(data, dict):
        return ""
    for key in ("response", "answer", "text", "output", "generated_text",
                "result", "content", "message", "completion", "generation"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            for sub_key in ("content", "text", "answer"):
                sv = val.get(sub_key)
                if isinstance(sv, str) and sv.strip():
                    return sv.strip()
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message") or first.get("text") or {}
            if isinstance(msg, str):
                return msg.strip()
            if isinstance(msg, dict):
                return str(msg.get("content", "")).strip()
    return ""


# Minimum seconds between calls to the LLM service (Revision 10, rate-limit
# hardening -- see module docstring §(b)). Every /query call fans out
# server-side to all 3 drag_data_source containers -- one HTTP request to
# EACH distinct source per external call, so each source's own 60-requests/
# minute-per-IP cap (RATE_LIMIT_DEFAULT) tracks this module's external call
# rate directly, not a multiple of it. MIN_QUERY_INTERVAL_S=1.3s matches the
# pacing already validated live against this exact limit by this project's
# other live-mode attack clients: attack/selective_forward_sim/
# live_network.py (1.1s) and attack/ssm_score/run_attack.py's QUERY_DELAY
# (1.3s, same family of attack as this one -- see that file's identical
# rate-limit comment). This module previously had NO pacing at all, unlike
# both of those -- that gap, not the cap itself, is the confirmed root
# cause of a measured 64% HTTP 429 rate during a real 100+100-sample run
# (problems/safin_faced_problems/). Overridable via MIA_MIN_QUERY_INTERVAL_S
# for a test deployment that has raised RATE_LIMIT_DEFAULT (see
# drag_data_source/app/server.py's comment on that env var).
MIN_QUERY_INTERVAL_S = float(os.getenv("MIA_MIN_QUERY_INTERVAL_S", "1.3"))
_last_query_time = 0.0


def _throttle_query() -> None:
    """Self-throttle so this module never exceeds MIN_QUERY_INTERVAL_S
    between calls to the LLM service, regardless of how tight the caller's
    own loop is (see module docstring §(b): the previous absence of any
    pacing here, unlike the SFA module's sibling code, is the confirmed
    root cause of a measured 64% HTTP 429 rate during a real 100+100-sample
    run)."""
    global _last_query_time
    elapsed = time.monotonic() - _last_query_time
    wait = MIN_QUERY_INTERVAL_S - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_query_time = time.monotonic()


def _query_llm_raw(
    question: str, url: str, api_key: str = "", no_retrieval: bool = False, max_retries: int = 2,
) -> Tuple[str, bool]:
    """
    Query the LLM service, self-throttled (see MIN_QUERY_INTERVAL_S) and
    retrying once on HTTP 429 (honoring the server's Retry-After header),
    mirroring the pattern already validated live in
    attack/selective_forward_sim/live_network.py's LivePeer.query().

    `no_retrieval=True` uses the same /query endpoint's no-RAG bypass
    (drag_llm_service/app/server.py) -- the model answers from an empty
    context, through the identical prompt template. Used by
    `_probe_documents()` to measure the pretraining-knowledge calibration
    baseline (see module docstring §(a)). This path never reaches the data
    sources, so it does not consume any of the 60/min-per-source budget,
    but is still throttled here to avoid hammering the LLM service itself.

    Returns (response_text, degraded), where `degraded` is True if the
    server reports at least one sampled data source did not return "ok"
    (rate-limited, errored, or empty) for a retrieval-mode query -- always
    False for no_retrieval=True queries, since no data source was
    contacted. `response_text` is `""` on any failure (network error,
    non-200 after retries, or an empty/blank generated answer) -- callers
    already treat an empty string as "no usable response" exactly as
    before this revision.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload: Dict[str, Any] = {"query": question}
    if no_retrieval:
        payload["no_retrieval"] = True

    for attempt in range(max_retries + 1):
        _throttle_query()
        try:
            r = requests.post(f"{url}/query", json=payload, headers=headers, timeout=120)
        except Exception:
            return "", False

        if r.status_code == 429 and attempt < max_retries:
            retry_after = float(r.headers.get("Retry-After", 2.0))
            time.sleep(retry_after)
            continue

        if r.status_code == 200:
            data = r.json()
            return _parse_llm_response(data), bool(data.get("degraded", False))

        return "", False

    return "", False


def _query_llm(question: str, url: str, api_key: str = "") -> str:
    """
    Backward-compatible string-only wrapper around `_query_llm_raw()` --
    unchanged signature and return type, since this function is imported
    directly by several other modules (`defense/mia_defense/mia_defense.py`,
    `attack/Mia_attack/run_ablation_eval.py`,
    `defense/mia_defense/validate_length_floor.py`). Those callers now get
    the Revision 10 throttling/retry hardening transparently, with no code
    changes required on their side.
    """
    text, _degraded = _query_llm_raw(question, url, api_key)
    return text


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _normalize_similarity(sim: float) -> float:
    """
    Remap raw cosine similarity from [-1, 1] to [0, 1]. Required so the
    gated composite's DECISION_WEIGHT > SIM_WEIGHT+CERTAINTY_WEIGHT+LEN_WEIGHT
    inequality actually holds -- an unbounded-below signal could otherwise
    (in principle) push a secondary-signal contribution outside its assumed
    [0, LEN_WEIGHT+CERTAINTY_WEIGHT+SIM_WEIGHT] range and violate the gate.
    """
    return (sim + 1.0) / 2.0


def _answer_length_ratio(response: str, gold_answer: str) -> float:
    """min(len(response)/len(gold_answer), 1.0) -- capped so a verbose non-member
    answer can't outscore a concise member answer on length alone."""
    gold_len = len(gold_answer.strip())
    if gold_len == 0 or not response.strip():
        return 0.0
    return min(len(response.strip()) / gold_len, 1.0)


def _certainty_score(response: str) -> float:
    """
    Redesigned (2nd revision) around PubMedQA's actual answer format.

    The original version was `1 - (hedge-word fraction)` -- measured
    completely degenerate on PubMedQA (constant 1.0 for every single
    response, member and non-member alike, AUC exactly 0.5000 in 2/3 live
    runs -- see reports/MIA_Security_Analysis_Report.md), because PubMedQA's
    terse yes/no/maybe answers essentially never contain a generic hedge
    word from a list built for free-text SQuAD-style answers.

    PubMedQA questions are fundamentally yes/no/maybe judgments. A grounded,
    retrieval-backed answer should be able to commit to one of those tokens
    directly; a guessed/hallucinated answer is more likely to hedge instead
    of committing. This version scores:
      1.0  -- response opens with an explicit yes/no/maybe commitment
      0.5  -- response hedges instead (contains a hedge word, no commitment)
      0.75 -- neither hedges nor commits (ambiguous middle ground)
      0.0  -- empty response

    Still a same-shape stand-in for the original DRAG system's hop-count
    term (`1 - normalized_hops`), not a literal substitute -- see module
    docstring.
    """
    words = [w.strip(".,;:!?").lower() for w in response.split()]
    if not words:
        return 0.0
    lead = words[:3]
    if any(w in DECISIVE_TOKENS for w in lead):
        return 1.0
    if any(w in HEDGE_WORDS for w in words):
        return 0.5
    return 0.75


def _decision_match(response: str, gold_decision: str) -> bool:
    """
    Does the response's leading commitment token match the gold yes/no/maybe
    decision? Replaces a full-sentence substring check against `long_answer`
    (PubMedQA's gold field is a full explanatory sentence that essentially
    never appears verbatim in a generated response -- measured 0.0000 match
    rate for both members and non-members in every live run, making that
    diagnostic blind). Checking for the correct decision token within the
    response's first few words is a much more targeted, dataset-appropriate
    signal than either a full-sentence substring check or a substring check
    anywhere in the response (which "yes"/"no" could trivially match by
    accident, e.g. inside "no evidence was found").
    """
    gold = gold_decision.strip().lower()
    if not gold:
        return False
    words = [w.strip(".,;:!?").lower() for w in response.split()[:5]]
    return gold in words


def _calibrated_decision_score(rag_match: bool, baseline_match: bool) -> float:
    """
    Revision 10 (see module docstring §(a)). Nets `_decision_match()` out
    against a no-RAG pretraining-knowledge baseline (the same question,
    same model, same prompt template, but with `no_retrieval=True` --
    i.e. what the model would answer with nothing retrieved at all), so a
    non-member (or a rate-limited member) that happens to guess or already
    know the correct yes/no/maybe decision doesn't get counted as
    full-strength evidence of retrieval-grounded membership.

        rag_match=True,  baseline_match=False -> 1.0  clean evidence: RAG
                                                        context changed the
                                                        answer to the
                                                        correct one.
        rag_match=True,  baseline_match=True  -> 0.5  ambiguous: correct
                                                        either way, so this
                                                        probe cannot tell
                                                        genuine grounding
                                                        from prior knowledge.
        rag_match=False, baseline_match=*     -> 0.0  no committed-correct
                                                        answer under RAG,
                                                        regardless of the
                                                        baseline.

    This is the value fed into the primary composite starting Revision 10
    (see DECISION_WEIGHT usage in `_probe_documents()`); the uncalibrated
    `_decision_match()` result is still computed and reported separately
    (`auc_roc_answer_match`) for continuity with prior revisions' reports.
    """
    if not rag_match:
        return 0.0
    return 0.5 if baseline_match else 1.0


def _decision_match_adaptive(response: str, gold_decision: str) -> bool:
    """
    Adaptive-attacker variant of `_decision_match()`: scans the *entire*
    response for the gold yes/no/maybe token, not just its first 5 words.

    Not used in the primary attack composite -- diagnostic-only, built
    specifically to test whether `defense/mia_defense/mia_defense.py`'s
    `obfuscate_decision()` (which defeats the *positional* check
    `_decision_match()` performs, by pushing the commitment token past the
    first 5 words) also defeats a more sophisticated attacker who doesn't
    restrict their search to the response's opening words. If it doesn't
    (expected, since obfuscation never removes the token, only relocates
    it), this function's AUC on obfuscated responses should closely track
    the *undefended* `_decision_match` AUC -- confirming the defense's
    honestly-disclosed scope limitation empirically rather than just in
    prose. See reports/MIA_Security_Analysis_Report.md for the measured
    result.
    """
    gold = gold_decision.strip().lower()
    if not gold:
        return False
    words = [w.strip(".,;:!?").lower() for w in response.split()]
    return gold in words


# Synonym vocabulary for `_decision_match_semantic` (Revision 6). Each set was
# checked against `normalize_length()`'s LENGTH_FILLER_WORDS (defense/mia_defense/
# mia_defense.py) to confirm no overlap -- an overlap would mean the length
# defense's padding accidentally counts as a semantic decision match.
_DECISION_SYNONYMS: Dict[str, frozenset] = {
    "yes":   frozenset({"yes", "correct", "true", "indeed", "affirmative", "confirmed"}),
    "no":    frozenset({"no", "incorrect", "false", "negative", "unlikely", "refuted"}),
    "maybe": frozenset({"maybe", "possibly", "unclear", "uncertain", "inconclusive", "mixed"}),
}


def _decision_match_semantic(response: str, gold_decision: str) -> float:
    """
    Combines two extensions to `_decision_match()`, proposed and tested in
    Revision 6 (reports/MIA_Security_Analysis_Report.md) as a stronger
    candidate attacker: (a) scans the *entire* response like
    `_decision_match_adaptive()`, not just the first 5 words, and (b) also
    accepts a small set of synonyms for yes/no/maybe (see `_DECISION_SYNONYMS`),
    so a defense or a model phrasing that avoids the literal gold token (e.g.
    "correct" instead of "yes") is still caught.

    Returns 1.0 for an exact gold-token match anywhere in the response, 0.5
    for a synonym-only match, 0.0 otherwise -- diagnostic-only, like
    `_decision_match_adaptive`, not folded into the production composite
    (see ABLATION_* weights above for why: any composite change must be
    validated on seeds the change wasn't tuned against, not swapped in based
    on the same run that motivated it).
    """
    gold = gold_decision.strip().lower()
    if not gold:
        return 0.0
    words = {w.strip(".,;:!?").lower() for w in response.split()}
    if gold in words:
        return 1.0
    if words & _DECISION_SYNONYMS.get(gold, frozenset()):
        return 0.5
    return 0.0


def _consistency_score(
    question: str,
    gold_decision: str,
    llm_service_url: str,
    api_key: str = "",
    n_probes: int = 5,
) -> float:
    """
    Multi-probe consistency signal (Revision 6, new hypothesis): queries the
    *same* question `n_probes` times and measures what fraction of responses
    agree with the majority `_decision_match_semantic` reading (1.0 exact
    match / 0.5 synonym / 0.0 no match, rounded to the nearest of those three
    buckets for majority-voting purposes).

    Rationale (untested before Revision 6): a member document has a real
    passage the retriever can consistently surface, so repeated queries
    should keep landing on the same yes/no/maybe judgment. A non-member has
    no backing passage, so the LLM is answering from general
    knowledge/guessing each time, which should vary more across repeated
    identical queries. This is independent of any single response's content,
    so `obfuscate_decision()`, `sanitize_response()`, and `normalize_length()`
    -- all of which only transform individual responses -- have no
    documented mechanism to defend against it. Whether that expectation
    holds was tested empirically, not assumed; see the report for the
    measured result before treating this as a proven attack vector.

    Costs `n_probes` additional live LLM calls per document -- not wired
    into the default per-document probing loop, since PubMedQA's ~1
    question/document would make this `n_probes` times more expensive across
    every document in every run. Intended to be run as a separate, smaller
    pilot measurement.
    """
    from collections import Counter

    buckets = []
    for _ in range(n_probes):
        response = _query_llm(question, llm_service_url, api_key)
        buckets.append(_decision_match_semantic(response, gold_decision))
    if not buckets:
        return 0.0
    majority_count = Counter(buckets).most_common(1)[0][1]
    return majority_count / len(buckets)


# ── Core attack class ────────────────────────────────────────────────────────

class MIAAttack:
    """
    Membership Inference Attack for Reliable-dRAG.

    Parameters
    ----------
    llm_service_url      : URL of the running LLM service (default localhost:9000)
    corpus_jsonl          : path to JSONL used to determine loaded contexts
    n_members             : how many member documents to probe (default 25)
    n_nonmembers          : how many non-member documents to probe (default 25)
    probes_per_doc        : real SQuAD questions tried per document, max
                             similarity kept (default 4)
    threshold_percentile  : similarity percentile as decision boundary (default 50)
    api_key               : X-API-Key header if auth is enabled on LLM service
    random_seed           : RNG seed for reproducibility (default 42)
    """

    def __init__(
        self,
        llm_service_url:      str = LLM_SERVICE_URL,
        corpus_jsonl:         str = CORPUS_JSONL,
        n_members:            int = DEFAULT_MEMBERS,
        n_nonmembers:         int = DEFAULT_NONMEMBERS,
        probes_per_doc:       int = 4,
        threshold_percentile: int = 50,
        api_key:              str = "",
        random_seed:          int = 42,
    ):
        if not _ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required.\n"
                "Install: pip install sentence-transformers"
            )
        self.llm_service_url   = llm_service_url
        self.corpus_jsonl      = corpus_jsonl
        self.n_members         = n_members
        self.n_nonmembers      = n_nonmembers
        self.probes_per_doc    = probes_per_doc
        self.threshold_pct     = threshold_percentile
        self.api_key           = api_key
        self.random_seed       = random_seed
        print(f"  [MIA] Loading sentence-transformer: {EMBEDDING_MODEL}")
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)

    def run(self) -> Dict[str, Any]:
        """
        Execute the full MIA pipeline.

        Returns dict with keys: confusion_matrix, attack_accuracy, precision,
        recall, f1_score, auc_roc (now driven by the Revision 10 calibrated
        decision score, see module docstring), privacy_risk,
        n_members_tested, n_non_members_tested, mean_member_similarity,
        mean_non_member_similarity, similarity_delta, plus the secondary
        answer-match diagnostic signal (auc_roc_answer_match,
        mean_member_answer_match_rate, mean_non_member_answer_match_rate,
        answer_match_delta -- uncalibrated, unchanged meaning from Revision
        7), and the new Revision 10 diagnostics: auc_roc_pretraining_baseline,
        mean_{member,non_member}_pretraining_baseline_match,
        pretraining_baseline_delta (how much of decision_match would have
        happened anyway with no retrieval at all -- see module docstring
        §(a)), and mean_{member,non_member}_degraded_probe_rate (how often a
        data source failed even after retry -- see module docstring §(b)).
        """
        members, non_members = load_membership_documents(
            self.n_members, self.n_nonmembers, self.random_seed,
            self.corpus_jsonl, self.probes_per_doc,
        )

        print("\n  [MIA] Checking LLM service connectivity ...")
        _test = _query_llm("connectivity test", self.llm_service_url, self.api_key)
        if not _test:
            print(f"  [warn] LLM service at {self.llm_service_url} returned empty "
                  "response. Similarities for unreachable queries will be 0.0.\n"
                  "  Ensure drag_llm_service is running and --llm_url is correct.")

        print("\n  === Probing MEMBER documents ===")
        (member_scores, member_matches, member_sims, member_lens, member_certs,
         member_baseline, member_degraded) = self._probe_documents(members, "MEMBER")

        print("\n  === Probing NON-MEMBER documents ===")
        (non_member_scores, non_member_matches, non_member_sims, non_member_lens, non_member_certs,
         non_member_baseline, non_member_degraded) = self._probe_documents(non_members, "NON-MEMBER")

        y_true    = np.array([1] * len(member_scores) + [0] * len(non_member_scores))
        y_scores  = np.array(member_scores + non_member_scores)  # calibrated weighted composite (primary, Revision 10)
        threshold = float(np.percentile(y_scores, self.threshold_pct))
        y_pred    = (y_scores >= threshold).astype(int)

        metrics = self._compute_metrics(
            y_true, y_pred, y_scores,
            member_sims, non_member_sims,
            member_matches, non_member_matches,
            member_lens, non_member_lens,
            member_certs, non_member_certs,
            member_baseline, non_member_baseline,
            member_degraded, non_member_degraded,
        )
        self._print_summary(metrics)
        return metrics

    # ── internals ──────────────────────────────────────────────────────────

    def _probe_documents(
        self, documents: List[List[Dict[str, str]]], label: str,
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
        """
        For each document, probe every available question (up to
        probes_per_doc) and take the probe with the highest similarity --
        one attacker keeping their strongest result over several tries, not
        a single noisy sample.

        Revision 10 (see module docstring §(a)/§(b)): each probe now also
        queries the same question with `no_retrieval=True` (the pretraining-
        knowledge calibration baseline) and records whether the server
        reported the retrieval-mode query as `degraded` (a sampled data
        source failed even after `_query_llm_raw()`'s retry). The primary
        composite score now uses `_calibrated_decision_score()` (RAG match
        netted against the no-RAG baseline) in place of the raw, uncalibrated
        decision-match rate -- see DECISION_WEIGHT usage below.

        Returns seven parallel per-document lists: composite score
        (primary, now calibrated), raw uncalibrated answer-match rate
        (diagnostic, unchanged meaning from Revision 7), raw max similarity
        (diagnostic), length ratio (diagnostic), certainty (diagnostic),
        pretraining-baseline match rate (diagnostic, new), and degraded-probe
        rate (diagnostic, new) -- all but the last two are taken at the
        best-sim probe, matching Revision 7's behavior exactly.
        """
        composite_scores: List[float] = []
        match_rates: List[float] = []
        raw_sims: List[float] = []
        length_ratios: List[float] = []
        certainties: List[float] = []
        baseline_match_rates: List[float] = []
        degraded_rates: List[float] = []
        n = len(documents)
        for i, qa_list in enumerate(documents, 1):
            sims, matches, lens, certs, calibrated, baseline_matches, degraded_flags = [], [], [], [], [], [], []
            for qa in qa_list:
                response, degraded = _query_llm_raw(
                    qa["question"], self.llm_service_url, self.api_key,
                )
                if response.strip():
                    emb_r = self._encoder.encode([response], convert_to_numpy=True)[0]
                    emb_c = self._encoder.encode([qa["context"]], convert_to_numpy=True)[0]
                    sim   = _cosine_similarity(emb_r, emb_c)
                else:
                    sim = 0.0
                sims.append(sim)
                rag_match = _decision_match(response, qa.get("decision", ""))
                matches.append(1.0 if rag_match else 0.0)
                lens.append(_answer_length_ratio(response, qa["answer"]))
                certs.append(_certainty_score(response))
                degraded_flags.append(degraded)

                # Pretraining-knowledge calibration baseline (module docstring
                # §(a)): same question, same model/prompt template, no
                # retrieved context. Only worth the extra call when the
                # RAG-grounded answer actually matched -- a non-match is
                # already scored 0.0 by _calibrated_decision_score()
                # regardless of the baseline, so skip it to halve the added
                # LLM-call cost in the (dataset-dependent) common case where
                # decision_match is 0.
                if rag_match:
                    baseline_response, _ = _query_llm_raw(
                        qa["question"], self.llm_service_url, self.api_key, no_retrieval=True,
                    )
                    baseline_match = _decision_match(baseline_response, qa.get("decision", ""))
                else:
                    baseline_match = False
                baseline_matches.append(1.0 if baseline_match else 0.0)
                calibrated.append(_calibrated_decision_score(rag_match, baseline_match))

            best_idx      = int(np.argmax(sims)) if sims else 0
            max_sim       = sims[best_idx] if sims else 0.0
            best_len_ratio = lens[best_idx] if lens else 0.0
            best_certainty = certs[best_idx] if certs else 0.0
            match_rate    = (sum(matches) / len(matches)) if matches else 0.0
            calibrated_rate = (sum(calibrated) / len(calibrated)) if calibrated else 0.0
            baseline_rate = (sum(baseline_matches) / len(baseline_matches)) if baseline_matches else 0.0
            degraded_rate = (sum(degraded_flags) / len(degraded_flags)) if degraded_flags else 0.0
            # Gated: DECISION_WEIGHT dominates by construction (see module docstring),
            # so a fully-calibrated match (1.0) always outranks a non-match (0.0)
            # regardless of how the other three signals land -- they only
            # re-rank within a gate tier. Uses the CALIBRATED rate (Revision 10),
            # not the raw match_rate (kept as a diagnostic below).
            composite     = (DECISION_WEIGHT * calibrated_rate
                              + SIM_WEIGHT * _normalize_similarity(max_sim)
                              + CERTAINTY_WEIGHT * best_certainty
                              + LEN_WEIGHT * best_len_ratio)

            composite_scores.append(composite)
            match_rates.append(match_rate)
            raw_sims.append(max_sim)
            length_ratios.append(best_len_ratio)
            certainties.append(best_certainty)
            baseline_match_rates.append(baseline_rate)
            degraded_rates.append(degraded_rate)

            preview = qa_list[0]["question"][:40] + "..." if qa_list and len(qa_list[0]["question"]) > 40 else (qa_list[0]["question"] if qa_list else "")
            print(f"    [{label}] doc {i:>2}/{n}  probes={len(qa_list)}  "
                  f"score={composite:.4f}  sim={max_sim:.4f}  certainty={best_certainty:.2f}  "
                  f"len_ratio={best_len_ratio:.2f}  match_rate={match_rate:.2f}  "
                  f"baseline_match={baseline_rate:.2f}  degraded={degraded_rate:.2f}  q0='{preview}'")
        return (composite_scores, match_rates, raw_sims, length_ratios, certainties,
                baseline_match_rates, degraded_rates)

    def _compute_metrics(
        self,
        y_true:             np.ndarray,
        y_pred:              np.ndarray,
        y_scores:            np.ndarray,
        member_scores:       List[float],
        non_member_scores:   List[float],
        member_matches:      List[float],
        non_member_matches:  List[float],
        member_lens:         List[float],
        non_member_lens:     List[float],
        member_certs:        List[float],
        non_member_certs:    List[float],
        member_baseline:     List[float],
        non_member_baseline: List[float],
        member_degraded:     List[float],
        non_member_degraded: List[float],
    ) -> Dict[str, Any]:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        accuracy  = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall    = float(recall_score(y_true, y_pred, zero_division=0))
        f1        = float(f1_score(y_true, y_pred, zero_division=0))
        # y_scores is the gated composite (calibrated decision_match, weight
        # 1.0 as of Revision 7's grid search, dominates by construction;
        # similarity/certainty/length_ratio are weighted 0.0 and only exist
        # as diagnostics -- see DECISION_WEIGHT etc. above). As of Revision 10
        # the decision_match term feeding this composite is the
        # pretraining-calibrated score, not the raw match rate -- see module
        # docstring §(a) and _calibrated_decision_score(). This is the
        # primary "thesis metric" AUC-ROC.
        auc_roc   = (
            float(roc_auc_score(y_true, y_scores))
            if len(np.unique(y_true)) > 1 else 0.5
        )
        mu_m  = float(np.mean(member_scores))     if member_scores     else 0.0
        mu_nm = float(np.mean(non_member_scores)) if non_member_scores else 0.0

        match_scores = np.array(member_matches + non_member_matches)
        auc_match = (
            float(roc_auc_score(y_true, match_scores))
            if len(np.unique(y_true)) > 1 and len(np.unique(match_scores)) > 1 else 0.5
        )
        mm_m  = float(np.mean(member_matches))     if member_matches     else 0.0
        mm_nm = float(np.mean(non_member_matches)) if non_member_matches else 0.0

        len_scores = np.array(member_lens + non_member_lens)
        auc_len = (
            float(roc_auc_score(y_true, len_scores))
            if len(np.unique(y_true)) > 1 and len(np.unique(len_scores)) > 1 else 0.5
        )
        ml_m  = float(np.mean(member_lens))     if member_lens     else 0.0
        ml_nm = float(np.mean(non_member_lens)) if non_member_lens else 0.0

        cert_scores = np.array(member_certs + non_member_certs)
        auc_cert = (
            float(roc_auc_score(y_true, cert_scores))
            if len(np.unique(y_true)) > 1 and len(np.unique(cert_scores)) > 1 else 0.5
        )
        mc_m  = float(np.mean(member_certs))     if member_certs     else 0.0
        mc_nm = float(np.mean(non_member_certs)) if non_member_certs else 0.0

        # Revision 10 diagnostics -- see module docstring §(a)/§(b).
        baseline_scores = np.array(member_baseline + non_member_baseline)
        # A HIGH auc_roc_pretraining_baseline is a bad sign, not a good one:
        # it means the no-RAG baseline alone already separates member from
        # non-member questions, which would mean the *dataset/probe design*
        # (not the RAG pipeline) is doing the discriminating -- the opposite
        # of what this attack claims to measure. Ideally this sits near 0.50
        # (RAGLeak paper §5.3's "before RAG" check, done live every run).
        auc_baseline = (
            float(roc_auc_score(y_true, baseline_scores))
            if len(np.unique(y_true)) > 1 and len(np.unique(baseline_scores)) > 1 else 0.5
        )
        mb_m  = float(np.mean(member_baseline))     if member_baseline     else 0.0
        mb_nm = float(np.mean(non_member_baseline)) if non_member_baseline else 0.0

        md_m  = float(np.mean(member_degraded))     if member_degraded     else 0.0
        md_nm = float(np.mean(non_member_degraded)) if non_member_degraded else 0.0

        return {
            "confusion_matrix":           {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "attack_accuracy":            round(accuracy,  4),
            "precision":                  round(precision, 4),
            "recall":                     round(recall,    4),
            "f1_score":                   round(f1,        4),
            "auc_roc":                    round(auc_roc,   4),
            "privacy_risk":               self._privacy_risk(auc_roc),
            "n_members_tested":           int(np.sum(y_true == 1)),
            "n_non_members_tested":       int(np.sum(y_true == 0)),
            "mean_member_similarity":     round(mu_m,         4),
            "mean_non_member_similarity": round(mu_nm,        4),
            "similarity_delta":           round(mu_m - mu_nm, 4),
            "score_weights":              {"similarity": SIM_WEIGHT, "certainty": CERTAINTY_WEIGHT,
                                            "length_ratio": LEN_WEIGHT, "decision_match": DECISION_WEIGHT},
            # auc_roc_answer_match/mean_*_answer_match_rate/answer_match_delta are the
            # RAW, UNCALIBRATED decision_match signal (see _decision_match). Through
            # Revision 7 this WAS folded directly into the primary composite via
            # DECISION_WEIGHT; as of Revision 10 the composite instead uses the
            # CALIBRATED decision score (see auc_roc/_calibrated_decision_score, and
            # module docstring §(a)) -- these fields are kept, unchanged in meaning,
            # purely as a diagnostic for continuity with prior revisions' reports.
            # Field names kept unchanged for JSON/back-compat with earlier logs.
            "auc_roc_answer_match":            round(auc_match, 4),
            "mean_member_answer_match_rate":     round(mm_m,  4),
            "mean_non_member_answer_match_rate": round(mm_nm, 4),
            "answer_match_delta":               round(mm_m - mm_nm, 4),
            # length_ratio, certainty remain diagnostic breakdowns of the other composite terms
            "auc_roc_length_ratio":             round(auc_len, 4),
            "mean_member_length_ratio":          round(ml_m,  4),
            "mean_non_member_length_ratio":      round(ml_nm, 4),
            "length_ratio_delta":               round(ml_m - ml_nm, 4),
            "auc_roc_certainty":                round(auc_cert, 4),
            "mean_member_certainty":             round(mc_m,  4),
            "mean_non_member_certainty":         round(mc_nm, 4),
            "certainty_delta":                  round(mc_m - mc_nm, 4),
            # Revision 10 diagnostics (module docstring §(a)) -- the pretraining-
            # knowledge calibration baseline: what decision_match would have been
            # with NO retrieved context at all, through the identical model/prompt
            # path (drag_llm_service's new `no_retrieval` bypass). A high value
            # here (especially a high non-member value) is the live signal that
            # "the LLM already knew the fact from pretraining" -- previously only
            # a print-time footnote, now measured and folded into auc_roc itself.
            "auc_roc_pretraining_baseline":             round(auc_baseline, 4),
            "mean_member_pretraining_baseline_match":     round(mb_m,  4),
            "mean_non_member_pretraining_baseline_match": round(mb_nm, 4),
            "pretraining_baseline_delta":                round(mb_m - mb_nm, 4),
            # Revision 10 diagnostics (module docstring §(b)) -- fraction of probes
            # where drag_llm_service reported `degraded: true` (a sampled data
            # source failed even after retry-with-backoff). Nonzero values mean
            # some fraction of this run's answers were generated on incomplete
            # retrieval -- a measurement-quality caveat on auc_roc, not a
            # membership signal itself.
            "mean_member_degraded_probe_rate":     round(md_m,  4),
            "mean_non_member_degraded_probe_rate": round(md_nm, 4),
        }

    @staticmethod
    def _privacy_risk(auc_roc: float) -> str:
        if auc_roc > 0.90: return "CRITICAL — severe privacy leak"
        if auc_roc > 0.75: return "HIGH — significant privacy risk"
        if auc_roc > 0.60: return "MEDIUM — moderate privacy concern"
        if auc_roc > 0.50: return "LOW — slight privacy risk"
        return "NEGLIGIBLE — attack ineffective (AUC ≈ random)"

    @staticmethod
    def _print_summary(m: Dict[str, Any]) -> None:
        cm = m["confusion_matrix"]
        print("\n" + "=" * 62)
        print("  MIA — Results")
        print("=" * 62)
        print(f"  Members tested       : {m['n_members_tested']}")
        print(f"  Non-members tested   : {m['n_non_members_tested']}")
        print(f"  Confusion matrix     : TP={cm['tp']}  TN={cm['tn']}  "
              f"FP={cm['fp']}  FN={cm['fn']}")
        print(f"  Attack accuracy      : {m['attack_accuracy']:.4f}")
        print(f"  Precision            : {m['precision']:.4f}")
        print(f"  Recall               : {m['recall']:.4f}")
        print(f"  F1 score             : {m['f1_score']:.4f}")
        w = m["score_weights"]
        print(f"  AUC-ROC              : {m['auc_roc']:.4f}  ← thesis metric "
              f"(GATED, Revision 10: CALIBRATED decision_match*{w['decision_match']} dominates; "
              f"sim*{w['similarity']} + certainty*{w['certainty']} + "
              f"len_ratio*{w['length_ratio']} only re-rank within a gate tier. "
              f"Calibrated against the no-RAG pretraining baseline below -- see module docstring §(a))")
        print(f"  Privacy risk         : {m['privacy_risk']}")
        print(f"\n  Mean member sim      : {m['mean_member_similarity']:.4f}")
        print(f"  Mean non-member sim  : {m['mean_non_member_similarity']:.4f}")
        print(f"  Similarity delta     : {m['similarity_delta']:+.4f}")
        print(f"\n  [diagnostic] Decision-match AUC-ROC (UNCALIBRATED, Rev. 7 signal): {m['auc_roc_answer_match']:.4f}")
        print(f"  [diagnostic] Member decision-match rate: {m['mean_member_answer_match_rate']:.4f}")
        print(f"  [diagnostic] Non-member decision-match : {m['mean_non_member_answer_match_rate']:.4f}"
              "  <- high here means the LLM already knew the fact from pretraining")
        print(f"  [diagnostic] Decision-match delta       : {m['answer_match_delta']:+.4f}")
        print(f"\n  [diagnostic] Pretraining-baseline AUC-ROC : {m['auc_roc_pretraining_baseline']:.4f}"
              "  <- ideally ~0.50; a HIGH value means the no-RAG baseline alone")
        print(f"                                                already separates members from non-members "
              "(dataset/probe confound, not RAG leakage -- see module docstring §(a))")
        print(f"  [diagnostic] Member pretraining-baseline match    : {m['mean_member_pretraining_baseline_match']:.4f}")
        print(f"  [diagnostic] Non-member pretraining-baseline match: {m['mean_non_member_pretraining_baseline_match']:.4f}")
        print(f"  [diagnostic] Pretraining-baseline delta            : {m['pretraining_baseline_delta']:+.4f}")
        print(f"\n  [diagnostic] Member degraded-probe rate     : {m['mean_member_degraded_probe_rate']:.4f}"
              "  <- fraction of probes answered on incomplete retrieval")
        print(f"  [diagnostic] Non-member degraded-probe rate : {m['mean_non_member_degraded_probe_rate']:.4f}"
              "  (a data source failed even after retry -- see module docstring §(b))")
        print(f"\n  [diagnostic] Length-ratio AUC-ROC      : {m['auc_roc_length_ratio']:.4f}")
        print(f"  [diagnostic] Member length ratio       : {m['mean_member_length_ratio']:.4f}")
        print(f"  [diagnostic] Non-member length ratio   : {m['mean_non_member_length_ratio']:.4f}")
        print(f"  [diagnostic] Length-ratio delta         : {m['length_ratio_delta']:+.4f}")
        print(f"\n  [diagnostic] Certainty AUC-ROC         : {m['auc_roc_certainty']:.4f}")
        print(f"  [diagnostic] Member certainty          : {m['mean_member_certainty']:.4f}")
        print(f"  [diagnostic] Non-member certainty      : {m['mean_non_member_certainty']:.4f}")
        print(f"  [diagnostic] Certainty delta            : {m['certainty_delta']:+.4f}")
        print("=" * 62)
