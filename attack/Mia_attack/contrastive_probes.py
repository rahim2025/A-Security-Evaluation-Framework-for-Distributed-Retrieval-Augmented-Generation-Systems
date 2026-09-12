"""
attack/Mia_attack/contrastive_probes.py

Revision 9, Phase 2 -- richer per-document features for the trained
attacker: wording + multiple contrastive probes per document (NAACL
reviewer feedback, see reports/updated_reports_safin/README.md).

Relationship to trained_attacker.py (Phase 1)
-----------------------------------------------
Phase 1 (`trained_attacker.py`) trains a classifier on the four signals
Revision 7 already computed (norm_sim, certainty, length_ratio,
decision_match), reusing already-collected data -- no new LLM calls. That
addresses "train a classifier instead of a hand-designed heuristic" but
not the rest of the reviewer's description of a stronger attacker: "using
features such as wording, confidence, similarity, and responses across
multiple probes."

This file collects that richer feature set. It requires fresh live LLM
calls -- unlike Phase 1, it has NOT been executed this session, because
the local Docker stack (`drag_llm_service`, `drag-data-source-0/20/100`)
was not brought up (see reports/updated_reports_safin/README.md for why:
bringing up and querying the full stack at DEV_SEEDS+TEST_SEEDS scale --
13+5 seeds x 50 docs x several probes/doc -- is a multi-hour live-inference
run against a real service, which this pass intentionally scoped out
rather than launching unreviewed).

Why "contrastive probes" instead of repeating the identical question
-----------------------------------------------------------------------
Revision 6 already tried repeating the *identical* question `n_probes`
times (`_consistency_score` in mia_attack.py) to get multiple independent
probes per document. Revision 8 found that signal is mechanically
non-viable on this deployment: under temperature=0.0 (deterministic)
decoding, the identical question always returns the identical answer, so
repeated-identical-question consistency is 1.0 for every document,
member or non-member alike (AUC exactly 0.50 at all 3 fresh seeds tested,
sec 2.10 of the MIA report).

A *contrastive* probe is a different question -- a paraphrase or a
different framing of the same underlying fact -- so it is not subject to
that critique: even under fully deterministic decoding, two different
input strings can and do produce different outputs. Whether the
underlying hypothesis holds (a member document's real backing passage
should let the model answer several differently-worded questions about it
consistently and correctly, whereas a non-member has no passage to ground
any of them) is an open empirical question this script is built to test,
not an assumed result -- see the "what this does NOT claim" note at the
bottom of this docstring.

Design of the contrastive templates
--------------------------------------
PubMedQA's questions are yes/no/maybe judgments about a research finding.
The templates below rephrase the *framing* of the same underlying
question without changing what is being asked, so the same `gold_decision`
label from mia_attack.py's `load_membership_documents()` still applies to
every paraphrase -- there is no need for separate gold labels per
paraphrase. These are hand-written templates, not LLM-paraphrased: using
the *target* LLM (or another LLM) to generate paraphrases would risk the
paraphraser itself leaking information correlated with membership (e.g.
producing more fluent paraphrases for text closer to its training
distribution), which would be a confound worth avoiding rather than
discovering after the fact. This is a limitation, not a hidden assumption:
five fixed templates give five fixed *framings*, not five independently-
sampled framings, so any residual template-specific bias (one phrasing
happening to be easier or harder in general, independent of membership)
is a real risk that a proper analysis should check for (e.g. per-template
AUC, not just the aggregate) -- see the README's caveats section.

Wording features
-------------------
`_wording_features()` computes, per response: character length, word
count, a hedge-word fraction (reusing mia_attack.py's HEDGE_WORDS list so
the two modules agree on what counts as hedging), and a small curated
refusal-phrase detector (e.g. "cannot determine", "not enough information",
"insufficient evidence") -- phrases a model might use to decline
committing to yes/no/maybe without necessarily using one of the single
hedge *words* already tracked. This is a small, hand-curated feature set,
not a general-purpose text vectorizer (TF-IDF, embeddings-as-features,
etc.) -- with only ~50 documents per seed and 13-18 seeds total, a
high-dimensional bag-of-words feature space fit per-seed would be a
serious overfitting risk (more features than documents, in a dataset
where the vocabulary itself -- "yes"/"no"/"maybe" plus a handful of
hedges -- is already known to be the dominant signal, per Revision 7's
finding that decision_match alone outperforms every richer composite
tried so far). A handful of interpretable, hand-justified features is a
deliberately conservative choice given that dataset size, consistent with
Revision 6/7's own repeated caution against adding signals without
evidence they generalize.

Output
--------
Per document, this script aggregates across all contrastive probes into a
feature vector meant as a drop-in replacement for tune_weights.py's
4-feature rows, with additional multi-probe and wording columns:

  norm_sim_mean, norm_sim_max, norm_sim_std     (was: single "norm_sim",
                                                  the best-of-N value only)
  certainty_mean, certainty_max
  length_ratio_mean
  decision_match_rate    (fraction of probes matching gold -- generalizes
                           the existing best-of-N decision_match to a
                           multi-probe agreement rate)
  decision_match_std     (0 if all probes agree; the literal "contrastive
                           consistency" signal the reviewer described --
                           NOT subject to Revision 8's negative finding,
                           since these are different questions, not
                           repeated-identical ones)
  hedge_fraction_mean, char_length_mean, word_count_mean, refusal_rate

Two commands, mirroring tune_weights.py:

  python contrastive_probes.py collect --split dev
  python contrastive_probes.py collect --split test

Deliberately no `search`/`evaluate` subcommands here -- once collected,
this data plugs into a variant of trained_attacker.py's model-selection
and locked one-shot evaluation (reuse that script's pattern rather than
duplicating it; left as the explicit next step in the README rather than
built speculatively before there is any real data to run it on).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.Mia_attack.mia_attack import (  # noqa: E402
    CORPUS_JSONL,
    DEFAULT_MEMBERS,
    DEFAULT_NONMEMBERS,
    HEDGE_WORDS,
    LLM_SERVICE_URL,
    _answer_length_ratio,
    _certainty_score,
    _cosine_similarity,
    _decision_match,
    _normalize_similarity,
    _parse_llm_response,
    load_membership_documents,
)
from attack.Mia_attack.tune_weights import DEV_SEEDS, TEST_SEEDS  # noqa: E402

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install sentence-transformers") from exc

import requests

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Same rationale as tune_weights.py's _QUERY_TIMEOUT_SECONDS: a handful of
# specific PubMedQA questions were confirmed (Revision 7) to hang the live
# service for 120s+; a short timeout treated as an empty response keeps a
# multi-probe x multi-seed collection tractable instead of stalling on a
# rare slow query. This script queries MORE times per document than
# tune_weights.py did (one call per contrastive template, not one call per
# available dataset question), so the cost of a long per-query timeout is
# proportionally higher here.
_QUERY_TIMEOUT_SECONDS = 20

_DATA_DIR = os.path.join(_ROOT, "attack_logs")
CONTRASTIVE_DEV_PATH = os.path.join(_DATA_DIR, "contrastive_dev_signals.json")
CONTRASTIVE_TEST_PATH = os.path.join(_DATA_DIR, "contrastive_test_signals.json")

# Refusal phrases distinct from single hedge WORDS (HEDGE_WORDS catches
# "may"/"might"/"unclear"/etc. as individual tokens; these are short
# multi-word phrases a model might use to decline a direct yes/no/maybe
# commitment without necessarily containing any single hedge word).
_REFUSAL_PHRASES = (
    "cannot determine", "can't determine", "cannot be determined",
    "not enough information", "insufficient evidence", "insufficient information",
    "not clear from", "unable to determine", "no clear answer",
    "not possible to say", "difficult to say",
)


def _generate_contrastive_probes(base_question: str) -> List[str]:
    """
    Five fixed-template paraphrases of the same underlying yes/no/maybe
    question, including the original. See module docstring's "Design of
    the contrastive templates" section for why these are hand-written
    rather than model-generated, and the per-template-bias caveat that
    follows from that choice.

    Every template preserves the original question's factual content so
    the same `gold_decision` label still applies -- these change framing
    (direct restatement, hedge-testing framing, confidence-probing
    framing, negation-probing framing, terse framing), not content.
    """
    q = base_question.strip().rstrip("?").strip()
    if not q:
        return [base_question]
    # Lowercase the first character for mid-sentence templates, but keep a
    # capitalized version for templates that start the sentence with it.
    q_lower_first = q[0].lower() + q[1:] if q else q
    return [
        base_question,  # original, verbatim
        f"Based on the available evidence, {q_lower_first}?",
        f"In your assessment, would you say: {q_lower_first}?",
        f"Is it accurate to state that {q_lower_first}?",
        f"{q}? Answer with yes, no, or maybe.",
    ]


def _wording_features(response: str) -> Dict[str, float]:
    text = response.strip()
    words = [w.strip(".,;:!?").lower() for w in text.split()]
    n_words = len(words)
    hedge_frac = (
        sum(1 for w in words if w in HEDGE_WORDS) / n_words if n_words else 0.0
    )
    lowered = text.lower()
    is_refusal = any(phrase in lowered for phrase in _REFUSAL_PHRASES)
    return {
        "char_length": float(len(text)),
        "word_count": float(n_words),
        "hedge_fraction": float(hedge_frac),
        "is_refusal": 1.0 if is_refusal else 0.0,
    }


def _query_llm_fast(question: str, url: str, api_key: str = "") -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        r = requests.post(
            f"{url}/query", json={"query": question}, headers=headers,
            timeout=_QUERY_TIMEOUT_SECONDS,
        )
        if r.status_code == 200:
            return _parse_llm_response(r.json())
    except Exception:
        pass
    return ""


def _collect_contrastive_signals(
    seeds: List[int], llm_url: str, api_key: str,
    n_members: int, n_nonmembers: int,
) -> List[Dict[str, Any]]:
    """
    For each document, probes it with every contrastive template applied
    to its (first available) real PubMedQA question, then aggregates
    across those probes into one feature row. Costs len(templates) LLM
    calls per document -- 5x tune_weights.py's per-document cost (which
    itself already used up to `probes_per_doc` real dataset questions
    where available; PubMedQA has ~1 question/doc, so that degraded to
    1 call/doc -- this script's 5 templates are the actual source of
    multi-probing for this dataset, not `probes_per_doc`).
    """
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    rows: List[Dict[str, Any]] = []
    n_timeouts = 0
    n_total_queries = 0

    for seed in seeds:
        # probes_per_doc=1: this script generates its own multi-probing via
        # contrastive templates, so it only needs the single base question
        # per document from load_membership_documents(), not PubMedQA's
        # (near-always-absent) additional dataset questions.
        members, non_members = load_membership_documents(
            n_members, n_nonmembers, seed, CORPUS_JSONL, probes_per_doc=1,
        )
        doc_counter = 0
        total_docs = len(members) + len(non_members)
        for label, documents in ((1, members), (0, non_members)):
            for qa_list in documents:
                doc_counter += 1
                if doc_counter % 5 == 0 or doc_counter == total_docs:
                    print(f"    [contrastive] seed={seed} doc {doc_counter}/{total_docs} "
                          f"[timeouts: {n_timeouts}/{n_total_queries}]", flush=True)
                if not qa_list:
                    continue
                qa = qa_list[0]
                gold_decision = qa.get("decision", "")
                templates = _generate_contrastive_probes(qa["question"])

                sims, certs, lens_, matches, wordings = [], [], [], [], []
                for probe_q in templates:
                    t0 = time.time()
                    response = _query_llm_fast(probe_q, llm_url, api_key)
                    n_total_queries += 1
                    if not response.strip() and time.time() - t0 >= _QUERY_TIMEOUT_SECONDS - 1:
                        n_timeouts += 1
                    if response.strip():
                        emb_r = encoder.encode([response], convert_to_numpy=True)[0]
                        emb_c = encoder.encode([qa["context"]], convert_to_numpy=True)[0]
                        sim = _cosine_similarity(emb_r, emb_c)
                    else:
                        sim = 0.0
                    sims.append(_normalize_similarity(sim))
                    certs.append(_certainty_score(response))
                    lens_.append(_answer_length_ratio(response, qa["answer"]))
                    matches.append(1.0 if _decision_match(response, gold_decision) else 0.0)
                    wordings.append(_wording_features(response))

                rows.append({
                    "seed": seed,
                    "label": label,
                    "n_probes": len(templates),
                    "norm_sim_mean": float(np.mean(sims)) if sims else 0.0,
                    "norm_sim_max": float(np.max(sims)) if sims else 0.0,
                    "norm_sim_std": float(np.std(sims)) if len(sims) > 1 else 0.0,
                    "certainty_mean": float(np.mean(certs)) if certs else 0.0,
                    "certainty_max": float(np.max(certs)) if certs else 0.0,
                    "length_ratio_mean": float(np.mean(lens_)) if lens_ else 0.0,
                    "decision_match_rate": float(np.mean(matches)) if matches else 0.0,
                    "decision_match_std": float(np.std(matches)) if len(matches) > 1 else 0.0,
                    "hedge_fraction_mean": float(np.mean([w["hedge_fraction"] for w in wordings])) if wordings else 0.0,
                    "char_length_mean": float(np.mean([w["char_length"] for w in wordings])) if wordings else 0.0,
                    "word_count_mean": float(np.mean([w["word_count"] for w in wordings])) if wordings else 0.0,
                    "refusal_rate": float(np.mean([w["is_refusal"] for w in wordings])) if wordings else 0.0,
                })
        print(f"  [contrastive] seed={seed} collected ({len(members)} members, "
              f"{len(non_members)} non-members)  [timeouts so far: {n_timeouts}/{n_total_queries}]",
              flush=True)

    print(f"  [contrastive] TOTAL timeouts: {n_timeouts}/{n_total_queries} queries "
          f"({100.0 * n_timeouts / max(n_total_queries, 1):.1f}%) -- treated as empty responses")
    return rows


def cmd_collect(args: argparse.Namespace) -> None:
    default_seeds = DEV_SEEDS if args.split == "dev" else TEST_SEEDS
    seeds = args.seeds if args.seeds else default_seeds
    if args.seeds:
        # Deliberately unrestricted (unlike trained_attacker.py's DEV_SEEDS/
        # TEST_SEEDS assertions): a --seeds override is meant for a cheaper
        # pilot run (e.g. 3 seeds instead of the full 13/5) or a scale-up to
        # brand-new seeds, both legitimate uses. The full-split default when
        # --seeds is omitted is what keeps this comparable to Phase 1's
        # locked DEV_SEEDS/TEST_SEEDS by default.
        print(f"[contrastive] --seeds override: collecting {args.split} data for "
              f"{seeds} instead of the full default {default_seeds}")
    out_path = CONTRASTIVE_DEV_PATH if args.split == "dev" else CONTRASTIVE_TEST_PATH
    n_templates = len(_generate_contrastive_probes("placeholder question"))
    n_docs = (args.n_members + args.n_nonmembers) * len(seeds)
    print(f"[contrastive] Collecting contrastive+wording signals for {args.split} "
          f"seeds: {seeds}")
    print(f"[contrastive] Estimated live LLM calls: {n_docs} docs x {n_templates} "
          f"templates = {n_docs * n_templates} queries. This is a live-inference run "
          f"against a real service -- see reports/updated_reports_safin/README.md "
          f"before running this against a shared deployment.")
    rows = _collect_contrastive_signals(
        seeds, args.llm_url, args.api_key, args.n_members, args.n_nonmembers,
    )
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"[contrastive] Wrote {len(rows)} document rows -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    c1 = sub.add_parser("collect")
    c1.add_argument("--split", choices=["dev", "test"], required=True)
    c1.add_argument("--llm_url", default=LLM_SERVICE_URL)
    c1.add_argument("--api_key", default="")
    c1.add_argument("--n_members", type=int, default=DEFAULT_MEMBERS)
    c1.add_argument("--n_nonmembers", type=int, default=DEFAULT_NONMEMBERS)
    c1.add_argument("--seeds", type=int, nargs="+", default=None,
                     help="Override the default DEV_SEEDS/TEST_SEEDS list -- e.g. a "
                          "3-seed pilot before committing to the full 13/5-seed split "
                          "(see reports/updated_reports_safin/README.md's recommended "
                          "next step). Omit to use the full default split.")
    c1.set_defaults(func=cmd_collect)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
