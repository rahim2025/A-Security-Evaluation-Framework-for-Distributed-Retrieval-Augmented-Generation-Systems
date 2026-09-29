# MIA Score Mechanism Fix + Rate-Limit Hardening (Revision 10)

**Status: code changed, not executed.** Per instruction, nothing in this
pass was run — no `docker compose up`, no live attack run, no
`py_compile`-beyond-syntax-check verification of behavior. Everything below
is a description of what changed and why; results are not yet re-measured
against the fixed code.

**Scope:** two problems, both raised directly:

1. *"what if the llm model answer based on the pretraining knowledge
   without answering from the rag? then the score mechanism everything is
   weighted to zero"* — the MIA composite score's blind spot to
   pretraining-knowledge confounds.
2. 429 (Too Many Requests) / 500 (Internal Server Error) seen while running
   the Source Selection Manipulation (SSM) attack, with supporting evidence
   in `problems/safin_faced_problems/`.

Both turned out to share a root mechanism and to compound each other — see
§3.

---

## 1. How this works in "normal RAG" (per the source paper)

Source: `RAGLeak: Membership Inference Attacks on RAG-Based Large Language
Models` (Feng, Zhang, Tian, Xu, Zhang, Zhu, Ding, Liu — ACISP 2025), the
PDF supplied for this task.

The paper's method, in brief (its Fig. 1 / §4):

- Crop a real query into a **cropped query** (first N−k words) and a
  **cropped answer** (the true last k words) — the cropped answer becomes
  the ground truth for a *continuation*, not a coarse label.
- Feed the cropped query through the RAG pipeline; if the sample is a
  member, the retriever surfaces the real passage and the LLM's completion
  should resemble the true cropped answer; if it's a non-member, no
  passage exists and the completion should diverge.
- **Grey-box**: threshold the completion's *perplexity* — lower perplexity
  ⇒ member.
- **Black-box**: threshold the *cosine similarity* between the completion
  and the true cropped answer — higher similarity ⇒ member.

Critically, §5.3 ("Exclude LLM Training Data") is the paper's own answer to
exactly the question raised here: *what if the LLM's training data lets it
answer without any retrieval at all?* Their fix is **probe design, not
score-time correction**: because the ground truth is a specific verbatim
continuation (not a coarse category), pretraining alone is extremely
unlikely to reproduce it. They validate this once, empirically (their
Fig. 3): before applying RAG at all, member and non-member similarity
distributions overlap almost completely (pretraining carries no signal);
after RAG, they diverge sharply. That one validation figure is their
confirmation that the probe design avoids the confound — it is not
something the running score itself detects or corrects for on every query,
because their probe design makes it structurally unlikely to occur in the
first place.

That is the key design choice this project's MIA module could not simply
adopt: PubMedQA's questions are yes/no/maybe judgments, not free-text
continuations, so `RAGLeak`'s "make the ground truth too specific to guess"
strategy doesn't transfer directly (switching PubMedQA's own ground truth
to a continuation-style target would mean redesigning the corpus/probe
generation, out of scope for this pass). Instead, Revision 10 (§2 below)
does the *complementary* thing the paper didn't need for its own dataset:
measure and calibrate against the confound live, on every run, rather than
relying on dataset design alone to make it structurally rare.

## 2. How this worked in Reliable-dRAG before this fix, and what "weighted to zero" actually meant

`attack/Mia_attack/mia_attack.py`'s composite score (Revision 7, still true
today except for the fix below):

```
membership_score = DECISION_WEIGHT * decision_match
                  + SIM_WEIGHT * normalized_similarity      # weight 0.0
                  + CERTAINTY_WEIGHT * certainty             # weight 0.0
                  + LEN_WEIGHT * length_ratio                # weight 0.0
```

`DECISION_WEIGHT = 1.0`; the other three are `0.0` (an empirical result
from a prior grid search, `tune_weights.py`, documented in the module's own
history). So in practice the entire score **is** `decision_match`: does the
response's first few words contain the correct gold `yes`/`no`/`maybe`
token (`_decision_match()`)?

This is exactly the failure mode you described. `decision_match` is a
coarse, 3-way categorical judgment:

- PubMedQA skews toward "yes" answers (roughly 55% of the dataset), so a
  model that always guesses "yes" already gets a non-trivial match rate on
  **non-members** it has never seen any context for.
- The deployed LLM (Qwen2.5-1.5B-Instruct) has some non-zero chance of
  already knowing a given biomedical fact from pretraining, independent of
  retrieval.
- Both effects push the **non-member** match rate up, without the RAG
  pipeline having grounded anything. Since the attack's entire signal is
  the *gap* between member and non-member match rates, an inflated
  non-member rate narrows that gap — and if pretraining knowledge (or
  guessing) is strong enough on a given run, the gap can narrow to nothing,
  i.e. AUC-ROC collapses toward 0.50 (chance) with no useful membership
  signal at all. That collapse toward "no discriminative power" is what you
  correctly flagged as "everything weighted to zero" — a real security
  evaluation framework producing a meaningless number without any sign
  in the output that something had gone wrong.

The code already *partially* knew this was possible: the print-time
summary had a footnote —
`"<- high here means the LLM already knew the fact from pretraining"` —
next to the non-member decision-match rate. But that was cosmetic. The
number driving `auc_roc`, the actual thesis metric, did nothing to detect
or correct for it. A run could produce a corrupted, near-chance AUC and
nothing in the returned metrics dict would tell you *why* — whether the
attack genuinely doesn't work here, or whether pretraining knowledge (or,
see §3, rate-limit-corrupted retrieval) quietly erased the signal that run.

## 3. The fix: live pretraining-knowledge calibration

Since PubMedQA's ground truth can't be swapped for a continuation-style
probe without a corpus/probe-generation redesign (out of scope here),
Revision 10 does the complementary thing: measure the pretraining
confound directly, on every probe, and calibrate the score against it,
rather than relying on dataset choice alone (as the paper's own PubMedQA-
equivalent choice — a "less memorizable" dataset — already does one layer
up, and which this project's own docstring already credits as the reason
it moved off SQuAD in the first place).

### 3.1 New no-RAG baseline in `drag_llm_service`

`drag_llm_service/app/server.py`'s `/query` endpoint now accepts an
additive, optional `no_retrieval: true` field. When set, it skips
retrieval and reranking entirely and asks the model to answer from an
**empty context**, through the exact same prompt template
(`"Question: {q}\n\nAnswer: "`) and the same model. This gives a genuine
"what would the LLM say with nothing retrieved at all" baseline, through
the identical generation path — not a separately-reasoned guess about what
pretraining "probably" would produce.

This is additive and backward-compatible: existing callers that never set
`no_retrieval` see identical behavior to before. It also never touches the
data sources, so it costs nothing against the rate-limit budget (§4).

### 3.2 Calibrated decision score in `mia_attack.py`

For every probe where the RAG-grounded response matches the gold decision
(`_decision_match()` returns true), the module now *also* queries the same
question with `no_retrieval=True` and checks whether **that** answer
matches too. `_calibrated_decision_score()` combines the two:

| RAG match | No-RAG baseline match | Calibrated score | Meaning |
|---|---|---|---|
| Yes | No | **1.0** | Clean evidence: retrieval changed the answer to the correct one. |
| Yes | Yes | **0.5** | Ambiguous: correct either way — can't tell genuine grounding from prior knowledge. |
| No | (either) | **0.0** | No committed-correct answer under RAG, regardless of the baseline. |

The primary composite (`auc_roc`, the thesis metric) now uses this
**calibrated** rate in place of the raw match rate. The raw, uncalibrated
`decision_match` is still computed and reported unchanged, under its
existing field name (`auc_roc_answer_match`), purely as a diagnostic for
continuity with prior revisions' numbers.

**Cost control:** the no-RAG baseline query only runs when the RAG-grounded
answer already matched (a non-match is already scored `0.0` regardless of
the baseline), roughly halving the added LLM-call cost versus querying the
baseline unconditionally.

### 3.3 New diagnostics, reported every run

- `auc_roc_pretraining_baseline`, `mean_{member,non_member}_pretraining_baseline_match`,
  `pretraining_baseline_delta` — the module's own live version of the
  paper's Fig. 3 "before RAG vs. after RAG" check. Ideally the baseline AUC
  sits near 0.50; a high value (especially a high **non-member** baseline
  match rate) is now a visible, numeric signal that pretraining knowledge
  is confounding this specific run/dataset, instead of a print-time
  footnote nobody has to look at.
- `mean_{member,non_member}_degraded_probe_rate` — see §4.

**Disclosed, not yet re-validated:** the production weights
(`DECISION_WEIGHT=1.0`, others `0.0`) were grid-searched (Revision 7)
against the *uncalibrated* `decision_match` signal. Calibration changes
what that signal measures. The weights are kept as-is (the gate structure
and the fact that `decision_match` dominates are still reasonable priors),
but a fresh grid search against the *calibrated* signal is recommended
before citing a new AUC number as validated the same way Revision 7's was.
This was flagged in the code and is repeated here rather than silently
assumed to still be optimal.

## 4. The 429 / 500 problem (SSM and MIA both affected)

### 4.1 What was actually happening

`drag_data_source`'s three containers each enforce a 60-requests/minute-
per-IP cap by default (`RATE_LIMIT_DEFAULT` in
`drag_data_source/app/server.py`). Every `/query` (and `/query_analyze`)
call to `drag_llm_service` fans out server-side to each configured data
source — **one HTTP request to each distinct source per external call**
(not three requests piled onto one source — see the correction in §4.4).

Two client scripts had **no pacing between calls at all**:

- `attack/Mia_attack/mia_attack.py`'s `_query_llm()` — a bare
  `requests.post`, no throttle, no retry.
- `attack/ssm_score/run_grounding_farming.py`'s `measure_accuracy()` —
  looped over every eval item back-to-back with zero delay (the *separate*
  `ROUNDS` loop in the same file did have a `time.sleep(0.2)`, but that was
  still faster than the 60/min cap allows for a sustained loop).

This project's *other* live-mode attack clients already knew about and
paced for this exact limit —
`attack/selective_forward_sim/live_network.py` (1.1s between calls) and
`attack/ssm_score/run_attack.py` (`QUERY_DELAY=1.3s`, with a code comment
citing the same root cause) — so the fix is bringing MIA and the
grounding-farming SSM variant in line with an already-established,
already-validated pattern in this codebase, not inventing a new one.

On the server side, `drag_llm_service/app/server.py` previously treated a
429/5xx from any data source identically to "this source has nothing
relevant": logged a warning and silently moved on. If **all** sources
failed this way, `/query` returned its own 500 ("No candidates retrieved
from data sources") — which is the 500 you saw. If **some** sources
failed, the query proceeded on whatever partial context was left,
indistinguishable, from the caller's side, from a complete retrieval.

### 4.2 Why this specifically corrupts membership-inference results

If a member document's real content happened to live on a data source that
got rate-limited away for a given probe, the model answered without any
grounding at all — and that document's response then looks exactly like a
genuine non-member's under the old, uncalibrated scoring. This is a second,
structurally distinct way to get the same symptom as §2/§3 (a real member
scored as if it were a non-member), and the two can compound: a
rate-limited member forced to answer blind is now *also* exactly the
scenario §3's calibration targets, so without both fixes a rate-limited
member could be scored as a confident, calibrated-looking non-member miss
rather than flagged as a measurement artifact.

### 4.3 The fix

- **`drag_llm_service/app/server.py`**: new `_post_data_source_with_retry()`
  helper — retries once on HTTP 429 (honoring the server's `Retry-After`
  header) and on transient 5xx (short exponential backoff), used by both
  `query_data_sources()` (`/query`) and `_query_analyze()`
  (`/query_analyze`, used by SSM's grounding-farming variant). `/query`'s
  response now additionally reports `sources_used` (per-source
  `"ok"`/`"empty"`/`"failed:<reason>"`) and `degraded` (true if any
  configured source did not return `"ok"`) — additive fields; existing
  consumers reading only `response` are unaffected.
- **`attack/Mia_attack/mia_attack.py`**: `_query_llm_raw()` (new) is
  self-throttled (`MIN_QUERY_INTERVAL_S`, default 1.3s, overridable via
  `MIA_MIN_QUERY_INTERVAL_S`) and retries once on 429. The existing
  `_query_llm()` wraps it with an unchanged string-only signature, so
  every other file that already imports `_query_llm`
  (`defense/mia_defense/mia_defense.py`, `run_ablation_eval.py`,
  `validate_length_floor.py`) gets the throttling/retry fix transparently,
  with no changes required on their side. Each probe now also records
  whether its retrieval-mode query came back `degraded`
  (`mean_*_degraded_probe_rate`, §3.3) instead of silently treating it the
  same as a clean probe.
- **`attack/ssm_score/run_grounding_farming.py`**: `query_analyze()` is
  now self-throttled the same way (`MIN_QUERY_INTERVAL_S`, default 1.3s,
  overridable via `SSM_MIN_QUERY_INTERVAL_S`) and retries once on 429
  instead of letting `raise_for_status()` propagate an unhandled
  `HTTPError` — which is the literal exception shape behind the 500s you
  saw when running this script. The old fixed `time.sleep(0.2)` in the
  `ROUNDS` loop is removed in favor of the shared throttle (it was too fast
  on its own, and redundant once the throttle lives inside
  `query_analyze()` itself).

### 4.4 Correction made mid-fix, disclosed rather than left silent

An early version of this fix assumed each `/query` call multiplies load
**3x onto every individual data source** (reasoning from a paraphrased
description in `problems/safin_faced_problems/`, not from the code
directly) and set both new throttle constants to `3.3s` on that basis.
Re-reading `query_data_sources()` in `drag_llm_service/app/server.py`
directly shows this is not how it works: the loop issues **one** HTTP
request to **each** of the three distinct data sources per external call —
so each individual source's own rate counter tracks the external call rate
one-to-one, not tripled. The constants were corrected to `1.3s`, matching
this project's own already-validated precedent
(`run_attack.py`'s `QUERY_DELAY=1.3s`, `live_network.py`'s `1.1s`) instead
of the incorrect 3x-derived figure. All prose/comments citing the "3x
fan-out" reasoning were corrected alongside the constants. Mentioned here
so the numbers in the code aren't taken as unreviewed.

## 5. What was changed — file list

| File | Change |
|---|---|
| `drag_llm_service/app/server.py` | New `_post_data_source_with_retry()` (retry-with-backoff on 429/5xx); `query_data_sources()` now returns `(candidates, source_status)`; `/query` gained additive `no_retrieval`, `sources_used`, `degraded` request/response fields; `_query_analyze()`'s duplicate fan-out loop uses the same retry helper and now also tracks `source_status`/`degraded`, surfaced in its final result dict. |
| `attack/Mia_attack/mia_attack.py` | New `_query_llm_raw()` (throttled + retrying, supports `no_retrieval`); `_query_llm()` now wraps it, unchanged signature; new `_calibrated_decision_score()`; `_probe_documents()` now computes the no-RAG baseline and calibrated score per probe, returns two new diagnostic lists; `_compute_metrics()`/`_print_summary()`/`run()` updated for the new fields; module docstring documents Revision 10 in full. |
| `attack/Mia_attack/certainty_weight_ablation.py` | Fixed to unpack `_probe_documents()`'s new 7-tuple return and reuse the calibrated composite directly (rather than re-deriving it from the now-stale raw match rate), so it can't silently drift out of sync with the production formula. |
| `attack/Mia_attack/run_ablation_eval.py`, `attack/Mia_attack/tune_weights.py`, `attack/Mia_attack/contrastive_probes.py` | **Not behaviorally changed** — each still computes the raw, uncalibrated `decision_match` via its own independent probe loop. A disclosure note was added to each pointing at this document and flagging that they have not been migrated to the calibrated scoring (see §6). |
| `defense/mia_defense/mia_defense.py` | **Not behaviorally changed** — same disclosure note added; this file's own composite/docstring were already stale before this pass (citing 3rd-revision weights, not the actual 7th-revision `DECISION_WEIGHT=1.0`), which is now also flagged rather than silently left as-is. |
| `attack/ssm_score/run_grounding_farming.py` | `query_analyze()` is now throttled + retries once on 429; the old unthrottled `measure_accuracy()` loop and the too-fast fixed `0.2s` round-loop sleep both now go through the same throttle. |

Every changed `.py` file was syntax-checked with `python3 -m py_compile`
(parses and byte-compiles only — does not execute any attack, network
call, or business logic) and all compiled cleanly. Nothing was run beyond
that.

## 6. What this pass does NOT do (disclosed, not hidden)

- **The production weights were not re-validated against the calibrated
  signal.** See §3.3's "Disclosed, not yet re-validated" note. Recommended
  next step: re-run `tune_weights.py`'s dev/test grid search using the
  calibrated composite once the live stack is up.
- **Four other files that independently compute `decision_match`
  (`run_ablation_eval.py`, `tune_weights.py`, `contrastive_probes.py`,
  `defense/mia_defense/mia_defense.py`) were flagged, not migrated**, to
  avoid changing several large, previously-validated files' behavior blind
  in the same pass as the core fix, without being able to run anything to
  check the changes. `defense/mia_defense/mia_defense.py` in particular is
  a large (~900 line), carefully-validated six-world attack/defense
  comparison — migrating its composite is real follow-up work, not a
  same-pass drive-by edit.
- **No PubMedQA-side redesign.** The dataset's yes/no/maybe ground truth
  was not changed to a continuation-style probe (the paper's own strategy,
  §1) — that would be a corpus/probe-generation change, out of scope here.
  Revision 10's live calibration is the complementary fix chosen instead.
- **The rate-limit fix was not load-tested.** The constants (1.3s pacing,
  one retry with `Retry-After` honored) are corrected to match this
  project's own already-validated precedent, but this exact code path
  (the new `_post_data_source_with_retry()` helper, the new `no_retrieval`
  bypass, the new throttled MIA/SSM clients) has not been exercised
  against a live stack this pass.
- **No fresh AUC numbers.** This document describes a mechanism fix, not a
  new experimental result. The next live run (with Docker/Hardhat up, per
  `.claude/CLAUDE.md`'s environment checklist) is what will show whether
  the calibrated composite's AUC differs materially from Revision 7's
  0.620, and whether the 429 rate actually drops under the corrected
  pacing.

## 7. Suggested next steps

1. Bring up the stack (Docker Compose + Hardhat node + deployed contract,
   per `.claude/CLAUDE.md`'s pre-flight checklist) and run a clean baseline
   before anything else, as already required.
2. Re-run `attack/Mia_attack/mia_attack.py` at a modest scale first (a few
   seeds, not the full multi-seed sweep) to confirm: (a) the 429 rate is
   near zero under the corrected 1.3s pacing, and (b)
   `mean_*_pretraining_baseline_match` / `mean_*_degraded_probe_rate` look
   sane before trusting a full run's `auc_roc`.
3. Re-run `attack/ssm_score/run_grounding_farming.py` the same way and
   confirm the 429/500s reported in `problems/safin_faced_problems/` are
   gone.
4. Once both look clean, re-run `tune_weights.py`'s dev/test grid search
   against the calibrated signal (§6) before treating a new AUC number as
   validated the way Revision 7's was.
5. Decide whether to migrate `defense/mia_defense/mia_defense.py` and the
   three other flagged diagnostic scripts (§6) to the calibrated scoring,
   as a separate, reviewed follow-up pass.
