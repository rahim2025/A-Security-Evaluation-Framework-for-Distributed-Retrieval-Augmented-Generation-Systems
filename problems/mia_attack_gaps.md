# Membership Inference Attack / MIA (`attack/Mia_attack/`) — Known Gaps & Limitations

Tracking doc for issues found reviewing the MIA module
(`attack/Mia_attack/` + `defense/mia_defense/`), cross-checked against
`reports/MIA_Security_Analysis_Report.md` (seven revisions deep) and
`Reliable-dRAG.md`.

**Context vs. the other three trackers: this is the most mature module in
the repository by a wide margin.** Seven successive report revisions, each
one testing (not assuming) the previous revision's open questions, three
real bugs self-caught during the module's own testing (a synonym leak in
the content-defense paraphrase templates, a degenerate length-calibration
target, a stale composite that was diagnosed-but-not-fixed across two prior
revisions before Revision 7 actually re-tuned it under a genuine train/test
split), and — most importantly for this cross-module tracker — **this is
the first module where the CLAUDE.md multi-seed requirement is actually
satisfied**: 13 dev seeds + 5 held-out fresh test seeds, with confidence
intervals, not a single-seed point estimate. Treat this module's report as
the template the other three trackers' "run more seeds" action items are
pointing toward.

**God Mode check: passed.** `obfuscate_decision_content()`'s own detector
(`_detect_decision_token`) is explicitly built to use only the response
text, never the gold label — verified directly in code (§13.5 of the
report, confirmed by reading `defense/mia_defense/mia_defense.py`). The
attack/defense comparison also correctly issues **one** live LLM call per
document and applies all six defense transforms to that same cached raw
response (confirmed directly in `MIADefenseEvaluator.run()`,
`defense/mia_defense/mia_defense.py:490,556-557`) — so the six-world
comparison is apples-to-apples per document, not contaminated by a fresh
non-deterministic call per world.

**KB-style oracle-leakage check: does not apply here, and that's correct
by the attack's own definition, not an oversight.** MIA's "best-of-N
probing" step picks the response with highest similarity to the true
context text as a document's representative probe. This looks superficially
like KB extraction's ground-truth-filtered probe set, but it isn't the same
problem: MIA's threat model (§1.3 of the report) is explicitly "attacker
already holds the candidate document/context and is testing whether *that
specific* text was used" (the standard Shokri et al. 2017 framing) — unlike
KB extraction, where the attacker doesn't know the target's content and is
trying to discover it. Having the candidate text in hand is the premise of
membership inference, not a hidden advantage. No action needed.

---

## 1. [NEW] Stray root-level legacy artifacts (third instance of this pattern)

- **`mia_report.html`** (repo root, dated 2026-07-02) — a standalone HTML
  report snapshot from the same early period as the now-deleted `attack/mia`
  module (pre-PubMedQA-switch, likely reporting the inverted SQuAD-era
  results per report §12.1). Not referenced by any code; only graphify's
  auto-generated index lists it (that's automatic repo indexing, not a real
  dependency).
- **`logs/mia_squad/`** — an empty leftover directory, named for the
  SQuAD-based methodology the project moved off of (module docstring:
  "Why PubMedQA instead of SQuAD").

**Consequence:** same pattern already flagged for the other two modules —
`problems/sfa_attack_gaps.md` §1 (`patch_sfa.py` +
`sfa_report_updated.html`) and, less directly, `problems/kb_extraction_gaps.md`
§2 (dead `kb_extraction_attack.py`). Three modules in a row have left a
stray HTML report snapshot and/or empty legacy log directory at repo root
or under `logs/`. This is now a repo-wide pattern worth a single cleanup
pass rather than three separate ad-hoc ones.

**Action:** delete `mia_report.html` and `logs/mia_squad/` (confirmed
unreferenced), and — since this is the third occurrence — consider one
repo-wide pass to find and clear any other pre-PubMedQA-migration debris
at root/`logs/` before the thesis writeup references the repo structure.

---

## 2. Positive note: `attack/mia` cleanup is the model the other modules should follow

Unlike KB extraction's silently-diverging dead file or SFA's still-live
sibling implementation, this module's stale duplicate
(`attack/mia/mia_attack.py`) was **found, diagnosed as having a genuine
correctness bug (cross-domain non-member contamination — sampling
"members" from PubMedQA-migrated `sources_0.jsonl` but "non-members" from
still-SQuAD `sources_20/100.jsonl`, which would have produced a misleadingly
inflated AUC had anyone run it), confirmed unused by every other consumer,
and deleted outright** (`git rm -r attack/mia/`, staged in the current git
status), rather than left in place. This is exactly the treatment
recommended for KB extraction's `kb_extraction_attack.py`
(`problems/kb_extraction_gaps.md` §2) and SFA's `patch_sfa.py`/legacy
`selective_forward` pairing (`problems/sfa_attack_gaps.md` §§1-2). Not a
gap — recorded here so the cross-module tracker has one clear positive
example to point back to.

---

## 3. Blocking limitation, already self-disclosed at the highest priority: LLM run-to-run non-determinism

The report's own §12.13/§14 already identify this as the **single largest
limitation in the project's history**, ranked above even the held-out-seed
count: the same seed's undefended composite AUC swung from 0.6432 to
0.5440 to 0.3968 across three separate live invocations of identical code
this session — not seed-to-seed noise, but call-to-call noise in the
underlying LLM's decoding (confirmed by direct evidence of leaked
chat-template role tokens and wildly variable response lengths on repeated
identical queries). The report is explicit that every confidence interval
computed so far (including the one supporting `decision_match`'s
generalization claim) **understates true uncertainty**, since it captures
seed-to-seed variance only, not run-to-run variance.

This is carried into this tracker, not as a new finding, but because it is
the module's own top blocking recommendation and shouldn't be lost among
the less severe items below. **This also has a cross-module implication
worth flagging explicitly**: if `drag_llm_service`'s decoding is genuinely
non-deterministic session-to-session, that same non-determinism is a
plausible confound for *any* other module in this repo that queries the
live LLM service and reports a single-run AUC/F1/similarity number
live-mode DDoS's `run_live_evaluation.py`, KB extraction's Phase C, SFA's
live PubMedQA questions all call the same service. None of those reports
currently test for this the way MIA's §12.13 does.

**Action:** root-cause the non-determinism (temperature/sampling config,
chat-template leakage) before treating any live-mode AUC/similarity number
from *any* module in this repo as fully trustworthy — not just MIA's.
Consider recommending the other three trackers add a "was this checked for
run-to-run non-determinism" note given this precedent.

---

## 4. Other self-disclosed open items, consolidated here for the cross-module tracker

- **Primary composite's held-out generalization is still formally
  unresolved at n=10** — 95% CI includes chance; only `decision_match`
  alone (now the sole production weight, Revision 7) has a CI that excludes
  it. Revision 7's 5 fresh-seed confirmation (mean 0.620, CI [0.546, 0.695])
  is a real, separate, positive result for `decision_match` specifically.
- **`CERTAINTY_WEIGHT` has never been shown to contribute positively in any
  revision** and is already zeroed in the production composite (Revision 7),
  but a direct `CERTAINTY_WEIGHT=0` ablation *on the production formula*
  (as opposed to the separate diagnostic ablation composite) is still an
  open recommendation (§14).
- **Consistency-score signal (n_probes=5) is the most promising and least
  replicated lead in the project** — AUC spread 0.49–0.75 across only 3
  seeds, entangled with the non-determinism finding above. Not yet
  replicated at full scale (report's own Recommendation, §14/§16).
- **`obfuscate_decision_content()` only tested against this project's own
  two attacker variants** (adaptive, semantic) — untested against a
  differently-designed detector (e.g., a classifier for hedged phrasing).
- **Length-normalization floor (100 chars) is itself an unmeasured
  constant**, chosen to stop a specific observed failure mode rather than
  derived from a stable measurement.
- **Docker volume-mount mismatch diagnosed, not fixed** (§12.9) — a
  deliberate scope boundary (fixing it requires restarting a live, shared
  service), not an oversight.
- **Single model, single corpus** — unchanged across all seven revisions.

---

## 5. [NEW] Untested interaction with reliability-weighted reranking — the module most exposed to this, of the four reviewed

`drag_llm_service/configs/config.yaml` has `rerank_with_reliability: true`
and `reliability_weight: 0.5` on by default — every `/query` call blends
on-chain R_i into which retrieved passages reach the LLM's prompt
(`server.py:581-596`). `reports/MIA_Security_Analysis_Report.md` never
mentions this once, despite otherwise being the most rigorous of the four
reports about isolating confounds.

**Why this module is the most exposed.** Unlike DDoS (headline metric
bypasses `drag_llm_service` entirely) or KB extraction (only Phase C
touches it), **100% of MIA's signal, across all seven revisions, comes
from calls to `drag_llm_service`'s `/query`.** Every AUC-ROC number in the
report was generated from responses that went through reliability-weighted
reranking.

**Why this specifically matters for §12.13's non-determinism finding.**
The report attributes the observed 0.6432→0.5440→0.3968 same-seed AUC
swing (§12.13) solely to LLM decoding non-determinism and chat-template
leakage. That diagnosis never considered or ruled out a second candidate
explanation: if on-chain R_i drifted between those three separate live
sessions (from any other query traffic hitting the shared Hardhat chain
in between — see below), the reranking composition feeding the LLM would
differ session to session too, independent of decoding randomness. Both
mechanisms would produce the same symptom (the same seed's documents
producing different final responses across runs), so the current
diagnosis may be incomplete, not necessarily wrong.

**Mechanistic reasoning for why the effect could plausibly be larger here
than for DDoS/KB:** MIA's signal depends on subtle content-selection at
the margin — specifically, whether the *correct* document for a member
question wins reranking, and whether *any* source_0-domain content gets
pulled into a non-member's context despite no true passage existing
anywhere in the corpus for it. A reliability-driven pull toward source_0's
content pool for non-member queries (if source_0 carries a different score
than sources_20/100) could systematically make non-members look more
"answerable," narrowing the member/non-member gap for reasons unrelated to
true membership — plausibly contributing to exactly the kind of noise the
report spent two revisions chasing.

**What would settle this:** current on-chain R_i for source_0 vs.
sources_20/100. Not checked — Docker/Hardhat was down at review time and
live verification was deferred by request. One relevant fact already
established: none of the four attacks' own `/query` traffic writes score
updates (only `/query_analyze` does), so MIA's own seven revisions of
querying could not have caused this drift themselves — if scores have
moved from baseline, it would have to be from something external, and the
only caller of `/query_analyze` found anywhere in this repo is
`drag_llm_service/test_service.py`'s standalone manual smoke test.

**Action:** highest priority of the three "reliability-reranking" notes
added across the DDoS/KB/MIA trackers, given MIA's full exposure and its
own already-documented reproducibility crisis. Check current on-chain
scores next time the stack is up; if source_0 differs meaningfully from
sources_20/100, re-run a small seed subset with `rerank_with_reliability:
false` as a second, independent test of §12.13's swing — this is a cheap,
decisive experiment given the module already has the infrastructure to
re-run any seed.

---

## Summary table

| # | Gap | Severity | Status |
|---|---|---|---|
| 1 | Stray `mia_report.html` + empty `logs/mia_squad/` at root/logs | Low-Medium | **New** — open, same pattern as SFA/KB trackers |
| 2 | `attack/mia` legacy duplicate — cleanly deleted, not a gap | — | **Resolved** (positive example for the other trackers) |
| 3 | Live LLM service is non-reproducible run-to-run, even at a fixed seed — understates every CI in this report and is a plausible confound for every other module's live-mode numbers | High (Blocking, per the report's own priority) | Self-disclosed — open, highest-priority item in the whole repo |
| 4 | Held-out composite CI still includes chance at n=10; `CERTAINTY_WEIGHT=0` not tested on production; consistency-score unreplicated; content defense untested against novel detectors; unmeasured length floor; Docker mount unfixed | Medium (various) | Self-disclosed in `reports/MIA_Security_Analysis_Report.md` — open, consolidated here |
| 5 | Untested interaction with `rerank_with_reliability` — MIA is 100% exposed (every AUC number goes through it), and it's a plausible second explanation for §12.13's non-determinism finding | Medium-High | **New** — open, highest priority of the three reliability-reranking notes; cheap to verify, not yet checked (Docker was down at review time) |
