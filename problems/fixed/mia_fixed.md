# Membership Inference Attack (MIA) — Fixes Applied

Companion to `problems/mia_attack_gaps.md` (numbering referenced below).

---

## 1. [Gap #1] Stray legacy artifacts removed

Deleted `mia_report.html` (repo root, confirmed unreferenced by any code)
and the empty leftover directories `logs/mia_squad/` and `logs/kb_squad/`
(pre-PubMedQA-migration debris, part of the same repo-wide pattern flagged
across the KB/SFA trackers).

## 2. [Gap #2] `attack/mia` legacy duplicate — already staged for deletion

No action needed this session — confirmed the `git rm -r attack/mia/`
already staged in git status (from before this session) is exactly the
right call: that module's cross-domain non-member contamination bug
(sampling members from PubMedQA-migrated data, non-members from
still-SQuAD data) would have produced a misleadingly inflated AUC.

## 3. [Gap #3] Non-determinism root cause — investigated live, evidence points away from "inherent LLM randomness"

The report's top-priority open item was whether run-to-run AUC swings
(0.6432 → 0.5440 → 0.3968 across identical live sessions) reflect genuine
LLM decoding non-determinism, or something else. This session found and fixed
two separate, real Docker deployment bugs affecting the shared `llm-service`
(see `ddos_fixed.md` §5 and `kb_extraction_fixed.md` §6): the container was
at different points running (a) a stale bind-mounted dataset from a
different checkout, and (b) an `open_model.py` build even older than the
report's own partial role-token fix, producing literal leaked tokens like
`{"response":"system\nno"}`. Critically, **MIA's own report cites exactly
this symptom** ("leaked chat-template role tokens and wildly variable
response lengths") as its evidence for non-determinism.

Direct test performed against the now-fixed, verified-stable `llm-service`
(the exact `/query` endpoint and `temperature=0.0`/no-override config MIA's
own `mia_attack.py:376` uses):
- 6 sequential identical calls in one session: byte-identical output every
  time.
- 3 independent full process restarts (fresh vLLM engine init, fresh CUDA
  graph compilation each time — the closest reproducible proxy to MIA's
  "separate live invocations"), same 3 test questions each time: **identical
  output across all 3 restarts** (`yes`/`yes`/`no` every time).
- Flask runs single-threaded (`app.run()`, no `threaded=True`) — concurrent
  requests queue rather than get batched together, so vLLM's continuous-batching
  numerical-composition-dependence (a real, separate phenomenon) doesn't
  even come into play in this specific deployment's request path.

**Conclusion (evidence-based, not just reasoned):** for this deployment's
actual generation config, decoding is deterministic session-to-session, at
least for short yes/no/maybe-style answers. The non-determinism the report
observed is much more plausibly explained by this repo's demonstrated
Docker-deployment fragility (silently running different code/data across
sessions — the *same* failure mode found and fixed twice elsewhere this
session) than by inherent LLM sampling randomness. This reframes, but does
not fully close, Gap #3 — see "Still open" below.

## 4. [Gap #5] `rerank_with_reliability` interaction — settled for all trackers

Read live on-chain scores directly (`DragScoresClient.get_scores_batch`):
`source_0`, `source_20`, `source_100` are all at `10000.0` — the initialized
baseline, unchanged. Confirmed in `drag_llm_service/src/retriever/reranker.py`:
`_minmax()` on a constant array returns all-zeros, so
`reliability_norm` contributes exactly 0 to every candidate's `rerank_score`
regardless of `reliability_weight`. **This is a code-confirmed fact, not a
reasoned inference**: `rerank_with_reliability` has had zero effect on any
result reported by MIA, KB extraction, or DDoS so far, since none of their
`/query` traffic ever writes a score update (confirmed: only `/query_analyze`
does, and its only caller anywhere in the repo is a standalone manual smoke
test).

## 5. [Gap #4] `CERTAINTY_WEIGHT=0` ablation run on the production formula itself

New `attack/Mia_attack/certainty_weight_ablation.py`. Rather than re-running
the attack twice (2x the live LLM cost), it calls
`MIAAttack._probe_documents()` **once** per seed (the expensive, real-LLM
part) and recomputes AUC-ROC from the *same* per-document signals at two
weight settings — production (`CERTAINTY_WEIGHT=0.0`) vs. certainty restored
(`CERTAINTY_WEIGHT=0.2`, still satisfies the module's own gate invariant
`DECISION_WEIGHT(1.0) > SIM+CERTAINTY+LEN`) — so both numbers come from
identical underlying probes, no resampling confound. Run at 3 fresh seeds
(4001/4002/4003, not previously used anywhere in this project's dev/test
history):

| seed | AUC (production, certainty=0) | AUC (certainty=0.2) |
|---|---|---|
| 4001 | 0.7000 | 0.6920 |
| 4002 | 0.5400 | 0.5224 |
| 4003 | 0.6000 | 0.6072 |
| **mean** | **0.6133** | **0.6072** |

Delta: **-0.0061**. Adding certainty back into the production formula does
not help (marginally hurts) on fresh held-out seeds — this directly confirms
Revision 7's decision to zero `CERTAINTY_WEIGHT`, now checked against the
actual production composite rather than only the separate `ABLATION_*`
diagnostic variables.

## 6. [Gap #4] Length-normalization floor — validated, and found to be necessary for a different reason than assumed

New `defense/mia_defense/validate_length_floor.py`. `calibrate_target_length()`'s
`floor=100` docstring attributes the need for a floor to leaked role tokens
producing degenerate ~10-char responses — a bug this session fixed and
verified (§3 below / `ddos_fixed.md` §5). The natural follow-up: does the
floor still matter now that the leak is fixed?

Collected 50 real live responses (25 members + 25 non-members, fresh seed
5001) from the now-fixed `llm-service`: **every single response was a bare
"Yes"/"No"/"yes"** (length 2-3 chars, min=2, median=3.0, max=3) — clean, not
truncated, not leaking any role token. `calibrate_target_length(floor=100)`
→ 100; `calibrate_target_length(floor=0)` → 3.

**The floor is still measurably necessary — but for a different, and
arguably stronger, reason than its docstring currently states.** It's not
compensating for a leaked-token bug (that's fixed); PubMedQA yes/no/maybe
answers under this system's own 3-word-max system prompt are *genuinely,
inherently* 2-3 characters long. Any unfloored calibration would collapse to
≈3 chars regardless of the role-token bug's status. The floor's docstring
should be updated to reflect this (not done this session — a doc-only
change, low priority relative to the empirical finding itself).

## 7. [Gap #4] Consistency-score signal — replicated at 3 fresh seeds, and the result reframes §3 and the signal itself

The report flagged this as "the most promising and least replicated lead"
(AUC 0.49-0.75 across only seeds 0/1/42) but tangled with the non-determinism
finding. Ran 3 fresh seeds (6001/6002/6003, `run_consistency_pilot.py`,
n_probes=5) against the now-fixed, verified-deterministic `llm-service`:

| seed | mean_member_consistency | mean_non_member_consistency | AUC |
|---|---|---|---|
| 6001 | 1.00 | 1.00 | 0.50 |
| 6002 | 1.00 | 1.00 | 0.50 |
| 6003 | 1.00 | 1.00 | 0.50 |

**Exactly chance, every time, mechanically.** This is the clearest possible
confirmation of §3's conclusion: the consistency-score signal's entire
premise is that repeated identical queries vary *more* for non-members than
members. Under `temperature=0.0` deterministic decoding (verified stable in
§3), repeating the *same* question 5 times **always returns the same
answer** for both groups — consistency=1.0 for everyone is a mechanical
certainty of greedy decoding, not a measurement of anything. The original
0.49-0.75 spread across seeds 0/1/42 was almost certainly measuring the same
Docker/deployment instability found and fixed elsewhere this session (stale
images/data producing genuine but spurious response variation across calls),
not a real membership signal. **This signal is structurally non-viable on
this deployment as currently configured** — it would need actual sampling
randomness (`temperature > 0`) to have any chance of carrying information,
which would reopen the exact "is this genuine stochasticity or environment
instability" question §3 worked to close.

## Still open

- The determinism test in §3 and the ablations in §§5-7 all use short,
  single-word/yes-no-maybe-style answers. MIA's paraphrase-probe prompts
  weren't separately tested for longer, more open-ended generations, where
  more token-level decoding surface exists even under greedy decoding.
- A full re-run of MIA's own primary-composite seed sweep (13+15 seeds) against
  the now-fixed `llm-service`, to see whether the originally-reported AUC
  swings (0.6432/0.5440/0.3968) shrink or disappear, was not performed —
  §§5-7 are targeted, cheaper ablations, not a full replication.
- Held-out composite CI at n=10 (95% CI including chance): this concern
  applies to a composite formula (linear blend of decision/sim/certainty/len)
  that Revision 7 no longer uses in production — the current production
  composite is decision_match alone, which already has a passing fresh-seed
  CI (mean 0.620, [0.546, 0.695], excludes chance). Not re-run, since it
  would be re-measuring a retired formula.
- Content-defense robustness to a differently-designed detector (e.g. a
  classifier for hedged phrasing) — not addressed this session.
- The Docker volume-mount mismatch reference in the report's own §12.9 is
  now understood in far more detail (see `kb_extraction_fixed.md` §5), though
  this session's fix was to a *different* checkout-mismatch instance of the
  same underlying fragility, not the exact one §12.9 originally diagnosed.
