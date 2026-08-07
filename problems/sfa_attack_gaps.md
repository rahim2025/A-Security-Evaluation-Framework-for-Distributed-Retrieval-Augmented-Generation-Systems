# Selective Forwarding Attack / SFA (`attack/selective_forward_sim/`) — Known Gaps & Limitations

Tracking doc for issues found reviewing the SFA module
(`attack/selective_forward_sim/` + `defense/sfa_sim_defense/`), cross-checked
against `reports/SFA_Security_Analysis_Report.md` and `Reliable-dRAG.md`.

**Context vs. the other two trackers:** `reports/SFA_Security_Analysis_Report.md`
is, by a wide margin, the most rigorous and self-critical of the three
attack reports reviewed so far. It documents **four real implementation
bugs found and fixed during its own development** (Phase-2 candidate-set
bug, unbounded blacklisting, an untested stealthy-attacker gap, a silently-
wrong live question loader), a live-validation attempt that failed, was
correctly re-diagnosed after an initial wrong hypothesis, and re-run
successfully, and it explicitly flags single-seed/single-trial results as
illustrative rather than final (§15) — already matching the standard this
tracker has been holding the other two modules to. Most of what would
normally be "new findings" here is already disclosed in that report. This
file therefore focuses on: (a) items the report doesn't cover (repo
hygiene, a check for the KB-style oracle-leakage pattern), and (b)
consolidating the report's own self-disclosed gaps into the same
cross-module tracker used for DDoS and KB extraction, so nothing gets lost
when acting on this backlog later.

**God Mode check: passed**, same as DDoS. Wrapper order is attack-first,
defense-second (`attack.apply()` then `defense.apply()` in
`run_defense.py`, mirroring `defense/ddos_sim_defense`'s pattern), so the
defense observes attack-induced silent drops as ordinary non-responses,
not privileged internal attack state. The attacker's "full topology /
on-chain-score visibility" advantage is an explicit, disclosed threat-model
assumption (§1.3), not a hidden one.

**KB-style probe pre-filtering check: does not apply here, and that's
correct, not an oversight.** SFA's live question loader (`_live_questions()`)
also filters PubMedQA questions down to ones matching the actual loaded
corpus, superficially similar to KB extraction's ground-truth pre-filter.
But the metric here is `hit_rate` — "does *any* peer in the network have
an answer to this question" — not attacker extraction efficiency. Filtering
to genuinely-answerable questions is *necessary* for this metric to mean
anything: an unanswerable question would trivially read as a miss
regardless of whether an attacker is present, contaminating the baseline
with noise unrelated to the attack. This is the opposite situation from
KB extraction, where pre-filtering handed the *attacker* free targeting
information. No action needed here.

---

## 1. [NEW] Repo-root clutter: a completed, now-broken one-shot patch script and a stray report snapshot

Two files sit in the repository **root** (not inside `attack/`, `defense/`,
or `reports/`), unrelated to any currently-run module:

- **`patch_sfa.py`** (33KB) — a one-time migration script that restores
  `attack/selective_forward/selective_forward_attack.py` from a `.py.bak`
  backup and appends `SFADetector`/`SFAMitigation`/`check_blockchain_status`
  classes to it via string concatenation. It has already run — the target
  file confirmed contains all three appended classes — and **its
  precondition is now gone**: `attack/selective_forward/*.bak` no longer
  exists on disk, so running `patch_sfa.py` again today would immediately
  exit with `ERROR: backup not found`. It is fully spent, non-functional
  dead weight.
- **`sfa_report_updated.html`** — a standalone, manually-styled HTML report
  snapshot, dated from the same early period as `patch_sfa.py`, sitting
  at root rather than under `reports/` or `html_reports/` where its
  siblings (`html_reports/sfa_report.html`, `html_reports/ddos_report.html`)
  live.

**Consequence:** same defensibility risk pattern as KB extraction's dead
`kb_extraction_attack.py`, but worse because these sit at repo root where
a committee member browsing the project structure is more likely to open
them first, and `patch_sfa.py` in particular could read as "this attack
module was hand-patched together with a shell script," which undersells
how much more principled the actual current module (documented so
thoroughly in `reports/SFA_Security_Analysis_Report.md`) actually is.

**Action:** delete both, or move them to a `legacy/` or `archive/` folder
with a short note, once confirmed nothing else depends on them (repo-wide
search found no other references to either file).

---

## 2. Dual implementation, already disclosed but still live in the repo — worth a decision, not just a footnote

Unlike KB extraction's silent duplication, the SFA report's own scope note
(top of the file) is upfront: *"This repository contains two independent
Selective Forwarding Attack implementations... This report focuses on the
[newer one] because it is the one with fresh, reproducible JSON evaluation
data... Where relevant, the older implementation is referenced for
contrast."* This is good practice — the ambiguity is labeled, not hidden.

That said, `attack/selective_forward/` (the older implementation, plus its
`defense/sfa_defense/` pair) is still a full, functional, undeleted module
sitting alongside the current one, and per §15's own admission: *"This
report does not evaluate the older... implementation in the same empirical
depth."* Two live, functional implementations of the same attack — one
thoroughly validated, one not — is a smaller version of the same
divergence risk flagged for KB extraction, just handled more transparently.

**Action:** decide explicitly whether `attack/selective_forward` /
`defense/sfa_defense` is (a) kept as a deliberate "sibling module" the
newer one ports concepts from (e.g. `SFADetector`'s binomial test, cited
directly in §2.6/§12.7 as the design source) — in which case it's worth
a one-line README note in `attack/selective_forward/` saying so explicitly,
or (b) retired now that its ideas have been ported into
`selective_forward_sim`/`sfa_sim_defense`. Right now it's ambiguous from
the repo structure alone which of the two is true.

---

## 3. Zero multi-trial evidence despite the module explicitly supporting it (same project-wide gap — deferred)

Every SFA attack/defense log on disk is `trials: 1`:

- All 9 files under `defense_logs/sfa_sim_defense/*.json` — confirmed
  `"trials": 1` in every one.
- `attack_logs/selective_forward_sim/` has exactly one attack-only log
  (`seed0`); the rest of that directory's files are `_ddos_sim_*`/`_sfa_sim_*`
  naming overlaps from sibling modules, not additional SFA trials.

This is the same CLAUDE.md gap tracked for DDoS (`problems/ddos_attack_gaps.md`
§1) and KB extraction (`problems/kb_extraction_gaps.md` §3) — *"Do not
report single-seed results as final."* The SFA module is actually
**ahead** of the other two here: `run_defense.py` already has a `--trials N`
flag built specifically for averaging across seeds (per §15's own text),
it's just never been invoked with `N > 1` in any log currently on disk.

**Status / why:** same reason as the other two modules — team has been
busy implementing the remaining attacks and hasn't yet run the multi-trial
sweep for SFA either. This one is the cheapest of the three to close,
since the flag already exists — it's a matter of invocation, not
implementation.

**Action:** re-run `defense/sfa_sim_defense/run_defense.py --trials 5`
(or similar) across the standard config (§8.1: 20 peers, `query_ttl=6`)
and report mean ± variance before citing the current `trials=1` numbers
as final. The live-mode equivalent additionally needs the rate-limit
fix's server/client throttle pair raised together first (already true
per §12.9/§12.10 — just needs `--n_questions 200 --trials 5+`, which the
report itself identifies as the correct next step and has not yet done).

---

## 4. Self-disclosed items carried into this tracker (not new, but tracked here for completeness)

These are already fully documented in `reports/SFA_Security_Analysis_Report.md`
with root causes, fixes, and evidence — listed here only so the cross-module
`problems/` tracker has one place a reader can scan for every attack's
open items without re-reading four full reports.

- **Binomial detector's growing window (not a true sliding window) inflates
  false positives** relative to the sibling module's design (§12.7,
  Recommendation 8) — open, tracked as future work.
- **`max_blacklist_fraction`'s over-exclusion-prevention scenario has only
  been demonstrated in mock mode** (8 peers, `peer_hit_prob=0.03`), not
  live — this 3-peer deployment's honest peers respond too reliably to
  ever approach that regime (§12.10, "what remains open"). Not closeable
  without a larger live deployment or an artificially degraded live corpus.
- **`detection_mode: binomial`'s `honest_miss_rate` must be calibrated to
  the real measured baseline, not left at its shipped default** — §12.7
  reproduced, on this project's own code, exactly the miscalibration bug
  its own theory section warns about (5/10 peers blacklisted when only 1
  was truly compromised). Recommendation 4 in the report is explicit that
  this must be re-measured per deployment, not assumed.
- **Live deployment is only 3 nodes**, so the live-mode ratio sweep
  collapses to 2-3 distinct attack severities regardless of the 5 nominal
  ratios configured (§15) — a structural ceiling on live statistical
  granularity, not fixable without more Docker replicas.
- **`LivePeer.query()`'s broad exception handling still coalesces genuine
  connection failures with content misses** — the 429/rate-limit case is
  now handled explicitly (§12.9's fix), but an unexpected 5xx or DNS
  failure still isn't distinguishable from "no relevant content" in the
  aggregate metrics (§15, last bullet).

---

## Summary table

| # | Gap | Severity | Status |
|---|---|---|---|
| 1 | Stray, now-broken `patch_sfa.py` + orphaned `sfa_report_updated.html` at repo root | Medium | **New** — open |
| 2 | Two live, functional SFA implementations (`selective_forward` vs `selective_forward_sim`) with only the newer one validated — disclosed in the report, but repo structure doesn't make the relationship explicit | Low-Medium | Partially disclosed — open |
| 3 | Zero multi-trial runs despite `--trials N` already existing | High | Deferred — known, project-wide, cheapest of the three to close since the flag already exists |
| 4 | Binomial detector false-positive gap, blacklist-cap over-exclusion scenario untested live, `honest_miss_rate` calibration requirement, 3-node live granularity ceiling, coarse exception handling | Medium (various) | Self-disclosed in `reports/SFA_Security_Analysis_Report.md` — open, consolidated here |
