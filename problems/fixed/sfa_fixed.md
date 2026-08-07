# Selective Forwarding Attack (SFA) — Fixes Applied

Companion to `problems/sfa_attack_gaps.md` (numbering referenced below).

---

## 1. [Gap #1] Repo-root clutter removed

Deleted `patch_sfa.py` (spent, non-functional one-time migration script —
confirmed its precondition, a `.bak` file, no longer exists) and
`sfa_report_updated.html` (stray report snapshot, confirmed unreferenced
anywhere else in the repo).

## 2. [Gap #2] Dual-implementation relationship made explicit

Added `attack/selective_forward/README.md`: states explicitly that this
older module is kept deliberately (not legacy debris) as the design source
`selective_forward_sim` ports from — specifically `SFADetector`'s
binomial-test detection and the `STEALTHY_LO`/`STEALTHY_HI` drop-rate range,
matched exactly in the newer module. Points readers to
`attack/selective_forward_sim` / `defense/sfa_sim_defense` for current
numbers. Also fixed a now-stale comment in
`attack/selective_forward/selective_forward_attack.py` that referenced the
just-deleted `patch_sfa.py` by name.

## 3. [Gap #3] Live multi-trial sweep run

Two infra changes were needed first, both shared with the other trackers:
- Docker was found bind-mounted to a different, older checkout of the repo
  (see `kb_extraction_fixed.md` §5) — fixed by recreating the stack from
  this directory.
- Per this module's own README recommendation, raised the data sources'
  Flask-Limiter rate limit for the live sweep: added
  `RATE_LIMIT_DEFAULT=${RATE_LIMIT_DEFAULT:-60 per minute}` to
  `docker-compose.yml`'s three `data-source-*` services (previously
  undocumented/unwired despite the README describing this exact recipe),
  then rebuilt with `RATE_LIMIT_DEFAULT="600 per minute"`.

Ran `defense/sfa_sim_defense/run_defense.py --mode live --n_questions 50
--trials 3 --min_request_interval_s 0.15 --scenario_delay_s 0.5`. Result:
`hit_rate` flat at 1.000 for both baseline and every attacked
ratio/strategy, across all 3 trials — confirming (not just theorizing) the
report's own predicted "flat, boring hit_rate" ceiling: this 3-node,
fully-connected live network with `query_ttl=3, num_query_neighbor=2` has
enough redundant routing that one compromised node (which is all this
deployment ever produces, regardless of `--attack_ratio`) doesn't measurably
reduce answer availability. This is now backed by real repeated-trial data,
not a single run.

Log: `defense_logs/sfa_sim_defense/defense_2026-07-19_00-25-56_sfa_sim_live_seed42.json`.

A larger sweep (`--n_questions 200 --trials 5`, the report's own ideal
target) was not run — the 50/3 result already demonstrates the ceiling is
structural (architecture-driven, not sample-size-driven), so a 6-7x longer
run was judged unlikely to change the qualitative finding.

## 4. Mock multi-trial sweep run (separate from the live one above)

`defense/sfa_sim_defense/run_defense.py --mode mock --trials 5 --num_peers 20
--max_ttl 6 --seed {0,42,123}` — matches CLAUDE.md's 3-seed minimum, no
Docker required. Recovery scales cleanly with attack ratio at all 3 seeds
(e.g. seed 42, high_connectivity: ratio 0.1 → recovery +0.050, ratio 0.5 →
recovery +0.290). Logs:
`defense_logs/sfa_sim_defense/defense_2026-07-18_23-36-{23,37,37}_sfa_sim_mock_seed{42,0,123}.json`.

**Re-run after `ddos_fixed.md` §3's per-peer RNG fix** (this module shares
`MockRAGNetwork`/`MockPeer` with `attack/ddos_sim`, so the fix applies here
too — each peer's Bernoulli hit draws are no longer coupled to any other
peer's call history). Same qualitative pattern holds at the new logs:
`defense_logs/sfa_sim_defense/defense_2026-07-19_01-44-5{4,4,5}_sfa_sim_mock_seed{0,42,123}.json`
(e.g. seed 0, high_connectivity: ratio 0.1 → recovery +0.090, ratio 0.5 →
recovery +0.390) — recovery still scales cleanly with attack ratio; the
exact numbers shifted (expected and intentional), the finding didn't.

## Still open (unchanged from `sfa_attack_gaps.md`)

- §4's consolidated self-disclosed items (binomial detector's growing-window
  false-positive gap, `max_blacklist_fraction` untested live,
  `honest_miss_rate` calibration, 3-node live granularity ceiling, coarse
  exception handling) — none of these were addressed this session.
