# KB Extraction — Fixes Applied

Companion to `problems/kb_extraction_gaps.md` (that file's numbering is
referenced below). This file records what was actually fixed/run, not a
re-statement of the gaps themselves.

---

## 1. [Gap #2] Dead legacy file removed

Deleted `attack/kb_extraction/kb_extraction_attack.py` (confirmed unused,
methodologically divergent from `run_attack.py`).

## 2. [Gap #1] Gray-box vs. blind/cold-start probe comparison added

`attack/kb_extraction/run_attack.py`:
- `load_probe_sets(..., blind: bool = False)` — when `blind=True`, probes are
  sampled from the *entire* matched-question pool for that source's domain,
  without the `ctx in ground_truth` pre-filter that guarantees a hit.
- `main()` now runs a new **Phase B'** with `blind=True` immediately after
  the existing gray-box Phase B, and reports both numbers side by side
  (console table + `phase_b_auth_blind` / `per_source_extraction_blind` in
  the JSON report), instead of only ever reporting the gray-box ceiling.

## 3. [Gap #6] Rate-limit contamination guard (not just a warning)

`defense/kb_extraction_defense/run_defense.py`:
- Added `_probe_with_rate_limit_guard()` — retries a phase after a cooldown
  if it hit HTTP 429s, and **aborts** (raises) rather than silently
  returning a contaminated `extraction_rate` if still rate-limited after
  `--rate_limit_max_retries` (default 1, `--rate_limit_cooldown_s` default 65s).
- Follow-up bug found and fixed in this same change: the first version of
  the guard reused the *same* `QueryDiversityThrottle` instance across a
  retry, so a rate-limited first attempt's partial query count leaked into
  the retry's window (observed: `blocked=54/54`, i.e. 100%, and
  `flagged_at_query` exceeding the actual probe count). Fixed by rebuilding
  the throttle fresh on every attempt via a `make_kwargs()` factory passed
  into the guard.

## 4. [Gap #5] Defense threshold recalibrated against a measured baseline

New `defense/kb_extraction_defense/calibrate_thresholds.py`: since no real
user telemetry exists, "legitimate" is modeled explicitly (not assumed) as a
session asking `window_size` questions about `k ∈ [1,3]` topics of interest,
contrasted against the attacker's own real sampling strategy (uniform across
the entire matched-question pool, i.e. what `run_attack.py` actually sends).

Measured on source_1 (10 real topics, `window_size=30`, 2000 simulated
sessions each side):
- Legitimate: p50=2, p90=3, p95=3, p99=3 distinct topics touched.
- Attacker (real sampling strategy): p50=9, p90-p100=10.
- Recommended `max_topics_per_window` = 4 (ceil(p99 legitimate) + 1) —
  clean separation from the attacker's 9-10, unlike the previous
  **unmeasured** `max_topics_per_window=5` used to produce the 82.5% block
  rate in `reports/kb.md`.

Re-run with the calibrated threshold (post-infra-fix, see below):
`extraction_rate` 0.390 → 0.134 (blocked 40/54 = 74.1%) — a real,
non-contaminated reduction, in
`defense_logs/kb_extraction_defense/defense_2026-07-19_00-07-11_kb_extraction_throttle.json`.
(The `...00-04-20...` log in the same directory is the throttle-state-leak
run superseded by the fix above — kept for the record, not a valid result.)

## 5. [NEW, not in the original tracker] Docker bind-mount pointed at a different, older checkout

Root cause of `sources_20`/`sources_100` extraction_rate reading exactly
0.000 under a live run (both gray-box *and* blind): the running Docker
containers were bind-mounted to
`C:\Users\user3\Desktop\reliable de-rag rahim\Reliable-dRAG-main\` — a
separate, older clone of the same GitHub repo (last commit 2026-07-14),
still on the pre-migration 3197-record TriviaQA-era `sources_20/100.jsonl`,
while this repo's ground-truth loader read the current, 500-record migrated
files. `source_0` coincidentally matched (500 records both sides), which is
why only sources_20/100 showed the failure and source_0 didn't.

Restarting the containers alone does *not* fix a bind-mount source path —
only recreating them does. Fixed by: `docker compose down` from the old
checkout, `docker compose up -d` from this one. Confirmed via
`docker inspect ... --format '{{json .Mounts}}'` before/after, and via
record counts read both from the host and via `docker exec ... cat`.

This also resolves **[Gap #4]** (Phase C's 75% error rate on n=5): with the
correct, smaller corpus loaded, Phase C now scores 20/20 probes with zero
errors in every seed's log, vs. 5/20 before.

## 6. [NEW] `llm-service` was running a build even staler than the pre-existing partial fix

While re-verifying Phase C output, live responses showed literal leaked
role tokens (`{"response":"system\nno"}`) — `_strip_leaked_role_tokens`
wasn't just failing, it wasn't present in the running container at all. The
image had never actually picked up either the pre-existing partial fix *or*
this session's `open_model.py` edit (see `ddos_fixed.md` for the header-id
regex fix itself). Root-caused and fixed together with the DDoS tracker
since it's the same shared service — see `ddos_fixed.md` §"llm-service
image" for the full story (a `--no-cache` rebuild attempt then broke vLLM
with `RuntimeError: UVA is not available`; recovered via the known-good
older image + `docker cp` of the current `app`/`src`/`configs`).

Once fixed, **Phase C was re-run for all 3 seeds** (see below) since the
earlier same-session Phase C numbers were computed against the broken image.

## 7. Live multi-seed run, now on correct data + fixed model (closes [Gap #3])

Re-ran `attack/kb_extraction/run_attack.py` live at `RANDOM_SEED={0,42,123}`
after both infra fixes above:

| seed | source_0 (pubmedqa) | source_1 (squad) | source_2 (squad) | Phase C CRR |
|---|---|---|---|---|
| 0   | 0.402 | 0.402 | 0.356 | 1.000 (n=6 scored) |
| 42  | 0.434 | 0.390 | 0.366 | 0.300 (n=20 scored) |
| 123 | 0.422 | 0.386 | 0.396 | 0.300 (n=20 scored) |

Extraction rates are stable in the 0.35–0.43 range across all 3 seeds and
all 3 sources — consistent with the historical 37–43% claim, and now backed
by correct ground truth instead of the stale-mount 0.000s. Logs:
`attack_logs/kb_extraction/attack_2026-07-19_01-{08-57,09-50,11-58}_kb_extraction.json`.

## Still open (unchanged from `kb_extraction_gaps.md`)

- Defense threshold recalibration was only run against source_1; source_2 and
  source_0 (no topic field) weren't separately recalibrated.
- No `--seed` CLI flag on `run_defense.py` — the defense comparison itself
  still isn't multi-seeded, only the attack-only script is.
