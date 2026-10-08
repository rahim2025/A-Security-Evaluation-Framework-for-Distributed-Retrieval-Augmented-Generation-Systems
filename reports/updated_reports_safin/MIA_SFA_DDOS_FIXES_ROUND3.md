# MIA / SFA / DoS — Third Fix Pass (Round 2's Own Bugs + New Correctness Fixes)

**Date:** 2026-09-21, 23:55 (+06)
**Status: code changed, not executed.** Per explicit instruction this pass
("don't run attacks, just fix, i will run later on"), everything below is
static analysis, code fixes, `py_compile` syntax checks, and — where a
class's own logic could be exercised in isolation, with no Docker/Hardhat/
LLM service involved — unit tests and small standalone sanity scripts.
**Nothing was run against the live Reliable-dRAG stack, no attack or
defense evaluation was executed, and no results in any prior report were
regenerated.** Docker was confirmed available on this machine but was not
brought up: this laptop is Apple Silicon (M1, no NVIDIA GPU), and
`drag_llm_service` depends on vLLM/CUDA — live runs belong on the
project's separate "powerful machine," not here.

Scope: `current_gaps_overview.md`'s three attacks under active work — MIA,
SFA, DoS — attack **and** defense sides. Continues directly from
`MIA_SFA_DDOS_FIXES_ROUND2.md`; read that first for the DoS quorum-cap
bug, the false-block metric fix, and the SFA `honest_miss_rate="auto"`
fix, all of which are prerequisites for what's below.

---

## Headline finding

Round 2 added a unit test suite for `LiveClientDefense`
(`test_live_client_defense.py`) alongside its quorum-cap fix, but — per
Round 2's own §4, everything that pass was "reasoned from reading the
code, not confirmed by ... running it" — **the test suite itself was
never executed.** Running it this pass (`python3 -m pytest
defense/ddos_sim_defense/test_live_client_defense.py`) immediately failed
2 of 11 tests, surfacing two real bugs in Round 2's own quorum-cap
implementation. Both are fixed below, and the suite now passes 11/11.
Running unit tests is not running an attack or a live evaluation — no
network call, no Docker, no LLM — so this stayed within the "don't run
attacks" instruction while still catching real defects the static reading
in Round 2 missed.

---

## 1. DoS — two new bugs found inside Round 2's own quorum-cap fix

**File:** `defense/ddos_sim_defense/live_client_defense.py`

### 1.1 `_peer_health_score()` divides by zero when `rate_limit_capacity=0`

`tokens_frac = bucket.tokens / bucket.capacity` crashes with
`ZeroDivisionError` whenever a peer's bucket is configured with zero
capacity (a valid, if extreme, config — "never let this client send
anything to this peer"). Caught by
`test_backup_candidates_excludes_blacklisted_and_tried`.

**Fix:** capacity `<= 0` now scores as `0.0` (least healthy, correctly —
a permanently-empty bucket is the worst case, not "healthy by default").
Only `bucket is None` (no bucket at all) still scores `1.0`.

### 1.2 The quorum cap could silently let an entire genuinely-congested subset through — a self-defeat, not just an edge case

Round 2's fix made `is_peer_blacklisted(peer_id)` ask, per peer,
independently: *"are enough **other** peers already raw-blocked to hit
the cap?"* — and fail open for `peer_id` if so.

That per-peer-independent framing has a real bug: when **more peers are
raw-blocked than the cap allows**, every one of them, asked in isolation,
sees enough *other* raw-blocked peers to justify failing open for
*itself*. Confirmed directly (no live infra — a 3-peer in-process fake
network, 2 of 3 peers with exhausted rate-limit buckets, default
`max_blacklist_fraction=0.5` → cap=1):

```
peer0 (congested): False   peer1 (congested): False   peer2 (healthy): False
```

**Zero of the three genuinely-differentiated peers were reported
blocked** — both congested peers escaped detection simultaneously,
because each one's independent check saw "the other guy" already
counted as raw-blocked and concluded the cap was already spent. This is
worse than having no cap at all: the router would keep sending real
traffic to both struggling peers as if nothing were wrong, which is
exactly the "defense silently does nothing" failure Round 2 was written
to prevent, just from the opposite direction (previously: *all* peers
falsely reported blocked; now: a genuinely-struggling *majority* falsely
reported healthy).

**Also found:** the previously-passing shape of this bug meant
`test_is_peer_blacklisted_reflects_rate_and_circuit_state` (a 1-peer
network) failed too — with `num_peers=1`, the cap formula
(`floor(n × 0.5)`, capped at `n-1`) gives `cap=0`, and the old per-peer
check made that mean "never report the sole peer blocked," contradicting
the test's own expectation that a real, sole, rate-limited peer should
be reported as such.

**Fix:** `is_peer_blacklisted()` now decides, once per call, which
raw-blocked peers count against the cap — deterministically the first
`cap` of them in ascending peer-id order — instead of asking each one
independently against the raw count of "others." It also special-cases
total outage: if **every** known peer is simultaneously raw-blocked,
there is no alternative left to preserve access to, so the cap does not
fire at all and the true (100%-blocked) state is reported honestly —
this is exactly what
`test_backup_candidates_excludes_blacklisted_and_tried` requires (3/3
peers permanently at zero capacity → 3/3 correctly reported blacklisted,
not artificially reduced to 1/3). Re-verified the original 2-of-3
scenario after the fix:

```
peer0 (congested): True   peer1 (congested): False (quorum override)   peer2 (healthy): False
```

One of the two struggling peers is now honestly reported and routed
around; the cap still bounds how many can be reported at once (1, per
`max_blacklist_fraction`), so the mechanism does what Round 2 intended
instead of collapsing to a no-op under exactly the correlated-congestion
scenario it exists for.

**Verification:** `defense/ddos_sim_defense/test_live_client_defense.py`
— was 9/11 passing (2 failures found this pass), now 11/11. Full
`defense/ddos_sim_defense/` suite also run clean.

### What was NOT touched in DoS

`run_live_defense_eval.py`'s false-block delta fix (Round 2 §2.2) and
`run_live_evaluation.py`'s per-tier fresh-baseline/cooldown fix were both
re-read this pass and are correctly implemented — no further changes.
`drag_llm_service/src/models/open_model.py`'s role-token stripping (flagged
in `problems/ddos_attack_gaps.md` #4 as a blind, non-word-boundary
`.replace("assistant", "")` still running before the regex fix) was
checked directly against the file on disk: **that bug is not present** —
the shipped code already uses only the regex-based
`_strip_leaked_role_tokens()` / `_HEADER_ID_TOKENS` path.
`problems/ddos_attack_gaps.md` is stale on this specific item; no code
change was needed.

---

## 2. MIA — Phase 2 collector migrated to Rev10, plus a new Revision 11 prompt fix

### 2.1 `contrastive_probes.py`'s fast query path had no throttling at all

**File:** `attack/Mia_attack/contrastive_probes.py`

This script's own docstring already flagged (before this pass) that it
had never been executed, and that its `_query_llm_fast()` helper was a
separate, unthrottled duplicate of `mia_attack.py`'s hardened
`_query_llm_raw()` — meaning a real run would immediately re-trigger the
64% HTTP 429 storm Revision 10 fixed elsewhere, since this script issues
5× the per-document call volume of `mia_attack.py`'s main probe loop.

**Fix:** `_query_llm_fast()` now calls `mia_attack.py`'s shared
`_throttle_query()` before every request (same `MIN_QUERY_INTERVAL_S`
pacing, same process) and retries once on HTTP 429 honoring
`Retry-After`, mirroring `_query_llm_raw()` exactly. Its own short
20-second timeout (needed because a handful of PubMedQA questions are
known to hang the service 120s+, and this script queries far more per
document) is preserved.

### 2.2 `contrastive_probes.py` was never migrated to Rev10's calibrated score

Same file, same pre-existing self-disclosed gap: every contrastive probe's
`decision_match` was the raw, uncalibrated signal — no correction for the
model already knowing a biomedical fact from pretraining, the exact
confound Revision 10 exists to net out in the main attack.

**Fix:** for every contrastive-template probe where the RAG-grounded
answer matches gold, the script now also queries the identical probe text
with `no_retrieval=True` (throttled the same way) and computes
`_calibrated_decision_score()` — imported directly from `mia_attack.py`,
not reimplemented — exactly mirroring `_probe_documents()`'s pattern. New
primary fields `calibrated_rate`/`calibrated_std` are added to each output
row; the existing raw `decision_match_rate`/`decision_match_std` are kept
unchanged as diagnostics, matching this project's established
uncalibrated-kept-for-continuity convention.

### 2.3 New: prompt/metric mismatch (Revision 11)

**File:** `attack/Mia_attack/mia_attack.py`

`drag_llm_service`'s own system prompt
(`drag_llm_service/configs/config.yaml`) asks for a short, direct answer
in "at most three words" — it never asks for a yes/no/maybe commitment.
`_decision_match()` greps the response's first few words for the literal
gold token (`"yes"`/`"no"`/`"maybe"`), so a response that answers the
substance correctly but never says the token verbatim (a paraphrased
clause) was silently scored as a non-match regardless of whether the
model was actually grounded — undercounting `decision_match` for reasons
unrelated to membership.

**Not fixed by changing the served system**: per this project's own
CLAUDE.md ("do not modify the Reliable-dRAG core system unless strictly
necessary for instrumentation"), `drag_llm_service/configs/config.yaml`
was left untouched. The fix instead steers the *query text the attacker
sends* — exactly what a real adversary probing this system for a
parseable answer would do, not a modification of the victim system.

**Fix:** new `_format_probe_question()` appends `"? Answer with yes, no,
or maybe."` to every probe question before it is sent, applied
identically to both the RAG-mode call and its paired `no_retrieval`
baseline call in `_probe_documents()` (both must see the exact same query
text apart from retrieval, or the Rev10 calibration comparison is no
longer apples-to-apples). Scoped to `_probe_documents()` (the currently-
live main attack path) only — `_query_llm()`/`_consistency_score()`
(Revision 6, already found non-viable under deterministic decoding) was
left untouched to keep this fix minimal.

**Verification:** `python3 -m py_compile` on both files; no unit test
suite exists for this module (none found under `attack/Mia_attack/`).

---

## 3. SFA — binomial detector's growing-window bug (self-disclosed, now fixed)

**File:** `defense/sfa_sim_defense/selective_forwarding_defense.py`

Already self-disclosed in the module's own docstring and
`problems/sfa_attack_gaps.md` §4 (not newly discovered): `binomial`-mode's
significance test re-tested a peer's **cumulative** `(n, raw_rate)` pair
on every single interaction — `n` only ever grew, never bounded or
reset — unlike the sibling `attack/selective_forward/SFADetector`, which
re-tests over a rolling last-`WINDOW` sample. Two compounding
consequences: as `n` grows without bound, even a noise-level deviation
from the honest baseline eventually reads as statistically "significant"
under a fixed `binom_alpha` (classic sequential-testing inflation of the
false-positive rate as the sample keeps accumulating); and an early,
unlucky patch of misses is diluted rather than aged out by later good
behavior, since old evidence never leaves the test.

**Fix:** both `_binomial_should_blacklist()` (the subject peer's own
test) and `_effective_honest_miss_rate()` (the other-peers median
baseline) now draw from a true fixed-size rolling window
(`self._binom_obs`, a `deque(maxlen=binom_window)` per peer, new
`binom_window` config parameter defaulting to 40 to match
`SFADetector.WINDOW`) instead of the all-time cumulative
`_query_count`/`_response_count` totals. Those cumulative totals are
unchanged and still drive reputation (Layer 1) and "threshold" mode —
only the binomial test's own `(n, rate)` inputs changed.

**Verification:** no prior unit test suite existed for this module. A
standalone, in-process sanity check (5 honest peers at ~5% miss, 1
stealthy attacker at ~25% miss, 300 rounds each, `detection_mode:
binomial`, `honest_miss_rate: auto`) confirmed: every peer's window is
correctly bounded at `maxlen=40` regardless of how many interactions
accumulate (was previously unbounded), and the attacker was caught. Some
honest false positives also occurred at this small population size
(4-5 honest peers) — expected and already disclosed in this same
module's docstring as a real, measured trade-off of the design (not a
regression from this fix); characterizing it properly needs a real
multi-seed live/mock run, which this pass does not include.

### What was NOT touched in SFA

`honest_miss_rate="auto"` (Round 2) and `run_realistic_defense_eval.py`
were both re-read this pass and are already correctly wired — no
further code changes. The 3-source-topology structural limit (F2 in
`current_gaps_overview.md`) and the HTTP-interception gap for measuring
real answer-correctness under SFA are both infrastructure/design work,
not bugs, and are out of scope for a code-fix pass.

---

## 4. Files changed this pass

- `attack/Mia_attack/contrastive_probes.py` — throttled `_query_llm_fast()`,
  migrated to Rev10 calibration, updated module docstring.
- `attack/Mia_attack/mia_attack.py` — new `_format_probe_question()`
  (Revision 11), applied in `_probe_documents()`.
- `defense/sfa_sim_defense/selective_forwarding_defense.py` — true sliding
  window (`_binom_obs`, `binom_window`) for the binomial detector,
  replacing the growing-window test.
- `defense/ddos_sim_defense/live_client_defense.py` — fixed
  `_peer_health_score()` divide-by-zero and the quorum-cap self-defeat bug
  in `is_peer_blacklisted()`.

## 5. What this pass does NOT do

- **No live verification of anything above.** Every fix is verified by
  `py_compile`, the existing/expanded unit test suite
  (`test_live_client_defense.py`, now 11/11), or a standalone in-process
  sanity script — never against the real Docker/Hardhat/`drag_llm_service`
  stack.
- **MIA's Phase 2 contrastive-probe collection has still never been run**
  — this pass makes it safe and calibration-correct *to* run, it does not
  run it.
- **SFA's growing-window fix is not yet characterized empirically** —
  whether it measurably reduces the false-positive rate versus the old
  version, as the module's own docstring speculates a true sliding window
  would, needs a real multi-seed run.
- **The 3-source topology limit (F2) is untouched** — it's the single
  highest-leverage remaining item for both SFA and DoS per
  `current_gaps_overview.md`, and is an infrastructure change, not a code
  bug.
- Data Poisoning, SSM, and KB Extraction were not in scope for this pass.

## 6. Suggested next step (when you're ready to run it)

1. `python3 -m pytest defense/ddos_sim_defense/test_live_client_defense.py`
   — already passing locally; re-confirm after any further edits.
2. Bring up the stack on the powerful machine (not this laptop — no GPU
   here) and run, in order: (a) `run_live_defense_eval.py` to confirm
   `total_quorum_overrides` now fires sensibly under partial congestion
   instead of 0 or "everyone," (b) SFA's `run_defense.py
   --detection_mode binomial` at the new `binom_window` default, watching
   whether the false-positive rate actually drops as expected, (c) a
   small-scale `contrastive_probes.py collect --split dev --seeds 0` pilot
   to confirm the throttle holds under real 429 conditions before scaling
   to the full 13/5-seed collection.
3. Multi-seed re-runs (0, 42, 123) for both SFA and DoS remain the
   highest-priority open item overall per `current_gaps_overview.md` Part
   3 — none of the fixes in this pass substitute for that.
