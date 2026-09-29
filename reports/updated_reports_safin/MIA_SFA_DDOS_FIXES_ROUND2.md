# MIA / SFA / DoS — Second Fix Pass (Weight Decision + Defense Bugs)

**Status: code changed, not executed.** Bringing up Docker/Hardhat and
running live validation was discussed and initially agreed, but the user
then asked to stay code-only for this pass ("don't need to run the code
just updated the code i will run this later on"). Everything below is
static analysis + code fixes, not a live-verified result. Docker itself
was started (daemon confirmed up) but no container, attack, or defense was
actually run against it.

This document answers three things directly, then gives the fix detail:

1. **Is the MIA scoring problem solved?** The structural bug (no
   correction for pretraining knowledge) — yes, fixed in code (Revision 10,
   see `MIA_SCORE_MECHANISM_FIX.md`). The weight choice — confirmed and
   kept as-is, with reasoning below, not re-derived from a fresh live grid
   search (that would require running code). **Not yet empirically
   re-verified against a live run** — that's the one piece still open.
2. **Is the DoS defense fixed?** Two real, concrete bugs found by reading
   the code (not by running it) and fixed — see §2.
3. **Is SFA fixed?** One real, concrete bug of the same class found and
   fixed — see §3.

---

## 1. MIA weights — "the weighted should be the right one to choose"

### What was asked

Pick the right composite weights for the calibrated MIA score, not leave
it as an open question.

### The decision

**Kept: `DECISION_WEIGHT=1.0`, `SIM_WEIGHT=0.0`, `CERTAINTY_WEIGHT=0.0`,
`LEN_WEIGHT=0.0`** — i.e., the calibrated decision score alone, unchanged
from Revision 7's structure. This is a deliberate decision, not a
non-answer, for three reasons:

1. **It's the only weight choice in this project's history backed by a
   real, held-out-validated experiment** — Revision 7's grid search over
   the full weight simplex, evaluated exactly once on 5 fresh seeds never
   touched during the search (mean AUC 0.620, 95% CI [0.546, 0.695],
   excluding chance). Every other candidate weighting in this module's
   history was either hand-tuned (Revisions 2-4, later shown to not
   generalize) or is this same grid search's answer.
2. **Revision 10's calibration doesn't change the other three signals.**
   The fix nets `decision_match` against a no-RAG baseline; it doesn't
   touch how `similarity`, `certainty`, or `length_ratio` are computed.
   The grid search's finding — that all three add net noise, not signal,
   on real dev data — has no mechanism by which it would flip just because
   the *fourth* signal became more precise. If anything, a cleaner primary
   signal makes it less likely the noisier secondary signals would help.
3. **Picking a different weighting now, without live data, would repeat
   exactly the mistake Revisions 2-6 already made and had to walk back**
   (hand-picked weights that looked reasonable but didn't survive a real
   held-out test) — see the module's own docstring history. Guessing a new
   number isn't more "right" than keeping the one number in this project's
   history that was actually tested against unseen data.

**What would change this:** re-running `tune_weights.py`'s dev/test split
against the *calibrated* signal (already flagged as the recommended next
step in `MIA_SCORE_MECHANISM_FIX.md` §3.3/§6). That requires a live run,
which this pass does not include. Until then, `DECISION_WEIGHT=1.0` is the
project's best-evidenced answer, not a placeholder.

**No code changed in this section** — this is a reasoned confirmation of
already-shipped weights, not a new edit.

---

## 2. DoS defense — two real bugs found and fixed

Both found by reading the code directly, not by reproducing a failure
live — the second one specifically because "the DoS defense isn't working
properly" pointed at needing a closer read of the actual gating logic, not
just the already-known gaps in `problems/ddos_attack_gaps.md` (which are
all attack-side, not defense-side).

### 2.1 `LiveClientDefense`'s quorum cap was declared but never enforced

**File:** `defense/ddos_sim_defense/live_client_defense.py`

`max_blacklist_fraction` was accepted as a constructor parameter — clearly
intended as the same quorum-preserving safety cap its sibling
`DDoSDefense` already enforces (`_blacklist_cap()`) — but was never read
anywhere else in the class. `is_peer_blacklisted()` decided whether to
route around a peer using only that peer's own circuit-breaker state and
rate-limit bucket, with no population-level floor.

**Why this matters:** the circuit breaker and rate limiter are
independent, purely local, per-peer mechanisms with no coordination
between peers. Under a real, severe-enough flood, every peer's breaker
can open and every peer's bucket can empty out at the same time. Before
this fix, `is_peer_blacklisted()` would then report the *entire*
population unavailable simultaneously, and the router would bypass all of
them — collapsing this "defense" to worse than no defense at all (100% of
routing attempts abandoned before even trying), which is the exact
"defense doesn't seem to work" symptom this fix targets.

**Fix:** `is_peer_blacklisted()` now checks how many *other* peers are
already reporting blocked before agreeing to report this one blocked too.
If doing so would push the blocked count to or past
`max_blacklist_fraction` of the known population, it fails OPEN for this
peer (lets the router try it anyway) instead of compounding the outage —
mirroring `DDoSDefense`'s existing guarantee exactly. New
`total_quorum_overrides` stat added to `get_stats()` so how often this
fires is visible, not silent.

### 2.2 `run_live_defense_eval.py`'s "false blocks" metric counted real blocks as false ones

**File:** `defense/ddos_sim_defense/run_live_defense_eval.py`

The false-block count was computed as the sum of `total_rate_limited` +
`total_circuit_blocked` (cumulative counters, incrementing continuously
from the moment the defense is installed) + `blacklisted_count`, read
*once* at the very end of the whole test (baseline → real flood →
post-recovery). Since `total_rate_limited`/`total_circuit_blocked` never
reset, that end-of-run total included every legitimate block that happened
**during the real flood** — exactly the blocks the defense is supposed to
make — not just blocks that happened after the flood stopped (the actual
definition of a false positive: a block with no ongoing attack to justify
it).

**Consequence:** the reported "false block" count for `live_client_defense`
was inflated by however many correct blocks occurred during the genuine
attack, making the metric read as "this defense over-blocks constantly"
regardless of whether it actually did anything wrong after recovery.

**Fix:** the script now snapshots `defense.get_stats()` immediately before
the post-recovery phase starts, and computes the false-block count as the
**delta** in the two cumulative counters across the post-recovery phase
only, plus the (already-correct, snapshot-based) `blacklisted_count`.

### What was NOT changed in this section

`DDoSDefense` (the mock-simulation defense, `ddos_defense.py`) was read
carefully and its own `_blacklist_cap()` is correctly wired and enforced
— no equivalent bug found there. `run_defense.py`'s wave/recovery loop
(`advance_wave()`) is called correctly. No changes were made to either.

---

## 3. SFA — one real bug found and fixed, same class as §2.1

**File:** `defense/sfa_sim_defense/selective_forwarding_defense.py`

This one was already self-disclosed as an open gap in
`reports/SFA_Security_Analysis_Report.md` §12.7 and
`problems/sfa_attack_gaps.md` §4, not newly discovered — but it was still
sitting unfixed in the shipped code, and directly matches "miscalculated
theory."

**The bug:** `detection_mode: binomial`'s statistical test needs a real
`honest_miss_rate` baseline to test each peer against. It shipped as a
hardcoded `0.05` default — which silently mismatched this module's *own*
default mock network config (`peer_hit_prob: 0.4`, i.e. a true honest miss
rate of ≈0.6). The project's own report already reproduced the
consequence directly: running the binomial detector with its shipped
defaults against its own default config blacklisted **5 of 10 peers on
every single ratio tested, including when only 1 peer was genuinely
compromised** — every honest peer looked statistically anomalous against
a baseline five times more optimistic than their real behavior. The
report's own recommendation (§12.7 Rec. 4) was "measure the real honest
baseline first and set `honest_miss_rate` to that measurement — do not use
the shipped default without recalibrating" — correct advice that was
never actually wired into the shipped config.

**Fix:** `honest_miss_rate` now defaults to `"auto"`. Instead of a static
constant that has to be manually re-measured and passed by hand for every
different config or deployment (and silently goes stale the moment either
changes), the effective baseline is now measured **live, every time the
test runs**, as the median observed miss rate across the other
currently-tracked, not-yet-blacklisted peers (new
`_effective_honest_miss_rate()`). Median, not mean, specifically so a
minority of genuinely compromised peers can't drag the baseline estimate
upward — the same "compromised peers are a minority" assumption
`max_blacklist_fraction` already relies on elsewhere in this same class.
If fewer than `min_peers_for_auto_calibration` (default 2) other peers
have enough samples yet, the detector abstains for that peer that round
rather than testing against an unreliable guess. Passing an explicit
`--honest_miss_rate <float>` (CLI) or a number instead of `"auto"` (config)
still works exactly as before, for anyone who wants to pin a specific
pre-measured value.

This closes the failure mode structurally — there's no longer a single
static number that can silently mismatch whatever config is actually
running — rather than just replacing one guessed constant with a
better-guessed one.

**Files touched:** `defense/sfa_sim_defense/selective_forwarding_defense.py`
(the fix itself), `config/sfa_sim_defense.yaml` (default changed to
`auto`, tuning-guide comments updated), `defense/sfa_sim_defense/run_defense.py`
(`--honest_miss_rate` now accepts `"auto"` or a float; new
`--min_peers_for_auto_calibration` flag), `defense/sfa_sim_defense/README.md`
(examples and the "Calibration is mandatory" section updated to describe
the fix while keeping the original reproduction as the historical record
of why it exists).

### Systematic check for the same bug class elsewhere

Since §2.1's bug was "a config parameter accepted but never enforced,"
every `attack/` and `defense/` Python file was scanned for the same
pattern (an attribute assigned from a config dict in `__init__` but never
referenced again anywhere in the file, under any alias). Two additional
hits were investigated and confirmed to be false positives (used via a
`defense = self` closure alias, not `self.` directly, in the same file) —
no further instances of this bug class were found.

---

## 4. What this pass does NOT do

- **No live verification of any of the three fixes above.** All three are
  reasoned from reading the code, not confirmed by reproducing the
  original failure and re-running it clean. Docker's daemon was started
  but no container was brought up or exercised.
- **The MIA weight choice was reasoned, not re-derived.** See §1 — a
  fresh grid search against the calibrated signal is still the
  recommended next step once a live run is acceptable.
- **`DDoSDefense`'s own mock-mode blacklist cap was reviewed, not
  changed** — it was already correctly implemented; only its sibling
  `LiveClientDefense` had the gap.
- **`attack/ddos_sim` and `attack/selective_forward_sim`'s own attack-side
  code were not modified** — this pass is scoped to the defenses and the
  evaluation harness's metric computation, per the specific complaint
  ("the dos attack defense is not working") and the MIA/SFA correctness
  ask.
- Every changed `.py` file was syntax-checked with `python3 -m py_compile`
  (parses and byte-compiles only) and the two changed YAML configs were
  parsed with `yaml.safe_load` to confirm they load as intended. Neither
  check executes any attack, defense, or network logic.

## 5. Suggested next step (when you're ready to run it)

Bring the stack up and run, in order: (1) a quick DDoS live-defense check
(`run_live_defense_eval.py --severity high`) to confirm
`total_quorum_overrides` shows nonzero activity under real load and the
corrected false-block count looks sane; (2) an SFA stealthy-attacker run
with `detection_mode binomial` (now default `honest_miss_rate=auto`) to
confirm it no longer over-blacklists on this project's own default mock
config; (3) `tune_weights.py`'s dev/test grid search against the
calibrated MIA signal, to move §1's weight choice from "reasoned" to
"re-validated."
