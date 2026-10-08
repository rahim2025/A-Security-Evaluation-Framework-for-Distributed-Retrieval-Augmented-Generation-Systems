# Verification Status — MIA / SFA / DDoS defenses

Direct answer: **not "working perfectly" — but no longer just "written and
never run," either.** This document is what actually changed when I went
and checked, including one real bug the checking caught and fixed. Treat
this as the authoritative status over anything implied by earlier
messages in this session.

## Summary table

| Defense | Logic verified offline? | Verified against real live traffic? | Bugs found this pass |
|---|---|---|---|
| MIA similarity-noise (`drag_llm_service/src/retriever/defense.py`) | **Yes** — 13/13 unit tests, `drag_llm_service/test_similarity_noise_defense.py` | No | None found (already passing before this pass) |
| SFA (`defense/sfa_sim_defense/run_realistic_defense_eval.py`) | **Yes, just now** — ran in mock mode (`delay`, `adaptive`, low-relevance scenarios); previously only syntax-checked, never executed | No | None found — ran cleanly, produced sane, internally-consistent output |
| DDoS baseline comparison (`attack/ddos_sim/run_baseline_comparison.py`) | **Yes, just now** — ran mock at both toy scale and validated-report scale | No (mock only, no live/Docker component) | None found after checking a suspicious result (see below) |
| DDoS `LiveClientDefense` (`defense/ddos_sim_defense/live_client_defense.py`) | **Yes, just now** — new `test_live_client_defense.py`, 11/11 passing | No | **Yes — one bug in the test itself, not the defense** (see below); confirmed the actual code was correct throughout |
| DDoS `run_live_defense_eval.py` | No | No | Not checked this pass — needs Docker, still only syntax/import-checked |
| `attack/ddos_sim/run_live_evaluation.py` multi-seed refactor | No | No | Not checked this pass — needs Docker, still only syntax/import-checked |

## What "verified offline" actually means here

For each row marked yes, I ran the actual code (not just `py_compile`) with
synthetic/mock data and confirmed it produces correct output — no live
Docker/HTTP traffic involved. That's real evidence the *logic* is sound,
but it is not the same as "confirmed correct against the real Reliable-dRAG
deployment." Nothing in this project has touched Docker this session.

## What checking actually found

**SFA defense evaluator (mock mode, `run_realistic_defense_eval.py`):**
Ran the `delay`, `adaptive`, and `low_relevance_honest_no_attack`
scenarios at small scale. All ran cleanly. One result is worth flagging as
a genuine (expected) finding, not a bug: the `adaptive` attacker scored
**0 true positives across both tested seeds** (never got blacklisted) while
still causing measurable damage — i.e., the threshold-adaptive attacker's
whole design goal (evade detection while still doing damage) is actually
working as intended in this harness. One false positive also occurred (an
honest peer blacklisted at seed 42) — also expected; no detector is
perfect, and that's exactly what the false-positive metric exists to
surface.

**DDoS baseline comparison (mock mode):** A toy-scale smoke run
(8 peers, 20 questions, 1 seed) showed `DDoSDefense` performing *worse*
than no defense — a result worth being suspicious of, so I re-ran it at
the scale the original, already-validated report uses (20 peers, 100
questions, 3 seeds). There, the defense recovered hit-rate as expected
(0.503 → 0.533) — smaller than the original report's own headline number
(different config defaults in this new comparison script, not a
contradiction), but directionally correct and consistent. The toy-scale
result was sampling noise from an undersized smoke test, not a wiring bug
— confirmed, not just assumed.

**`LiveClientDefense` (new offline test suite, 11 tests):** First real
execution of this code since it was written. One test failed on first
run. Root cause, checked and confirmed: **the test's assumption was wrong,
not the code.** It asserted `peer.query is original` after
`defense.remove()` — but Python creates a new bound-method wrapper object
on every attribute access, so that identity check was never going to hold
for *any* correct implementation. I checked the sibling, already-validated
`SelectiveForwardingDefense.remove()` and confirmed it uses the exact same
"reassign, don't delete" restoration pattern `LiveClientDefense` does.
Fixed the test to check actual behavior (a drained rate limiter's
capacity is gone after removal; calls succeed again) instead of object
identity. All 11 now pass, including circuit-breaker open → half-open →
closed transitions, rate-limiter exhaustion, queue-limit blocking, and
cache hit/expiry — the specific mechanisms item 3 of the improvement
proposal asked for.

## Is this "good"?

For a thesis-generalizability argument, yes, with the caveats above stated
plainly rather than implied away:

- The **offline-verifiable parts are now actually verified**, not just
  written — including catching and fixing a real issue, which is the
  point of testing at all.
- The **live/Docker-dependent parts remain unverified** — `run_live_defense_eval.py`,
  the multi-seed `run_live_evaluation.py`, and every MIA/SFA/DDoS live
  path this session touched. These need a real run on your research PC
  before any number from them goes in a table. Don't report a live figure
  from any of this code without running it first.
- None of this offline testing substitutes for the live validation the
  improvement proposal actually asked for (real traffic, real containers,
  real latency). It only rules out the class of bug that offline testing
  *can* catch (wrong logic, wrong wiring, wrong assumptions) — which is a
  meaningful but partial claim, not "this works."
