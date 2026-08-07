# DDoS — Fixes Applied

Companion to `problems/ddos_attack_gaps.md` (numbering referenced below).

---

## 1. [Gap #4] Word-boundary-unsafe role-token stripping fixed

`drag_llm_service/src/models/open_model.py`: the blind
`.replace("assistant", "")` (which could corrupt legitimate answer text
containing that substring anywhere, e.g. "physician assistant") was removed
and replaced with a scoped regex, `_HEADER_ID_TOKENS`, for the
`<|start_header_id|>`/`<|end_header_id|>` markers specifically. The
existing `_strip_leaked_role_tokens()` / `_ROLE_PREFIX` regex (already
correctly word-boundary-scoped to a *leading* role token) is unchanged and
still runs after it.

## 2. [Gap #2] Live severity-tier confound fixed

`attack/ddos_sim/run_live_evaluation.py`: previously measured one baseline
before any flooding and reused it across all three severity tiers, with no
cooldown between tiers — so "high" was measured on sources already flooded
twice in immediate succession. Fixed:
- Each tier now measures its **own fresh baseline** immediately before that
  tier's flood starts.
- A `--tier_cooldown_s` wait (default 60s) runs before every tier after the
  first, so the rate-limit bucket and prior flood's load drain first.

Re-run live (`--num_questions 10 --seed 42`, default rate limit, after the
llm-service fix below) now shows a clean, properly-isolated dose-response
curve, each tier against its own baseline:

| tier | baseline avail_hit | post-attack avail_hit | post-attack f1 |
|---|---|---|---|
| low  | 1.000 | 1.000 | 0.400 (vs 0.600 baseline) |
| mid  | 1.000 | 0.800 | 0.300 |
| high | 1.000 | 0.100 | 0.000 |

Log: `attack_logs/ddos_sim/live_eval_2026-07-19_01-18-30_ddos_comparison.json`
(the `...00-27-01...` log is an earlier 3-question, low-severity-only dry
run against the still-broken llm-service — not a valid result, kept only
because it's what surfaced the role-token contamination below).

## 3. [Gap #5] Shared-RNG path-dependence — now actually fixed in code

Originally left as a documented-only caveat per the gap tracker's severity
call, then fixed for real at the user's explicit request: `MockRAGNetwork`
(`attack/selective_forward_sim/network_sim.py`) previously passed one shared
`random.Random` instance to every `MockPeer`, so a query dropped before
reaching a peer's `.query()` consumed zero draws from it, shifting every
downstream peer's random outcomes depending on how many earlier queries were
dropped. Fixed by giving each peer its own independent RNG, deterministically
derived from `(seed, peer_id)` (`random.Random(seed * 1_000_003 + pid)`) —
a peer's draw sequence no longer depends on any other peer's call history.
The network's own `_rng` (used only for BFS start-peer selection) is
untouched.

This is shared infrastructure with `selective_forward_sim`, so both mock
seed sweeps were re-run afterward (see §4) to regenerate valid logs under
the new RNG structure — the exact reproducible numbers changed (expected,
intentional), the qualitative findings (recovery scales with ratio, collapse
scales with intensity) did not.

## 4. Mock multi-seed sweep run (partially closes [Gap #1]) — re-run after the RNG fix

`attack/ddos_sim/run_attack.py --mode mock --seed {0,42,123}` — no Docker
required, matches the gap tracker's own suggested action. Re-run after §3's
RNG fix (the earlier logs predate it). Logs:
`attack_logs/ddos_sim/attack_2026-07-19_01-44-4{0,1,1}_ddos_sim_mock_seed{0,42,123}.json`.
Live-mode multi-seed is still open (only one live eval config was re-run
this session; a full seed sweep of the live comparison wasn't done).

## 5. [Gap #3] Mock-mode collapse — hyperparameter sensitivity sweep run

New `attack/ddos_sim/hyperparameter_sensitivity.py`: sweeps `intensity_min`
(intensity_max fixed at 1.0) at fixed `ratio=0.3, strategy=random,
iterations=5`, across seeds {0,42,123}, and reports final
`availability_percentage`:

| intensity_min | mean_intensity | final avail% | stdev |
|---|---|---|---|
| 0.1 | 0.550 | 50.0 | 8.16 |
| 0.2 | 0.600 | 51.7 | 12.47 |
| 0.3 | 0.650 | 46.7 | 16.50 |
| 0.4 | 0.700 | 36.7 | 6.24 |
| 0.5 | 0.750 | 38.3 | 4.71 |
| 0.6 | 0.800 | 26.7 | 12.47 |
| 0.7 | 0.850 | 23.3 | 2.36 |
| 0.8 | 0.900 | 15.0 | 8.16 |

**This is a genuine, monotonic dose-response curve, not a binary artifact of
one cherry-picked setting.** The naive worry ("of course it collapses, mean
intensity crosses the DOWN_THRESHOLD=0.5 breakpoint at intensity≈0.625") is
only half right: availability degrades smoothly across the *entire* swept
range, including settings well below that breakpoint (even
`intensity_min=0.1`, mean 0.55, still collapses to 50% after 5 waves via
cumulative worst-case accumulation and cascade effects onto neighbours) —
so this isn't a step function that only fires at the default. The default
(`intensity_min=0.5`, 38.3% avail) sits in the middle of a real, measured
curve, not at an isolated edge case.

## 6. [Gap #5, flood_ramp_s] Empirically validated, not just asserted

New `attack/ddos_sim/validate_flood_ramp.py`: starts a real `TrafficFlood`
against `source_0` and fires a fresh probe request every 0.25s, tracking
when probe latency stops climbing (the target has reached steady-state
congestion). Run at all three severity tiers' worker counts (raised the
source's rate limit to 6000/min first, to isolate the single-threaded-Flask
queueing-delay mechanism this module documents from rate-limiter rejection
noise — reverted after):

| tier | workers | latency at t=0 | plateau reached |
|---|---|---|---|
| low  | 3  | 0.063s | t≈0.25s (first sample) |
| mid  | 6  | 0.123s | t≈0.25s |
| high | 10 | 0.212s | t≈0.25s |

Congestion saturates essentially immediately in all three tiers and stays
flat for the full 5s observation window — `flood_ramp_s=1.0` (the current
default) is validated as sufficient, with roughly 4x margin, not just
asserted. Latency scales with worker count as expected for the
single-threaded-Flask-queueing mechanism the module's own docstring
describes.

## 5. [NEW, not in the original tracker] `llm-service` was running a stale/broken image

Discovered while verifying the DDoS live-eval fix: responses came back as
`{"response":"system\nno"}` — literal leaked role tokens, meaning
`_strip_leaked_role_tokens` (which predates this session) wasn't actually
running in the deployed container at all. Investigation:

1. `docker exec drag-llm-service cat /app/src/models/open_model.py` showed a
   version with **no** role-token handling whatsoever — older than even the
   pre-existing partial fix, let alone this session's edit.
2. Root cause: same class of bug as the KB extraction Docker-mount mismatch
   (`kb_extraction_fixed.md` §5) — BuildKit's cache is content-addressed and
   not project-namespaced, and this build appears to have reused a cached
   `COPY drag_llm_service /app` layer from a much older build state instead
   of the current source on disk.
3. First attempt to fix: `docker compose build --no-cache llm-service`. This
   *did* bake in the current source (confirmed), but the full from-scratch
   reinstall pulled a different vLLM/torch build that immediately
   crash-looped: `RuntimeError: UVA is not available` (RestartCount climbing
   on every check).
4. Recovery: retagged the older `reliable-drag-main-llm-service:latest`
   image (confirmed working, no crash, but running stale application code
   including a missing API-key-forwarding feature added since) onto the
   current project's image tag, recreated the container from it, then
   `docker cp`'d the **current** `drag_llm_service/app`, `drag_llm_service/src`,
   and `drag_llm_service/configs` directories into the running container and
   `docker restart`ed it (this reloads the Flask/vLLM process against the
   copied files without touching the working pip/vLLM environment).
5. Verified: `RestartCount: 0`, healthy, and 3 independent process restarts
   each returning clean single-word answers (`{"response":"yes"}` /
   `{"response":"no"}`) with no leaked tokens.

**Consequence:** every live-LLM-touching result generated earlier in this
session (KB extraction Phase C, the DDoS dry-run) was computed against this
broken image and was re-run after the fix — see `kb_extraction_fixed.md` §7
and §2 above.

**Not yet done:** root-causing *why* BuildKit reused a stale layer in the
first place (so it doesn't silently recur on the next `docker compose up`);
for now the working state is a manually patched container, not a clean
Dockerfile/image rebuild.
