# DDoS Attack — Improvement-Proposal Follow-up

**Status: code written, not executed.** No Docker/live service was touched.
Same scope-note as `SELECTIVE_FORWARDING.md`: this includes defense work,
on the standing authorization given earlier this session.

## What the improvement proposal asked for (DDoS section)

1. Test on a larger isolated live deployment, more sources, repeated
   trials, different resource limits/datasets/models/traffic levels.
2. Compare against strong baselines: centralized RAG, no reliability-aware
   routing, replicated retrieval, ordinary rate limiting, load-aware
   routing.
3. Implement and test a real live defense, not only a simulation defense:
   authenticated quotas, global rate limits, queue limits, circuit
   breakers, load-aware failover, caching, source-health routing;
   availability, latency, answer quality, defense cost, false blocks.

## What already existed

`attack/ddos_sim/` + `defense/ddos_sim_defense/` were already mature
(741-line `reports/ddos_attack.md`): a mock congestion simulation
(`DDoSAttack`, 3-seed validated), a real HTTP flood
(`TrafficFlood`/`run_live_evaluation.py`, live-validated but
**single-seed**), and a simulation-layer defense (`DDoSDefense`,
reputation/blacklist/bypass/redundant-probe, same shape as the SFA
defense) that the report's own section 13 states was **never tested
against real live congestion** — only the mock simulation.

## What was built

### Item 1 — repeated trials (`attack/ddos_sim/run_live_evaluation.py`)

Refactored (not rewritten) to add `--seeds` (plural): repeats the whole
severities sweep per seed and reports mean/std availability drop per
severity across seeds, via new `run_for_seed()` and
`_aggregate_across_seeds()`. `--seed` (singular) is unchanged — default
behavior is bit-identical to before. Flagged in the script's own help
text that multi-seed mode reproportionally re-floods the live containers
more (use deliberately). "Larger deployment, more sources, different
resource limits/datasets/models" — **not done**, infra work, same
reasoning as SFA item 1.

### Item 2 — baseline comparison (`attack/ddos_sim/run_baseline_comparison.py`)

Mock-mode (no Docker), reuses `DDoSAttack`/`MockRAGNetwork`/`DDoSDefense`
unmodified. Three of five named baselines faithfully reproduced:
- **centralized** — a genuine single-node network (bypasses
  `MockRAGNetwork.__init__`'s Barabási–Albert graph construction via
  `__new__`, since `n=1` has no valid BA graph — not a 2-node
  approximation).
- **replicated retrieval** — new `replicated_query()` helper: queries
  every peer, succeeds if any answers (vs. this project's actual
  TTL-bounded BFS), with its added cost (always `num_peers` hops)
  reported alongside its hit-rate.
- **ordinary rate limiting vs. load-aware routing** — new
  `_FlatRateLimitDefense` (uninformed, fixed admission probability,
  duck-typed to the same `_sfa_defense` hook) compared against the
  existing `DDoSDefense`.

Two explicitly **not** reproduced, flagged rather than faked:
- "Reliability-aware routing on/off" — that's a `drag_llm_service`
  reranker property (`rerank_with_reliability`), not something this mock
  BFS simulation models at all; testing it correctly needs a live
  config-flip + real flood, not a mock approximation.
- A true centralized RAG system is a different system, not a
  configuration of this one — `topology="centralized"` is the closest
  honest analogue inside this project's own network model.

### Item 3 — real live defense

Two new pieces, kept separate from (not replacing) `DDoSDefense`:

- **`defense/ddos_sim_defense/live_client_defense.py`** —
  `LiveClientDefense`: token-bucket rate limiting, a three-state circuit
  breaker (closed/open/half-open), a per-peer concurrency/queue bound, and
  a short-TTL response cache — the four mechanisms the proposal names that
  a reputation-EMA defense (`DDoSDefense`) doesn't provide. Same duck-typed
  `_sfa_defense` interface as the two sibling defenses (`apply`/`remove`/
  `is_peer_blacklisted`/`backup_candidates`/`record_redundant_probe`/
  `get_stats`) — installable identically, no changes needed to either
  network module.
- **`defense/ddos_sim_defense/run_live_defense_eval.py`** — runs the real
  `TrafficFlood` (reused unmodified) against the real 3 data sources while
  comparing `no_defense` / `ddos_defense` / `live_client_defense` head to
  head. Reports, per the proposal's list: availability + latency + hops
  (retrieval layer, via `LiveRAGNetwork` with the defense actually
  installed), answer quality (generation layer, via real `/query` calls
  to `drag_llm_service`, `nlg_metrics` reused unmodified), defense cost
  (`LiveClientDefense`'s own tracked overhead seconds), and false blocks
  (a dedicated post-recovery phase — any block after the flood has
  stopped and a cooldown elapsed is unambiguously false, ground truth
  known since this script controls the flood).

**Important, deliberately-surfaced asymmetry** (documented at the top of
the evaluator's docstring, not left to be discovered in results): the
retrieval-layer metrics DO benefit from whichever defense is installed
(both defenses wrap `LiveRAGNetwork`'s own client). The generation-layer
metrics do NOT — `drag_llm_service` makes its own separate calls to the
data sources that neither defense wraps, so end-to-end answer quality is
expected to be statistically indistinguishable across all three defense
conditions. A defense that also protected layer 2 would mean wrapping
`drag_llm_service`'s own request path (`app/server.py`), which is
deliberately out of scope (`.claude/CLAUDE.md`: don't modify the core
system unless strictly necessary for instrumentation) — same boundary
already hit and documented for MIA and SFA.

## Verification this session

All four new/changed files: `py_compile` + import-only check (no live
service, no execution of attack/defense logic). Not run end-to-end —
per this turn's explicit instruction.

## Commands to run (later)

```bash
# Item 1 — multi-seed live evaluation
python attack/ddos_sim/run_live_evaluation.py --seeds 0 42 123

# Item 2 — mock baseline comparison, no Docker needed
python attack/ddos_sim/run_baseline_comparison.py

# Item 3 — real live defense validation
python defense/ddos_sim_defense/run_live_defense_eval.py --severity high
```

## Honest caveats

- `LiveClientDefense`'s circuit breaker and rate limiter are genuinely
  new mechanisms, not a port of anything already in this repo — unlike
  `DDoSDefense`/`SelectiveForwardingDefense`'s reputation-EMA lineage,
  there is no prior revision history or live validation behind these
  specific parameter defaults (`circuit_failure_threshold=5`,
  `rate_limit_capacity=20`, etc.) — they are reasonable starting points,
  not calibrated against this deployment's actual traffic the way e.g.
  the MIA report's length-floor or certainty-weight constants were.
- `run_baseline_comparison.py`'s `replicated` topology queries every peer
  for every question with no early exit — on a real deployment this would
  be `num_peers`x the real HTTP call volume per question. Fine for the
  mock simulation; would need real cost accounting (not just hop count)
  before treating a live version of this comparison as fair.
- Nothing in this pass touches `drag_data_source`'s or
  `drag_llm_service`'s own server code — both new defenses are
  client-side wrappers around this project's own evaluation scripts'
  calls, per the established scope boundary.
