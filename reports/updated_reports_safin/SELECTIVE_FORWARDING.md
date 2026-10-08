# Selective Forwarding Attack (SFA) — Improvement-Proposal Follow-up

**Status: code written, not executed** (per explicit request this pass).
No Docker/live service was touched.

## Scope note — this pass includes defense work

`.claude/CLAUDE.md` scopes defenses as future work for this system
("Do not claim defenses are in scope for this system"). This pass
implements defense-validation code anyway, **on the user's explicit
instruction** given after I flagged the conflict directly and asked how
to proceed. Two things worth knowing before using this:

1. `defense/sfa_sim_defense/` **already existed** before this session,
   already targets Reliable-dRAG's real 3-node live deployment (not
   DRAG), and is already validated with real numbers in
   `reports/SFA_Security_Analysis_Report.md` (866 lines, 8+ revisions) —
   this pass did not build the reputation/blacklist/bypass/redundant-probe
   mechanism from scratch, it already existed. What was missing was
   validating it against the *harder, more realistic* attacker shapes the
   improvement proposal asks for, plus the specific metrics item 3 names.
2. `SelectiveForwardingDefense` itself was **not modified** — this pass
   only adds an evaluation harness around it. Nothing about the existing
   report's already-validated numbers changes.

## What the improvement proposal asked for (SFA section)

1. Build a larger live testbed with more than three real data-source peers.
2. Test realistic attackers: partial drops, delayed responses, changing
   drop rates, colluding peers, adaptive-to-threshold attackers. Measure
   retrieval success and final LLM answer correctness.
3. Validate the reputation/blacklist and redundant-probe defense on live
   traffic: false positives, detection delay, extra latency, network cost,
   and behavior when honest peers naturally have low relevance.

## What was built

### Item 2 — realistic attackers (`attack/selective_forward_sim/`)

- **`realistic_attackers.py`** — `AdvancedSelectiveForwardingAttack`,
  subclassing the existing, already-validated `SelectiveForwardingAttack`
  rather than editing it (that class is the subject of the 866-line
  report; editing it in place would put every already-validated number in
  that report at risk). Four new, independently composable capabilities,
  all off by default:
  - `delay_range` — injects a random delay before every compromised-peer
    response (drop or pass-through) — a distinct failure mode from
    silence.
  - `drift` — drop probability oscillates sinusoidally over the query
    sequence instead of being fixed per peer for the whole run.
  - `collusion` — compromised peers share one round-robin counter so only
    one is "on duty" to drop any given query, spreading aggregate damage
    thinner per peer than independent dropping would.
  - `adaptive_to_threshold` — each compromised peer replicates the
    defense's own EMA/raw-rate formula locally and self-throttles to stay
    just above the configured `blacklist_threshold` (a worst-case,
    "attacker knows the defense's parameters" threat model, consistent
    with this module's existing `high_connectivity` targeting assumption).
- **`run_realistic_attack.py`** — CLI runner (mock, no Docker needed, or
  `--mode live`). **Actually run** as an offline mock-mode smoke test this
  session (20 queries, 10 peers) — all four variants + combined produced
  correct, internally-consistent output; verified the adaptive attacker's
  final per-peer response rate converges just above its target line, and
  that with everything off (default construction) the subclass reproduces
  the base class's behavior exactly. Not run against real Docker/live
  traffic.
- **Not done:** "measure final LLM answer correctness" (item 2's last
  sentence). `LiveRAGNetwork`'s BFS routing is a *separate* client-side
  simulation against the real data-source containers — it does not run
  inside `drag_llm_service`'s own request path, so a drop simulated here
  has no effect on what `drag_llm_service` actually retrieves or
  generates. Measuring real answer-quality degradation under SFA would
  need an HTTP-level interception layer in front of the real data sources
  (so `drag_llm_service`'s own `/query` traffic is actually affected),
  which is a meaningfully larger change than this pass — flagged rather
  than attempted partially.
- **Not done:** item 1, a larger live testbed (>3 real peers). Needs new
  Docker data-source containers, new corpus splits, and on-chain identity
  registration — infrastructure work, not attack/defense logic. Flagged
  as a gap, not attempted.

### Item 3 — defense validation (`defense/sfa_sim_defense/`)

- **`run_realistic_defense_eval.py`** — reuses `SelectiveForwardingDefense`
  (unmodified) and `AdvancedSelectiveForwardingAttack` (above) together.
  For each attacker variant (delay/drift/collusion/adaptive/combined),
  runs an undefended and a defended sweep at the same seed(s) and reports:
  - **False positives** — any blacklisted peer that was not actually
    compromised (ground truth is known, since this script controls the
    attack).
  - **Detection delay** — the top-level query index at which each
    actually-compromised peer first appears in `blacklisted_peers`
    (polled after every query rather than via a new callback hook in the
    defense class, to avoid modifying it).
  - **Extra latency** — mean per-query wall-clock time, defended minus
    undefended, at the identical attack config/seed — isolates the
    defense's own overhead from the attacker's own injected delay (which
    is present in both runs equally).
  - **Network cost** — `total_bypasses`, `redundant_probes`,
    `redundant_probe_hits` (already tracked by the existing
    `get_stats()`), plus mean hops/query defended vs. undefended.
  - **Low-relevance honest peers** — a separate scenario with **no attack
    applied at all**, where honest peers' hit probability is deliberately
    set low (`--low_relevance_hit_prob`, default 0.1). Any peer
    blacklisted here is unconditionally a false positive — a direct,
    literal test of the exact ambiguity `SelectiveForwardingDefense`'s own
    docstring names ("a single fixed absolute blacklist_threshold cannot
    tell 'this peer is actively dropping queries' apart from 'the whole
    population's natural response rate is just low'").
  - Multi-seed via `--seeds` (default `[0, 42, 123]`, this project's
    documented convention).
- **Verified via `py_compile` and an import-only check only** — not
  executed, per this turn's explicit instruction. Unlike
  `run_realistic_attack.py` (verified with a live mock smoke test last
  turn), this file's actual behavior at runtime has not been observed.

## Commands to run (on your machine, later)

```bash
# Attack side — mock, no Docker needed
python attack/selective_forward_sim/run_realistic_attack.py --mode mock

# Attack side — live, real 3-node deployment
python attack/selective_forward_sim/run_realistic_attack.py --mode live --variant adaptive

# Defense validation — mock, all variants + low-relevance stress test
python defense/sfa_sim_defense/run_realistic_defense_eval.py --mode mock

# Defense validation — live
python defense/sfa_sim_defense/run_realistic_defense_eval.py --mode live --variant collusion
```

## Honest caveats

- The defense evaluator's `detection_delay_queries` is query-granularity,
  not interaction-granularity — a compromised peer visited multiple times
  within one top-level query only updates the delay estimate to that
  query's index, not the exact interaction count. Coarser than what a
  defense-class-internal hook could give, traded for not modifying
  `SelectiveForwardingDefense`.
- `adaptive_to_threshold` assumes the attacker knows the defense's
  configured `blacklist_threshold` exactly. This is an explicit worst-case
  assumption (stated in the module docstring), not a claim that a real
  attacker would have this information in practice — matches this
  module's pre-existing `high_connectivity` targeting assumption
  ("full knowledge of the overlay/on-chain state"), not a new, weaker
  standard introduced here.
- `collusion`'s round-robin coordination assumes compromised peers can
  synchronize a shared counter — realistic for peers under common
  attacker control (this project's own threat model already assumes
  "controls a subset of peer nodes," i.e., common control), not for
  independently-compromised peers with no coordination channel.
- None of this session's new code has been run against live Docker
  traffic. The only execution that happened was `run_realistic_attack.py`
  in mock mode, last turn, as a correctness smoke test.
