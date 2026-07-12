# Congestion-Based DDoS Attack — Simulation Module (`ddos_sim`)

An application-layer congestion simulation of a DDoS attack against the
DRAG peer overlay, following the same dual-mode (`mock`/`live`) pattern as
`attack/selective_forward_sim`. Full design rationale in
`reports/SFA_Security_Analysis_Report.md`.

## What this attack is — and isn't

Not a real network-layer DDoS: no packets, no sockets, no botnet
infrastructure. A subset of peers is marked "overloaded" per wave; every
query routed to an overloaded peer has a chance of being silently dropped
(Bernoulli trial against a per-peer `drop_probability`), and a non-dropped
call to a congested peer still accrues simulated `extra_hops`/`extra_msgs`
overhead. **Design invariant: peers are never removed from the overlay** —
an attacked peer stays reachable, it just degrades and recovers
automatically once its wave's `ddos_duration` elapses, mirroring how a real
DDoS victim stays "up" but becomes unable to serve requests.

## Two modes, one code path

Reuses `attack/selective_forward_sim`'s network classes directly rather
than duplicating them — `DDoSAttack` only needs the same `.peers` /
`.network` (networkx graph) / `.topic_aware_query()` surface
`SelectiveForwardingAttack` already drives.

| | `--mode mock` | `--mode live` |
|---|---|---|
| Network | `network_sim.MockRAGNetwork` | `live_network.LiveRAGNetwork` — real `source_0/20/100` Docker containers |
| Requires | nothing (pure Python) | `docker compose up -d` at the repo root |

## Wave lifecycle

Each call to `DDoSAttack.run_wave()`:
1. **Recover** — any peer whose recovery timer has expired is cleared back to healthy.
2. **Select targets** — `random` (uniform sample), `targeted` (prefers peers
   not already overloaded, maximizing new damage per wave), `sequential`
   (deterministic round-robin sweep by peer index), or an explicit
   `target_peers` list that overrides strategy selection every wave.
3. **Assign congestion** — sample `intensity ~ Uniform(intensity_min, intensity_max)`,
   derive `drop_probability = min(0.95, intensity * 0.8)` and
   `load_penalty = min(0.90, intensity * 0.9)`. Re-attacking an
   already-overloaded peer keeps the higher of the old/new intensity.
4. **Cascade** — spill `intensity * cascade_factor` (capped at
   `max_cascade_intensity`) onto each target's graph neighbours.

Recovery is expressed in simulated **wave units**, not wall-clock sleeps:
`wave_interval_s` (default 30s) is the assumed real-world duration one wave
represents, so `ddos_duration` (seconds) converts to
`waves_to_recover = round(ddos_duration / wave_interval_s)`. A high
`attack_ratio` combined with a `ddos_duration` long relative to
`wave_interval_s` means the network never gets a chance to recover between
waves — reproduce the report's worst-case collapse with:

```bash
python attack/ddos_sim/run_attack.py --mode mock --single \
    --ratio 0.6 --strategy sequential --duration 600 --iterations 4
```

## Query-time interception

`attach(network)` installs a dynamic-lookup wrapper on every peer's
`.query()` (same monkey-patch mechanism as `SelectiveForwardingAttack`):
called once per network, then `run_wave()` can be called repeatedly without
re-patching, since the wrapper always consults the live `overload_table`.
A peer not in the table is a zero-overhead no-op. A dropped query returns
`(None, None, 0.0, False)` — scored as a full retrieval failure by the same
BFS routing loop `selective_forward_sim` already drives, so the attack is
directly visible in `hit_rate` with no routing-loop changes.

## Running

```bash
# Mock sweep across all ratios/strategies in the config (no Docker needed);
# each scenario reports one row per wave so you can watch availability move
python attack/ddos_sim/run_attack.py --mode mock

# Single scenario: larger network, more waves
python attack/ddos_sim/run_attack.py --mode mock \
    --num_peers 30 --max_ttl 8 --num_queries 200 \
    --ratio 0.3 --strategy targeted --iterations 6 --single

# Live sweep against the real Docker sources
docker compose up -d          # repo root
python attack/ddos_sim/run_attack.py --mode live --n_questions 30
```

Config: `config/ddos_sim.yaml` — `mode`, `ddos:` (attack_ratio/strategy/
target_peers/iterations/ddos_duration/wave_interval_s/intensity bounds/
cascade params/seed), `sweep:` (ratio/strategy grids), `network:`/
`simulation:` (mock params), `live:` (Docker source URLs, api_key,
blockchain_url, query_ttl, n_questions). CLI flags override YAML values.

Writes both a JSON log (full config + per-wave metrics) and a CSV table to
`attack_logs/ddos_sim/`.

## Metrics (one row per wave)

| Metric | Meaning |
|---|---|
| `availability_percentage` | `100 * active_nodes / total_peers`, where a peer counts as down once `drop_probability >= 0.5` |
| `active_nodes` / `overloaded_count` / `down_count` | raw peer counts this wave |
| `avg_load_intensity` | mean intensity across all currently-overloaded peers |
| `hit_rate` | fraction of this wave's query batch answered |
| `avg_hops_per_query` / `ttl_exhaustion_rate` | routing cost/failure for this wave's batch |
| `dropped_queries` | queries silently dropped by an overloaded peer during this wave's batch |
| `targeted_count` / `cascaded_count` | peers directly targeted vs. cascade-damaged this wave |

## No defense wired into this module

This attack module measures impact only. See `defense/ddos_sim_defense` for
the countermeasure — load-aware routing (consult `load_penalty` before
selecting a peer) plus a reputation-EMA/backoff mechanism reused from
`defense/sfa_sim_defense`'s pattern, with a recovery-aware timeout so a
deprioritized peer is retried once its attack wave's `ddos_duration` would
plausibly have elapsed rather than requiring permanent manual re-evaluation.

## Why live mode doesn't write to the blockchain

Same rationale as `attack/selective_forward_sim`: `DragScores.feedbackAndUpdateScoreRecords()`
requires the `llmService` signing key and a per-source signature — that
authority belongs to `drag_llm_service`'s real query pipeline, not a
standalone attack script. Live mode only reads on-chain reliability scores
(view calls, no private key needed) via the same bridge.
