# Selective Forwarding Attack — Simulation Module (`selective_forward_sim`)

A second, config-driven Selective Forwarding Attack (SFA) implementation,
separate from `attack/selective_forward` (the existing real-Docker
probabilistic gray-hole attack + EWMA detector + hash-chained ledger). This
module adds what that one doesn't have: a `config/*.yaml`-driven setup and
an in-process **network simulation** (Barabasi-Albert overlay + TTL-bounded
BFS routing over many synthetic peers) for studying attack_ratio/strategy
effects at a scale the real 3-node deployment can't reach — while still
being able to point the exact same attack/defense code at the real
docker-compose services and the real on-chain reputation ledger.

## Two modes, one code path

| | `--mode mock` | `--mode live` |
|---|---|---|
| Network | `network_sim.MockRAGNetwork` — networkx Barabasi-Albert graph, configurable `num_peers` | `live_network.LiveRAGNetwork` — fully-connected graph over the real `source_0/20/100` Docker containers |
| Peers | `MockPeer` — synthetic Bernoulli knowledge model | `LivePeer` — real HTTP calls to `drag_data_source` |
| Requires | nothing (pure Python) | `docker compose up -d` at the repo root |
| Targeting/reporting trust score | in-process, tied at start | real `DragScores` on-chain reliability score (read-only) |

## Live mode is rate-limited — confirmed, and handled by default

Each real `drag_data_source` Flask server enforces **60 requests/minute per
source**. A full sweep can generate hundreds of requests to one source in
seconds and used to silently exceed that — the resulting HTTP `429`
responses were coerced into ordinary "miss" results, corrupting `hit_rate`
with no indication anything was wrong (confirmed directly: `curl` against a
loaded source returned `429 Too Many Requests: 60 per 1 minute` verbatim;
full root-cause writeup in `reports/SFA_Security_Analysis_Report.md` §12.9).

`LivePeer` now self-throttles to `--min_request_interval_s` (default
`1.1`s per source) and retries once on a `429`, respecting the server's
`Retry-After` header. Both runners print and log `rate_limited_total`
(429s specifically) and a broader `error_total` (429s plus timeouts,
connection errors, other non-200s, and malformed responses).

**The acceptance test for "the run is clean" is three checks, not one:**
1. `rate_limited_total == 0` — necessary, not sufficient.
2. `error_total == 0` — the check that actually matters once you raise
   concurrency (see below): raising it can trade 429s for timeouts/5xx,
   which `rate_limited_total` alone would miss entirely.
3. **Attack-only sweep only:** `hit_rate` comes back *flat* across ratios
   within each strategy — on this 3-peer network `num_compromise` is `1`
   for every tested ratio, so the attack itself is identical at every row,
   and a clean measurement cannot show a ratio-dependent curve. This check
   does **not** apply to `run_defense.py`'s comparison sweep — `recovery`
   and `redundant_hits` there depend on stochastic evidence accumulation
   and are expected to vary run to run even when everything is clean.

For a statistically meaningful sample size (`--n_questions 200+ --trials
5+`), client-side throttling alone makes that impractically slow. Raising
just the *server's* rate limit doesn't fix this by itself, though — the
client throttle stays the binding constraint unless it's lowered too.
Move both together:
```bash
RATE_LIMIT_DEFAULT="600 per minute" docker compose up -d --build data-source-0 data-source-20 data-source-100
python3 run_attack.py --mode live --n_questions 200 --min_request_interval_s 0.1
```
`drag_data_source/app/server.py` reads `RATE_LIMIT_DEFAULT` from the
environment (defaults to the original `"60 per minute"`, so nothing
changes unless you opt in). Keep watching `error_total`, not just
`rate_limited_total`, once concurrency goes up.

**Calibration note:** even a fully clean live run only tests *one* attack
severity — `num_compromise=1` is fixed by network size (3 peers), not by
`--attack_ratio`, so the "sweep" varies the ratio without varying the
actual attack. A flat, boring `hit_rate` across ratios is the correct,
expected result on this deployment, not a disappointing one — it is not a
dose-response curve the way the mock-mode sweeps (up to 20 peers) are.

`SelectiveForwardingAttack` and `SelectiveForwardingDefense`
(`defense/sfa_sim_defense/`) don't know or care which mode built the
network — both network classes expose the same `.peers`, `.network`
(networkx graph), `.topic_aware_query()`, `._sfa_defense` surface.

## What the attack does

A compromised peer's object stays intact and reachable in the overlay graph
— it isn't removed, so any health check still passes — but its `.query()`
is monkey-patched to drop with probability `drop_rate`, returning
`(None, None, 0.0, False)`: a silent miss, indistinguishable from an honest
peer with no relevant knowledge. Each visit still burns one TTL hop, so
enough compromised peers — especially high-degree hubs on the mock BA graph
— exhaust a query's hop budget before it reaches a peer with a real answer.
See `selective_forwarding_attack.py`'s module docstring for the full
node-removal-vs-selective-forwarding distinction.

**`drop_rate`: black-hole vs. stealthy.** `drop_rate=1.0` (default) always
drops — trivially detected by any response-rate-based defense, since the
response rate reads ~0%. `drop_rate="stealthy"` draws each compromised
peer's rate independently from `Uniform(0.10, 0.30)` (matching
`attack/selective_forward`'s `SelectiveForwardingAttack.STEALTHY_LO/HI`
exactly), the adversarially harder case: a peer that still answers 70-90%
of the time. `defense/sfa_sim_defense`'s default `detection_mode:
threshold` cannot catch this (see that module's README) — use
`detection_mode: binomial` to test against it.

```bash
python attack/selective_forward_sim/run_attack.py --mode mock --drop_rate stealthy
```

Two targeting strategies (`select_targets()`):
- `random` — uniform sample of `floor(num_peers * attack_ratio)` peers.
- `high_connectivity` — highest-degree peers on the mock BA graph (hubs
  carry disproportionate BFS traffic); on the live 3-node fully-connected
  network, where every peer has equal degree, this falls back to the real
  on-chain reliability score instead — compromising the most-trusted real
  source does the most damage.

## Why live mode doesn't write to the blockchain

`DragScores.feedbackAndUpdateScoreRecords()` can only be called by the
`llmService` address and requires a per-source ECDSA signature (see
`drag_contract/contracts/drag_scores.sol`) — that authority legitimately
belongs to `drag_llm_service`'s real query pipeline, not a standalone
attack script. This module only makes **view calls** (`get_scores_batch`,
no private key needed) to read the genuine on-chain reliability ledger for
targeting and reporting. The chain itself keeps getting maintained the
normal way: by real queries flowing through `drag_llm_service` (the
`llm-service` container in `docker-compose.yml`). Run that alongside these
attack/defense scripts if you want to see the on-chain scores actually move
during a live SFA run.

## Config

`config/selective_forwarding_sim.yaml` — `mode`, attack ratio/strategy/seed,
sweep ranges, mock network params (`num_peers`, `num_attachments`,
`num_query_neighbor`, `query_ttl`, `query_confidence_threshold`), live
params (Docker source URLs, API key, blockchain URL). CLI flags override
YAML values.

## Running

```bash
# Mock sweep across all ratios/strategies in the config (no Docker needed)
python attack/selective_forward_sim/run_attack.py --mode mock

# Larger mock network, single scenario
python attack/selective_forward_sim/run_attack.py --mode mock \
    --num_peers 30 --max_ttl 8 --num_queries 200 \
    --ratio 0.3 --strategy high_connectivity --single

# Live sweep against the real Docker sources + on-chain ledger
docker compose up -d          # repo root; starts hardhat-node + 3 data sources + llm-service
python attack/selective_forward_sim/run_attack.py --mode live --n_questions 50

# Long live sweep: pause between scenarios to avoid saturating the containers
python attack/selective_forward_sim/run_attack.py --mode live --n_questions 50 --scenario_delay_s 1.0
```

A full live sweep is `1 baseline + strategies × ratios` scenarios, each
sending `n_questions` real HTTP calls sequentially — that adds up fast
(e.g. 11 scenarios × 60 questions = 660 calls for the default 2×5 sweep).
On resource-constrained setups (e.g. WSL2 without GPU passthrough) a long
sweep can transiently saturate the source containers, producing results
that look like an attack effect but are actually infra load. `--scenario_delay_s`
adds a pause between scenarios; `defense/sfa_sim_defense/run_defense.py`
has the same flag for the attack+defense comparison, where it matters more
(see that module's README for the related execution-order fix).

Writes both a JSON log (full config + per-scenario metrics) and a CSV sweep
table to `attack_logs/selective_forward_sim/`.

## Metrics (`SelectiveForwardingAttack.collect_metrics()`)

| Metric | Meaning |
|---|---|
| `hit_rate` | fraction of queries answered |
| `avg_hops_per_query` | mean hops spent per query (wasted + useful) |
| `ttl_exhaustion_rate` | fraction of queries that ran out of hop budget unanswered |
| `dropped_queries` | total `.query()` calls silently dropped by compromised peers |
