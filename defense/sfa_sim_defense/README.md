# Selective Forwarding Defense — Simulation Module (`sfa_sim_defense`)

Countermeasure for `attack/selective_forward_sim`. Separate from the
existing `defense/sfa_defense` (which wraps the EWMA/binomial detector +
hash-chained ledger already inside `attack/selective_forward`). This module
implements a different, simpler three-layer defense — response-rate EMA
reputation → threshold blacklist → BFS routing bypass — driven by
`config/sfa_sim_defense.yaml`, and runs against either the mock
network simulation or the real docker-compose deployment (see the
attack module's README for the mock/live split).

## The four layers (`selective_forwarding_defense.py`)

1. **Reputation tracking** — `apply()` wraps every peer's `.query()`. Each
   call updates an EMA-blended reputation score (`alpha = 1 -
   reputation_decay`), starting at `1.0` (trusted by default) and falling
   only as drops accumulate.
2. **Blacklisting** — once a peer has `min_queries_before_blacklist`
   tracked interactions, if its raw response rate is below
   `blacklist_threshold` it's auto-blacklisted. `suspicion_threshold` is a
   softer, reputation-based flag below full blacklisting
   (`is_peer_suspicious`), used by `filter_trusted_peers()` and `validate()`.
3. **Routing bypass** — `network.topic_aware_query()` (see
   `attack/selective_forward_sim/network_sim.py` /
   `live_network.py`) checks `defense.is_peer_blacklisted()` on every hop;
   a blacklisted peer is skipped entirely and its neighbours are expanded
   at the *same* hop depth instead, recovering reachability without
   spending extra TTL budget on a peer already known to be compromised.
4. **Bounded redundant backup probe** (`redundancy_k`, `backup_candidates()`)
   — if the primary TTL-bounded BFS still exhausts its hop budget without a
   hit (common on a small network with a tight `query_ttl`, where bypass
   alone can't recover a hop that was already spent), the network gives the
   defense up to `redundancy_k` *extra* attempts against peers it hasn't
   already tried and hasn't blacklisted, ranked by reputation. This is what
   actually lets correct detection translate into a recovered answer once
   the hop budget alone is too tight — without it, blacklisting can be 100%
   correct and still recover nothing (see "What changed" below). It's
   capped, unlike `attack/selective_forward`'s `SFAMitigation.route()`
   Phase 2, whose backup probe has no bound relative to `max_hops` at all.

**Quorum-preserving blacklist cap** (`max_blacklist_fraction`, default `0.5`)
— a safety net across all of layer 2: auto-blacklisting never excludes more
than this fraction of the known peer population, and never all of them
regardless of fraction (`_blacklist_cap()` always leaves at least one peer
reachable). `blacklist_threshold` is a single fixed number that cannot tell
"this peer is actively dropping queries" apart from "the whole population's
natural response rate is just low" (e.g. a genuinely low-relevance corpus
on some real sources). Without a cap, that ambiguity can escalate: once one
peer's low rate crosses the threshold, others close to it can follow, and
if enough do, `backup_candidates()` runs out of candidates entirely —
`redundant_probes` keeps firing but finds nothing, and recovery collapses
back to (or below) the undefended baseline. Excluding a *majority* of peers
is itself a signal the threshold is miscalibrated for this population, not
evidence to act on. See "Verified: preventing over-blacklisting collapse"
below.

**Caveat on where the recovery actually comes from**: on a large peer pool
with few queries per peer (e.g. 20 mock peers, 100 queries), no single peer
accumulates the `min_queries_before_blacklist` interactions needed to
trigger blacklisting, so observed recovery in that regime comes almost
entirely from layer 4 (generic "try someone else" redundancy) rather than
reputation-informed avoidance — `defense_blacklisted` will read `0` in the
logs even though `hit_rate` improves substantially. Blacklisting-driven
recovery (layers 2-3) is verifiable on a *small* peer pool where
interactions concentrate (e.g. `--num_peers 5`), or in `--mode live` where
all traffic already concentrates onto the same 3 real sources.

## Running

```bash
# Mock: baseline vs attack-only vs attack+defense, all ratios/strategies
python defense/sfa_sim_defense/run_defense.py --mode mock

# Average over multiple trials for statistical stability
python defense/sfa_sim_defense/run_defense.py --mode mock --trials 5

# Live, against the real Docker sources + on-chain ledger
docker compose up -d
python defense/sfa_sim_defense/run_defense.py --mode live --n_questions 50

# Live, with a pause between scenarios on a long sweep (recommended for
# --n_questions above ~30 live -- see "Ordering / load fix" below)
python defense/sfa_sim_defense/run_defense.py --mode live --n_questions 50 --scenario_delay_s 1.0

# Tune defense thresholds directly
python defense/sfa_sim_defense/run_defense.py --mode mock \
    --blacklist_threshold 0.05 --min_queries 15 \
    --suspicion_threshold 0.10 --reputation_decay 0.70 \
    --redundancy_k 2 --max_blacklist_fraction 0.5

# Stealthy attacker (10-30% drop) vs statistically-principled detection --
# see "Detection modes" below; honest_miss_rate MUST be calibrated
# (here, to the default mock peer_hit_prob=0.4 -> honest miss rate ~0.6)
python defense/sfa_sim_defense/run_defense.py --mode mock \
    --drop_rate stealthy --detection_mode binomial --honest_miss_rate 0.6
```

Each row is labeled `baseline` / `attack_only` / `attack_plus_defense`.
`recovery = attack_plus_defense.hit_rate - attack_only.hit_rate` is the
headline number — how much of the hit-rate collapse the defense recovered.
Writes JSON + CSV to `defense_logs/sfa_sim_defense/`.

Verified locally (`--mode mock`, 300 queries, 20 peers, `max_ttl=6`):
recovery is consistently positive and grows with `attack_ratio`, and
`high_connectivity` attacks recover *more* than `random` at the same ratio
— because hub peers also receive more query traffic, so they cross
`min_queries_before_blacklist` and get blacklisted sooner. This matches the
threat model: the strategy that does the most damage undefended is also the
one the reputation system detects fastest.

## What changed: hop-limited (`max_ttl=1`) recovery was near zero before layer 4

An earlier version of this defense had only layers 1-3 (reputation →
blacklist → free bypass). Under a genuinely tight hop budget (`max_ttl=1`,
`--mode live` on the real 3-node deployment), that version's recovery
collapsed to near-zero or even went briefly negative in one scenario:
blacklisting fired correctly (`defense_blacklisted` > 0, `defense_bypasses`
in the dozens) but bypass alone doesn't refund a hop that was already
spent, so a query that started at a now-blacklisted peer and had no hops
left afterward still failed — `ttl_exhaustion_rate` climbed instead of
`hit_rate` recovering. Layer 4 fixes that: verified locally (`--mode mock
--max_ttl 1 --num_peers 5 --num_queries 100`), `high_connectivity`
recovery went from `attack=0.300 → defended=0.640` (`+0.340`) with 1-2
peers correctly blacklisted and 11-28 bypasses recorded — the detection was
already correct before this fix, it just had nowhere to spend that
information under a 1-hop budget.

## Verified: preventing over-blacklisting collapse

At higher attack ratios (or against any population whose *honest* baseline
response rate is naturally low — plausible on the live deployment if a
polluted source's corpus doesn't semantically match a given question batch
well), a fixed `blacklist_threshold` with no cap can blacklist most or all
peers, not just the true attacker(s) — which empties `backup_candidates()`
and collapses recovery back toward zero even though the primary mechanisms
(reputation, blacklist, bypass, redundant probe) are all individually
working as designed. Reproduced directly (8 peers, `query_ttl=1`, `ttl`
forces full connectivity, honest `peer_hit_prob=0.03` to simulate a
low-relevance honest baseline, `attack_ratio=0.4` → 3 of 8 peers truly
compromised):

| Config | Blacklisted | hit_rate | redundant_probes | redundant_probe_hits |
|---|---|---|---|---|
| `max_blacklist_fraction=1.0` (uncapped) | **7 / 8** (5 of them honest) | 0.010 | 75 | 1 |
| `max_blacklist_fraction=0.5` (default) | 4 / 8 (capped) | 0.070 | 571 | 9 |

Uncapped, the defense blacklists nearly the entire population — including
5 honest peers whose only "fault" was a low natural response rate — leaving
`backup_candidates()` almost nothing to try (1 redundant hit in 200
queries). Capped at the default `0.5`, blacklisting still catches 2 of the
3 true attackers but is prevented from also excluding a majority of honest
peers, giving the redundant probe a real pool to draw from (9x more hits).
Regression-tested against the standard config (20 peers, `max_ttl=6`, all
ratios up to `0.5`) to confirm the cap doesn't interfere with legitimate
full detection there — results are byte-for-byte identical to before this
fix, since `blacklisted_count` never approaches the `0.5` cap in that
regime.

## Detection modes: `threshold` catches a black-hole, not a stealthy attacker

The default `detection_mode: threshold` (`blacklist_threshold=0.05`)
blacklists once a peer's response rate drops below 5% — trivial to trigger
against `attack_ratio`'s default `drop_rate=1.0` (always drops, response
rate ~0%), and **structurally incapable** of catching
`attack/selective_forward_sim`'s `drop_rate="stealthy"` mode (each
compromised peer drops only 10-30% of the time, response rate 70-90% —
nowhere near 5%). This isn't a calibration issue to tune away; a single
fixed absolute threshold cannot serve both cases at once.

`detection_mode: binomial` is the statistically-principled alternative —
ported from `attack/selective_forward`'s `SFADetector` design (one-sided
binomial significance test against a measured `honest_miss_rate`,
escalating over `streak_required` consecutive significant windows before
blacklisting) as an explicit opt-in mode. **Verified** (2/10 peers
compromised at a fixed 20% drop rate, honest `peer_hit_prob=0.95`, 300
queries, 5 seeds):

| Mode | True attackers caught | False positives |
|---|---|---|
| `threshold` | 0/2 every single run | 0 |
| `binomial` (`streak_required=3`, this module's default) | 1-2/2 depending on seed | 1/5 seeds |

`threshold` mode isn't "worse-tuned" here — it is mathematically unable to
fire at this drop rate regardless of sample size, identical to the root
cause behind the sibling module's original detector bug (see that module's
README). `binomial` mode catches it, at a real, non-zero false-positive
cost this module reports honestly rather than hiding — see
`selective_forwarding_defense.py`'s module docstring for why (a
growing-window test, not a true sliding window, correlates consecutive
tests for the same peer).

**Calibration is mandatory, not optional — get it wrong and you'll
over-blacklist instead of never-detecting.** `honest_miss_rate` must match
this deployment's *actual* honest response-rate distribution, not an
assumed constant. Concretely reproduced: running
`--detection_mode binomial` with the config default `honest_miss_rate:
0.05` against the *actual* mock-network default `peer_hit_prob: 0.4`
(true honest miss rate ≈ 0.6) blacklisted **5 of 10 peers on every single
ratio row, including ratio=0.1 where only 1 peer was truly compromised** —
every honest peer looked "significant" against a baseline five times
lower than their real miss rate, and the quorum cap (`max_blacklist_
fraction=0.5`) is the only thing that kept it from blacklisting all 10.
Passing the correctly-calibrated `--honest_miss_rate 0.6` fixed it
immediately (`blacklisted` dropped to 0-1 per row, proportional to the
actual attack ratio). For mock mode, calibrate with
`honest_miss_rate ≈ 1 - simulation.peer_hit_prob`; for live mode, measure
the real deployment's honest miss rate the same way the sibling module
did (baseline queries against known-honest sources) before trusting this
mode's output.

```bash
# Correctly calibrated for the default mock config (peer_hit_prob=0.4)
python defense/sfa_sim_defense/run_defense.py --mode mock \
    --drop_rate stealthy --detection_mode binomial --honest_miss_rate 0.6
```

## Ordering / load fix: attack_only vs attack_plus_defense execution order

Earlier versions always ran `attack_only` before `attack_plus_defense` for
every ratio, in one long sequential live sweep against the same real
containers. That meant `attack_plus_defense` always carried strictly more
cumulative real-world load/drift than its paired `attack_only` run — a
systematic bias toward making the defense look worse that had nothing to do
with its logic, and part of the explanation for an observed
`high_connectivity, ratio=0.1` case where `attack_only=1.0` but
`attack_plus_defense=0.0` (the *confirmed* primary cause of that class of
anomaly turned out to be the rate-limit issue below, not just ordering —
see `reports/SFA_Security_Analysis_Report.md` §12.9). `run_defense.py` now
randomizes, per `(strategy, ratio)` pair and per trial (seeded by
`--seed`), which variant actually executes first against the live
containers. Use `--scenario_delay_s` on top of that for long live sweeps to
reduce cumulative load pressure further.

## Live mode is rate-limited — confirmed, and handled by default

Each real `drag_data_source` Flask server enforces **60 requests/minute per
source**; a live sweep here rebuilds a fresh `LiveRAGNetwork` (and fresh
`LivePeer` self-throttle state) *per scenario*, not once for the whole
sweep, so `--min_request_interval_s` (default `1.1`s, paces requests
*within* one scenario) needs `--scenario_delay_s` alongside it (paces the
gaps *between* scenarios) to stay under the limit for a full sweep — using
only one of the two is not sufficient. Both runners print and log
`rate_limited_total` (429s specifically) and a broader `error_total`
(429s plus timeouts, connection errors, other non-200s, malformed bodies).

**Three checks are required before trusting any `hit_rate`/`recovery`
number from a live run, not one:**
1. `rate_limited_total == 0`.
2. `error_total == 0` — the check that actually matters once concurrency
   is raised (below): more simultaneous load can trade 429s for
   timeouts/5xx that `rate_limited_total` alone would never catch.
3. **Attack-only sweep only** (`run_attack.py`, not this module's defense
   comparison): `hit_rate` flat across ratios within each strategy — this
   network's `num_compromise` is `1` for every tested ratio, so a clean
   measurement can't show a ratio-dependent curve. This flatness check
   does **not** apply to `recovery`/`redundant_hits` here in
   `run_defense.py` — those depend on stochastic evidence accumulation
   and are expected to vary run to run even when the run is clean.

For a practical `--n_questions 200+ --trials 5+` sample size, raise the
server-side limit *and* lower the client throttle together — raising the
server limit alone leaves the client throttle as the binding constraint
and gains nothing:
```bash
RATE_LIMIT_DEFAULT="600 per minute" docker compose up -d --build data-source-0 data-source-20 data-source-100
python3 run_defense.py --mode live --n_questions 200 --min_request_interval_s 0.1 --scenario_delay_s 0.2
```
(see the attack module's README for the `RATE_LIMIT_DEFAULT` mechanism).
Full root-cause writeup (confirmed via a reproduced `429 Too Many
Requests: 60 per 1 minute` server response, and confirming the earlier
"run 1 staircase" was this same confound rather than partial
attack-signal validation) in `reports/SFA_Security_Analysis_Report.md`
§12.9, including why a clean live run — with `num_compromise=1` fixed by
network size — tests one attack severity, not a dose-response curve, so a
flat result is the expected pass condition, not a disappointing one.

## Config tuning guide (`config/sfa_sim_defense.yaml`)

| Parameter | Too low | Too high |
|---|---|---|
| `blacklist_threshold` | misses compromised peers | flags honest peers with noisy responses |
| `min_queries_before_blacklist` | blacklists on insufficient evidence | slow to detect and blacklist attackers |
| `suspicion_threshold` | few peers ever treated as suspicious | many honest peers treated as suspicious |
| `reputation_decay` | reputation swings on every single query | reputation barely moves even after many drops |
| `redundancy_k` | detection can be correct and still recover nothing once the primary hop budget is exhausted | approaches "just try everyone regardless of hop budget," defeating the point of a hop-limited test |
| `max_blacklist_fraction` | caps out detection early, capping recovery even when the extra blacklisting would have been correct | (near 1.0) removes the safety net; a miscalibrated threshold can blacklist most/all peers and empty Phase 2's candidate pool |
| `honest_miss_rate` (binomial mode) | flags honest peers as significant outliers — see "Detection modes" above for a reproduced example | a real stealthy attacker's miss rate no longer looks significant relative to the (too-generous) assumed baseline |
| `binom_alpha` (binomial mode) | requires overwhelming evidence, slow to detect | more false positives on honest peers with noisy responses |
| `streak_required` (binomial mode) | a single unlucky significant window can blacklist an honest peer | slower to detect a real stealthy attacker |

## On-chain reads, not writes

Like the attack module, this defense only reads the real `DragScores`
on-chain reliability ledger (via `live_network.get_onchain_reliability_scores`,
a view call) for reporting when run in `--mode live` — it does not attempt
to write blacklist decisions to the contract. `feedbackAndUpdateScoreRecords`
requires the `llmService` signing key and per-source signatures, which
belong to `drag_llm_service`'s real query pipeline, not a standalone
defense-evaluation script. See `attack/selective_forward_sim/README.md` for
the full rationale.
