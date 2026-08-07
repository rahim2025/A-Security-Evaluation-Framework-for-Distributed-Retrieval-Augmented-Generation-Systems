# Congestion-Based DDoS Defense — Simulation Module (`ddos_sim_defense`)

Countermeasure for `attack/ddos_sim.DDoSAttack`. `reports/SFA_Security_Analysis_Report.md`'s
DDoS section identifies three missing defense directions for this attack;
this module implements all three.

## The three layers (`ddos_defense.py`)

1. **Reputation tracking → deprioritization** — `apply()` wraps every
   peer's `.query()`. A defender has no legitimate access to the
   attacker's internal `load_penalty` state (reading it directly would be
   cheating the simulation), so this observes the same signal a real
   client actually has: response outcomes over time, EMA-blended into a
   reputation score exactly like `defense/sfa_sim_defense`'s layer 1, then
   deprioritized once the response rate crosses `deprioritize_threshold`.
2. **Routing bypass + bounded redundant backup probe** — reuses the
   *same* `network._sfa_defense` hook slot `network_sim.MockRAGNetwork` /
   `live_network.LiveRAGNetwork`'s `topic_aware_query()` already
   consults on every hop. That hook is duck-typed (`is_peer_blacklisted()`
   / `record_bypass()` / `backup_candidates()` / `record_redundant_probe()`),
   so this reuses it directly rather than adding a second, parallel hook
   the routing loop would need to learn about — no changes to either
   network module.
3. **Recovery-aware backoff** — the piece the SFA defense doesn't need,
   because congestion (unlike a compromised peer) clears itself after a
   fixed duration in this model. `advance_wave()` is called once per
   attack wave and automatically gives a deprioritized peer a fresh
   evaluation after `backoff_waves` waves, instead of leaving it flagged
   forever on stale evidence — matching real-world DDoS mitigation
   (scrubbing, rate-limiting) restoring a peer within minutes.

**Quorum-preserving cap** (`max_blacklist_fraction`, default `0.5`) is
carried over from `defense/sfa_sim_defense` verbatim: a single fixed
response-rate threshold can't tell "this peer is being flooded" from "the
whole population's honest response rate is just low," so
auto-deprioritization never excludes more than this fraction of the known
population (and never all of it).

## Running

```bash
# Mock: baseline vs attack-only vs attack+defense, all ratios/strategies
python defense/ddos_sim_defense/run_defense.py --mode mock

# Single scenario -- the report's worst-case collapse config, defended
python defense/ddos_sim_defense/run_defense.py --mode mock --single \
    --ratio 0.6 --strategy sequential --duration 600 --iterations 4

# Live, against the real Docker sources
docker compose up -d
python defense/ddos_sim_defense/run_defense.py --mode live --n_questions 30

# Tune defense thresholds directly
python defense/ddos_sim_defense/run_defense.py --mode mock \
    --deprioritize_threshold 0.5 --min_queries_before_action 5 \
    --reputation_decay 0.6 --redundancy_k 2 --max_blacklist_fraction 0.5 \
    --backoff_waves 2
```

Each row is labeled `baseline` / `attack_only` / `attack_plus_defense`, one
row per wave. `recovery = mean(defended.hit_rate) - mean(attack_only.hit_rate)`
across waves is the headline number printed per `(strategy, ratio)` pair.
Writes JSON + CSV to `defense_logs/ddos_sim_defense/`.

## Config tuning guide (`config/ddos_sim_defense.yaml`)

| Parameter | Too low | Too high |
|---|---|---|
| `deprioritize_threshold` | misses congested peers | flags honest peers with naturally noisy responses |
| `min_queries_before_action` | deprioritizes on insufficient evidence | slow to react to a real congestion wave |
| `reputation_decay` | reputation swings on every single query | reputation barely moves even after many drops |
| `redundancy_k` | detection can be correct and still recover nothing once the primary hop budget is exhausted | approaches "just try everyone," defeating a hop-limited test |
| `max_blacklist_fraction` | caps out mitigation early even when more would be correct | (near 1.0) a miscalibrated threshold can deprioritize most/all peers, emptying the backup-probe pool |
| `backoff_waves` | re-trusts a peer while its congestion wave is still active | slow to recover availability after a wave legitimately clears |

## On-chain reads, not writes

Same rationale as `defense/sfa_sim_defense`: this defense does not attempt
to write anything to the `DragScores` contract. `--mode live` only reads
on-chain reliability scores (view calls) via the existing
`live_network.get_onchain_reliability_scores` bridge, for reporting.
