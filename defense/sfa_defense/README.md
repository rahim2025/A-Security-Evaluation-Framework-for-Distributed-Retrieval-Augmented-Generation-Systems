# Selective Forwarding (SFA) Defense

Defends against `attack/selective_forward`, where a compromised-but-online
data source silently drops (gray-holes) 10-30% of queries while continuing
to pass health checks and, on Reliable-dRAG specifically, potentially riding
a high SSM/reliability score so the system keeps routing to it.

## What already existed vs. what this adds

Unlike SSM and MIA, **the detector and mitigation logic already existed** in
`attack/selective_forward/selective_forward_attack.py`, just not wired up
under `defense/`:

- `SFADetector` — per-node EWMA miss-rate tracker + one-sided binomial
  significance test. Escalates a node through suspicion levels 0 (clean) to
  3 (confirmed) once its miss rate is significantly above the expected
  honest baseline for `STREAK_REQ` consecutive windows.
- `SFAMitigation` — routes queries to the least-suspicious, highest-ledger-score
  nodes first, skips blacklisted nodes entirely, and falls back to a
  redundant probe of non-suspected backup nodes on failure.
- `_SSMChain` — an in-process SHA-256-hash-chained reputation ledger:
  rewards hits, penalizes misses (3x penalty if the node is already
  suspected), and auto-blacklists a node once its score drops below a
  threshold.

`attack/selective_forward/run_attack.py` already has `--mode detection` and
`--mode mitigation` that exercise these against the live Docker data
sources with real SQuAD questions. This module (`defense/sfa_defense/`)
doesn't reimplement any of that — it's a thin harness, matching the
`ssm_defense`/`mia_defense` layout, that runs baseline → stealthy (undefended)
→ detection → mitigation back-to-back and writes one consolidated
before/after report to `defense_logs/` instead of four separate
`attack_logs/` files.

**Bug fixed along the way**: `attack/selective_forward/__init__.py` imported
a class named `SelectiveForwardAttack` (missing "ing") that doesn't exist —
the real class is `SelectiveForwardingAttack`. This didn't break
`attack/selective_forward/run_attack.py` itself (it loads the sibling file
directly by path specifically to avoid `__init__.py`), but it broke any
proper package import of `attack.selective_forward` — which is what this
new module needs. Fixed the typo.

## Dataset

Uses the exact same corpus as the attack: `load_squad_questions()` pulls
from HF `rajpurkar/squad` (train split), filtered to the questions whose
context passage is one of the documents actually loaded into the running
`data-source-0/20/100` containers. No new dataset introduced.

## Running the evaluation

```bash
python defense/sfa_defense/run_defense.py
python defense/sfa_defense/run_defense.py --strategy high_ssm_score
python defense/sfa_defense/run_defense.py --seed 0   # repeat --seed 1, 2 for thesis variance
python defense/sfa_defense/run_defense.py --n_questions 150   # more margin above the 40-query warm-up window
```

## Two real bugs found and fixed (verified live, not just documented)

An earlier pass just documented these as caveats; they're now actually fixed
and re-verified against the live containers.

1. **The detector was structurally incapable of ever firing.**
   `SFADetector.HONEST_MISS = 0.68` and `MISS_THRESH = 0.55` were tuned for
   a generic many-node mock topology (`_MockSource`: hit iff
   `Beta(2,3) >= 0.45`). The real Reliable-dRAG deployment doesn't behave
   like that: `RealSource.query()` reports a hit whenever a source returns
   *any* top-k document, which happens almost regardless of relevance —
   measured directly against the live containers (60 real SQuAD questions x
   3 honest sources = 180 queries), the true honest miss rate is **0.000**.
   With `MISS_THRESH=0.55`, a realistic 10-30% stealthy drop rate could
   *never* cross the threshold, so detection was guaranteed to never fire —
   regardless of how many queries you ran or how obvious the attack was.
   Recalibrated to `HONEST_MISS=0.05`, `MISS_THRESH=0.08` (verified against
   the measured baseline with `scipy.stats.binomtest`) in
   `selective_forward_attack.py`. Re-verified live at `n_questions=100`:
   detection now genuinely catches the compromised node (1/1), with the
   honest nodes correctly reporting `miss=0.0` and the compromised one
   `miss≈0.28` (within the attack's 10-30% stealthy range).
2. **Mitigated accuracy wasn't comparable to baseline/attack accuracy** —
   different counting methods (`_evaluate()`'s aggregate-all-sources vs.
   `SFAMitigation.route()`'s first-hit-stops routing). Added
   `naive_route()` in `selective_forward_attack.py`: the identical
   first-hit routing mechanism as `SFAMitigation.route()`, but without
   suspicion-awareness or blacklisting — the fair "undefended routing"
   comparison point. `run_mitigation()` in `run_attack.py` now reports both
   `accuracy` (mitigated) and `naive_routing_accuracy` (same attack, naive
   routing) side by side.
3. **`high_ssm_score` targeting was completely disconnected from the real
   blockchain** — a design gap, not just a calibration issue, caught by a
   sharp question about whether SFA and SSM-Score share the same trust
   score in this framework (they didn't). `_LEDGER` in
   `selective_forward_attack.py` is a self-contained, in-process
   `_SSMChain` simulation: every node starts at the same `INIT_SCORE=10,000`
   and only diverges from *local* hit/miss observations made *during that
   process*. It is never connected to the real `DragScores` contract SSM-Score
   manipulates. Reading `ssm_scores` before any queries run therefore always
   saw three tied scores, so `sorted(..., reverse=True)` fell back to
   Python's stable-sort tie-break — the first source in
   `DATA_SOURCE_URLS` (`source_0`) — **every single time**, regardless of
   which node was actually trusted on-chain. It could also never reflect a
   real prior SSM-Score attack having inflated a node's genuine on-chain
   score, which is the exact scenario the PDF's threat model describes
   ("system prioritises high-scoring sources... forwarding-dropper can
   cause maximum damage").

   Fixed by adding `get_onchain_reliability_scores()` in `run_attack.py`,
   which reads the real `DragScores` contract (bridging SFA's `source_0`
   naming to the chain's `sources_0`) and is now used for (a) `high_ssm_score`
   targeting in every mode, and (b) seeding `SFAMitigation`'s pre-suspicion
   routing-priority baseline in `_build_real_sources()`, so the attacker's
   targeting and the defender's routing tie-breaker read the *same* trust
   score SSM-Score attacks — falling back to the local simulated ledger only
   if the blockchain is unreachable (keeps `--mode mock` working offline).

   **Caveat on the `0.69 → 1.00` result reported below**: that run predates
   this fix. The compromised node in that run (`source_0`) was chosen by the
   stale tie-break, not because it was verified to be the actual
   highest-scoring node — so the *routing-intelligence* finding is still
   real (naive vs. suspicion-aware routing under an equivalent attack), but
   the "targeted because it was trusted" framing wasn't actually true yet at
   that point. Re-run the commands below to get a targeting decision that
   genuinely reflects on-chain trust.

## Why max_hops matters for seeing the attack's real impact

With `max_hops` equal to the total source count (today's actual deployment:
`n_retrievers: 3` for 3 sources in `drag_llm_service/configs/config.yaml`),
every routing strategy -- naive or suspicion-aware -- eventually tries all
3 sources, so they converge and the attack barely moves accuracy. That's a
genuine property of full redundancy, not a bug to paper over. The PDF's
threat model ("system prioritises high-scoring sources... forwarding-dropper
can cause maximum damage") only bites when the hop budget is *below* the
source count. `run_defense.py` now runs both: phase `[4/5]` at your
configured `--max_hops` (matching current deployment behavior), and phase
`[5/5]` at a dedicated `max_hops=1` worst-case. A live run
(`--strategy high_ssm_score --n_questions 60`) showed:

| Scenario | Naive routing | Mitigated routing |
|---|---|---|
| `max_hops=3` (current config) | 1.00 | 1.00 |
| `max_hops=1` (hop-limited) | 0.68 | **1.00** |

That's the real, verified value of the suspicion-aware routing — visible
only once you test the scenario the threat model actually describes.

## Verifying `high_ssm_score` targeting now follows real trust

The SSM-Score contract fix (see `defense/ssm_defense/`) means an actual
inflation attack can no longer create a differentiated score to test
against — that's the fix working as intended, not a testing obstacle. To
verify `high_ssm_score` targeting genuinely reads the chain rather than
falling back to the tie-break, give one source a real (small, legitimate,
within the ±5,000 cap) score bump first, then confirm targeting follows it:

```bash
python - <<'EOF'
import sys, json
sys.path.insert(0, "drag_python_client")
from drag_python_client import DragScoresClient, sign_message_personal
PK_SOURCES_20 = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
PK_LLM        = "0x701b615bbdfb9de65240bc28bd21bbc0d996645a3dd57e7b12bc2bdf6f192c82"
client = DragScoresClient(project_root=".", provider_url="http://localhost:8545")
ids, rel, use = client.get_scores_batch(["sources_20"])
msg = json.dumps({"query": "legit_bump", "selected_sources": {"sources_20": [0, 0]}}, sort_keys=True)
sig = sign_message_personal(msg, PK_SOURCES_20)
client.feedback_and_update_score_records(
    caller_private_key=PK_LLM, message=msg, signatures=[sig],
    update_source_ids=["sources_20"], update_reliability_scores=[rel[0] + 4000],
    update_usefulness_scores=[use[0] + 4000], info="legit_test_bump",
)
print("sources_20 bumped:", client.get_scores_batch(["sources_20"]))
EOF

python attack/selective_forward/run_attack.py --mode detection --strategy high_ssm_score --n_questions 100
# Expect: compromised == ['source_20'], not the previous stale default of source_0
```
