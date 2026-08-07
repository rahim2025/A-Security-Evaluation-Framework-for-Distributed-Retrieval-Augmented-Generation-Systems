# SSM-Score Defense

Defends against `attack/ssm_score`, which forges score-update transactions to
inflate a malicious source's on-chain reliability/usefulness so the system
routes queries to it preferentially.

## What changed

`drag_contract/contracts/drag_scores.sol` — `feedbackAndUpdateScoreRecords` now
enforces, per source, on every update:

| Check | Rule | Stops |
|---|---|---|
| Caller allow-list | only the registered `llmService` address may call the function | third parties submitting feedback directly |
| Cooldown | must wait `MIN_UPDATE_INTERVAL` (2s) since that source's last update | the attack's rapid multi-round bursts |
| Delta cap | `\|new - old\| <= MAX_DELTA_PER_UPDATE` (5,000) per call | single-tx score slams (attack default is +999,999) |
| Bounds | score must stay in `[MIN_SCORE, MAX_SCORE]` (±1,000,000) | drift to extreme values via many small updates |

Violations revert with `UnauthorizedCaller`, `UpdateTooFrequent`,
`ScoreDeltaTooLarge`, or `ScoreOutOfBounds` — the transaction never lands, so
the malicious score change never happens. Verified with a Hardhat test that
replays the attack's exact `+999999` delta and rapid-round pattern (see
commit history for the throwaway test; the checked-in regression coverage is
`drag_contract/test/DragScores.gas.test.js`, updated for the new constructor).

`defense/ssm_defense/ssm_score_defense.py` is a companion off-chain monitor:
it replays `ScoreRecordUpdated` events and flags any historical transition
that violates the same rules, for detection metrics independent of reading
Solidity revert reasons.

## Redeploying the patched contract

The contract address is deterministic (first tx from Hardhat account #0 on a
fresh chain), so it will **not** change — no need to update
`DRAG_SCORES_ADDRESS` or any config after redeploying.

```bash
# from the repo root
docker compose build hardhat-node
docker compose up -d hardhat-node
docker compose logs -f hardhat-node   # wait for "Initialization complete!"

# refresh the host-side ABI copy that llm-service reads (address is unchanged)
docker cp drag-hardhat-node:/app/drag_contract/artifacts ./drag_contract/
docker cp drag-hardhat-node:/app/drag_contract/ignition/deployments ./drag_contract/ignition/
```

Rebuilding `hardhat-node` resets the in-memory chain (all on-chain score
history), but its entrypoint automatically reseeds `sources_0/20/100` — no
manual re-seeding needed. `data-source-*` and `llm-service` don't need a
rebuild or restart; they reconnect over the docker network automatically and
none of the existing function signatures changed.

## Running the evaluation

```bash
# should now fail every round / show ~0 accuracy drop
python attack/ssm_score/run_attack.py

# full before/after + on-chain anomaly scan, logs to defense_logs/
python defense/ssm_defense/run_defense.py
```
