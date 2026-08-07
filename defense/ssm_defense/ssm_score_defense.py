"""
SSM-Score Manipulation Defense for Reliable-dRAG.

Defense Principle
------------------
The SSM-Score attack (attack/ssm_score) forges score-update transactions that
arbitrarily inflate a source's on-chain reliability/usefulness so the system
preferentially routes queries to a fully-poisoned source.

Two complementary layers close this off:

1. On-chain enforcement (drag_contract/contracts/drag_scores.sol):
   feedbackAndUpdateScoreRecords now rejects, per source, any update that:
     - is not submitted by the registered llmService address (UnauthorizedCaller)
     - arrives less than MIN_UPDATE_INTERVAL seconds after the previous
       update for that source (UpdateTooFrequent)
     - moves reliability/usefulness by more than MAX_DELTA_PER_UPDATE in a
       single transaction (ScoreDeltaTooLarge)
     - pushes a score outside [MIN_SCORE, MAX_SCORE] (ScoreOutOfBounds)
   This is what actually stops the attack script's rounds from landing.

2. Off-chain monitoring (this module): replays the ScoreRecordUpdated event
   log and flags any update that *would* have violated the same bounds, so
   attacks (including ones from before the contract fix was deployed, or
   against a future version of the contract) are visible and quantifiable
   even without needing to read Solidity revert reasons.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Mirror the constants enforced on-chain in drag_scores.sol so detection and
# enforcement never silently drift apart.
MAX_DELTA_PER_UPDATE = 5_000
MIN_UPDATE_INTERVAL = 2  # seconds
MAX_SCORE = 1_000_000
MIN_SCORE = -1_000_000

BLOCKCHAIN_URL = "http://localhost:8545"

DEFAULT_DATA_SOURCES = {
    "sources_0":   "http://localhost:8001",
    "sources_20":  "http://localhost:8002",
    "sources_100": "http://localhost:8003",
}


@dataclass
class Anomaly:
    source_id: str
    kind: str  # "delta_violation" | "rate_violation" | "bounds_violation"
    detail: str
    block_number: int
    transaction_hash: str


@dataclass
class DefenseReport:
    sources_scanned: List[str]
    events_scanned: int
    anomalies: List[Anomaly] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "sources_scanned": self.sources_scanned,
            "events_scanned": self.events_scanned,
            "anomaly_count": len(self.anomalies),
            "anomalies": [
                {
                    "source_id": a.source_id,
                    "kind": a.kind,
                    "detail": a.detail,
                    "block_number": a.block_number,
                    "transaction_hash": a.transaction_hash,
                }
                for a in self.anomalies
            ],
        }


class SSMScoreDefense:
    def __init__(
        self,
        blockchain_url: str = BLOCKCHAIN_URL,
        project_root: Optional[str] = None,
        max_delta: int = MAX_DELTA_PER_UPDATE,
        min_update_interval: int = MIN_UPDATE_INTERVAL,
    ):
        self.blockchain_url = blockchain_url
        self.max_delta = max_delta
        self.min_update_interval = min_update_interval
        self.project_root = project_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        sys.path.insert(0, os.path.join(self.project_root, 'drag_python_client'))
        from drag_python_client import DragScoresClient
        self._client = DragScoresClient(
            project_root=self.project_root,
            provider_url=self.blockchain_url,
        )
        return self._client

    def get_current_scores(self, source_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, int]]:
        client = self._get_client()
        ids = source_ids or list(DEFAULT_DATA_SOURCES.keys())
        returned_ids, rel_scores, use_scores = client.get_scores_batch(ids)
        return {
            sid: {"reliability": rel_scores[i], "usefulness": use_scores[i]}
            for i, sid in enumerate(returned_ids)
        }

    def scan(
        self,
        source_ids: Optional[List[str]] = None,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
    ) -> DefenseReport:
        """
        Replay ScoreRecordUpdated events per source and flag any transition
        that violates the delta cap, cooldown, or absolute bounds enforced
        on-chain. Consecutive events are compared in the order the chain
        recorded them (by block number, then log index).
        """
        client = self._get_client()
        ids = source_ids or list(DEFAULT_DATA_SOURCES.keys())
        report = DefenseReport(sources_scanned=ids, events_scanned=0)

        # sourceID is `string indexed`, so web3.py would need to filter by its
        # keccak topic hash rather than the raw string -- fetch everything in
        # range once and filter client-side on the unindexed `sourceName`
        # copy the contract also emits, which avoids that entirely.
        all_events = client.get_score_record_updated_events(
            from_block=from_block, to_block=to_block,
        )
        report.events_scanned = len(all_events)

        for source_id in ids:
            events = [e for e in all_events if e["sourceName"] == source_id]
            events.sort(key=lambda e: (e["blockNumber"], e["logIndex"]))

            prev = None
            for ev in events:
                if prev is not None:
                    reli_delta = abs(ev["reliabilityScore"] - prev["reliabilityScore"])
                    use_delta = abs(ev["usefulnessScore"] - prev["usefulnessScore"])
                    time_gap = ev["timestamp"] - prev["timestamp"]

                    if reli_delta > self.max_delta or use_delta > self.max_delta:
                        report.anomalies.append(Anomaly(
                            source_id=source_id,
                            kind="delta_violation",
                            detail=(
                                f"reliability delta={reli_delta}, usefulness delta={use_delta} "
                                f"exceed cap of {self.max_delta} in a single update"
                            ),
                            block_number=ev["blockNumber"],
                            transaction_hash=ev["transactionHash"],
                        ))

                    if time_gap < self.min_update_interval:
                        report.anomalies.append(Anomaly(
                            source_id=source_id,
                            kind="rate_violation",
                            detail=(
                                f"update arrived {time_gap}s after the previous one "
                                f"(minimum interval is {self.min_update_interval}s)"
                            ),
                            block_number=ev["blockNumber"],
                            transaction_hash=ev["transactionHash"],
                        ))

                if ev["reliabilityScore"] < MIN_SCORE or ev["reliabilityScore"] > MAX_SCORE \
                        or ev["usefulnessScore"] < MIN_SCORE or ev["usefulnessScore"] > MAX_SCORE:
                    report.anomalies.append(Anomaly(
                        source_id=source_id,
                        kind="bounds_violation",
                        detail=(
                            f"reliability={ev['reliabilityScore']}, usefulness={ev['usefulnessScore']} "
                            f"outside [{MIN_SCORE}, {MAX_SCORE}]"
                        ),
                        block_number=ev["blockNumber"],
                        transaction_hash=ev["transactionHash"],
                    ))

                prev = ev

        return report
