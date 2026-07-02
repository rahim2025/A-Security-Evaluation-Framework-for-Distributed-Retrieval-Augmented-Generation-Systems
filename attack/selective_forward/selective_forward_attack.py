import json, os, random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import requests

DATA_SOURCES = {
    "sources_0":   "http://localhost:8001",
    "sources_20":  "http://localhost:8002",
    "sources_100": "http://localhost:8003",
}
LLM_SERVICE_URL = "http://localhost:9000"
SSM_SCORES = {
    "sources_0":   {"reliability": 10_000, "usefulness": 10_000},
    "sources_20":  {"reliability": 10_000, "usefulness": 10_000},
    "sources_100": {"reliability": 10_000, "usefulness": 10_000},
}
EVAL_DATA = [
    {"question": "who got the first nobel prize in physics",
     "answers":  ["wilhelm conrad röntgen", "röntgen", "roentgen"]},
    {"question": "when is the next deadpool movie being released",
     "answers":  ["may 18, 2018", "2018"]},
    {"question": "which mode is used for short wave broadcast service",
     "answers":  ["mfsk", "olivia"]},
    {"question": "the south west wind blows across nigeria between",
     "answers":  ["till september", "september"]},
    {"question": "who wrote the first declaration of human rights",
     "answers":  ["cyrus"]},
    {"question": "who is the owner of reading football club",
     "answers":  ["dai xiuli", "dai yongge"]},
    {"question": "swan lake the sleeping beauty and the nutcracker are three famous ballets by",
     "answers":  ["pyotr ilyich tchaikovsky", "tchaikovsky"]},
]

@dataclass
class RAGAnswer:
    answer: str; relevant_knowledge: str; relevant_score: float
    num_hops: int; num_messages: int; is_query_hit: bool

class MockSource:
    """Mirrors DRAG MockPeer. query() monkey-patched by apply()."""
    def __init__(self, source_id: str, hit_prob: float = 0.4):
        self.source_id = source_id
        self.hit_prob  = hit_prob
    def query(self, question: str, k: int = 5) -> Tuple[List[Dict], float, bool]:
        score = float(np.random.beta(2, 3))
        if score >= 0.5 and random.random() < self.hit_prob:
            docs = [{"id": f"{self.source_id}_{i}",
                     "text": f"Passage {i} from {self.source_id}: {question}",
                     "score": round(score, 4)} for i in range(k)]
            return docs, score, True
        return [], score, False

class MockRAGNetwork:
    """Mirrors DRAG MockRAGNetwork. 3 sources = 3 hops max."""
    MAX_HOPS = len(DATA_SOURCES)
    def __init__(self, peer_hit_prob: float = 0.4, seed: int = 0):
        self.peer_hit_prob = peer_hit_prob
        self.seed = seed
        self.sources: Dict[str, MockSource] = {
            sid: MockSource(sid, hit_prob=peer_hit_prob) for sid in DATA_SOURCES}
        self.ssm_scores: Dict[str, Dict] = {sid: dict(v) for sid, v in SSM_SCORES.items()}
    @property
    def num_sources(self) -> int: return len(self.sources)
    def query(self, question: str, k: int = 5) -> RAGAnswer:
        hops = 0
        for sid, src in self.sources.items():
            docs, score, is_hit = src.query(question, k)
            hops += 1
            if is_hit:
                return RAGAnswer(docs[0]["text"][:80], docs[0]["text"],
                                 score, hops, hops, True)
        return RAGAnswer("", "", 0.0, self.MAX_HOPS, self.MAX_HOPS, False)

class SelectiveForwardingAttack:
    """
    Mirrors DRAG SelectiveForwardingAttack API: select_targets/apply/revert/collect_metrics.

    KEY DESIGN: _silent_drop uses a SEPARATE random.Random() instance (drop_rng)
    for all drop decisions.  This leaves the global random/np RNG used by
    MockSource.query() completely undisturbed across scenarios, so the only
    thing that differs between runs is the probabilistic drop, not the
    network's hit/miss luck.  This gives monotonic, comparable results.
    """
    def __init__(self, attack_ratio: float = 0.3, seed: int = 42, api_key: str = ""):
        self.attack_ratio = attack_ratio
        self.seed = seed
        self.api_key = api_key
        self._patched_sources: Dict[str, object] = {}
        self.compromised_ids: Set[str] = set()
        self._dropped_queries: int = 0
        # Separate RNG for target selection and drop decisions (never touches global RNG)
        self._rng = random.Random(seed)

    def select_targets(self, network: MockRAGNetwork, strategy: str = "random") -> List[str]:
        source_ids = list(network.sources.keys())
        n = max(1, int(len(source_ids) * self.attack_ratio))
        if strategy == "high_ssm_score":
            return sorted(source_ids,
                key=lambda s: network.ssm_scores.get(s, {}).get("reliability", 0),
                reverse=True)[:n]
        # Use self._rng (not global random) so target selection never perturbs global RNG
        return self._rng.sample(source_ids, min(n, len(source_ids)))

    def apply(self, network: MockRAGNetwork, strategy: str = "random") -> Dict:
        """Monkey-patch compromised sources. Mirrors DRAG apply()."""
        if self._patched_sources: return {}
        targets = self.select_targets(network, strategy)
        self.compromised_ids = set(targets)
        self._dropped_queries = 0
        for sid in targets:
            src = network.sources.get(sid)
            if src is None: continue
            self._patched_sources[sid] = src.query
            attack_ref    = self
            attack_ratio  = self.attack_ratio
            peer_hit_prob = network.peer_hit_prob
            # drop_rng: separate per-source RNG, never touches global random state
            drop_rng = random.Random(self.seed ^ hash(sid))
            def _silent_drop(question, k=5, _sid=sid, _atk=attack_ref,
                             _ratio=attack_ratio, _prob=peer_hit_prob, _rng=drop_rng):
                if _rng.random() < _ratio:
                    # Dropped — return empty (mirrors DRAG: None, None, 0.0, False)
                    _atk._dropped_queries += 1
                    return [], 0.0, False
                # Not dropped — forward normally using global RNG (same as MockSource)
                score = float(np.random.beta(2, 3))
                if score >= 0.5 and random.random() < _prob:
                    docs = [{"id": f"{_sid}_{i}",
                             "text": f"Passage {i} from {_sid}: {question}",
                             "score": round(score, 4)} for i in range(k)]
                    return docs, score, True
                return [], score, False
            src.query = _silent_drop
        ssm = network.ssm_scores
        return {
            "attack": "selective_forwarding", "strategy": strategy,
            "attack_ratio": self.attack_ratio,
            "num_compromised": len(self._patched_sources),
            "compromised_ids": sorted(self._patched_sources.keys()),
            "avg_ssm_score_compromised": float(np.mean(
                [ssm.get(s, {}).get("reliability", 0) for s in self._patched_sources])
            ) if self._patched_sources else 0.0,
            "avg_ssm_score_all": float(np.mean(
                [ssm.get(s, {}).get("reliability", 0) for s in ssm])),
        }

    def revert(self, network: MockRAGNetwork) -> int:
        for sid, orig in self._patched_sources.items():
            src = network.sources.get(sid)
            if src is not None: src.query = orig
        n = len(self._patched_sources)
        self._patched_sources.clear(); self.compromised_ids.clear()
        return n

    @property
    def dropped_queries(self) -> int: return self._dropped_queries

    def collect_metrics(self, answers: List[RAGAnswer], max_hops: int) -> Dict:
        if not answers:
            return {"hit_rate": 0.0, "avg_hops_per_query": 0.0, "ttl_exhaustion_rate": 0.0,
                    "dropped_queries": self._dropped_queries, "total_queries": 0,
                    "answered_queries": 0, "exhausted_queries": 0}
        total = len(answers)
        answered  = sum(1 for r in answers if r.is_query_hit)
        exhausted = sum(1 for r in answers if r.num_hops >= max_hops and not r.is_query_hit)
        return {
            "hit_rate":            answered / total,
            "avg_hops_per_query":  float(np.mean([r.num_hops for r in answers])),
            "ttl_exhaustion_rate": exhausted / total,
            "dropped_queries":     self._dropped_queries,
            "total_queries":       total,
            "answered_queries":    answered,
            "exhausted_queries":   exhausted,
        }

    def run_live(self, eval_data=None, k_per_source=5):
        if eval_data is None: eval_data = EVAL_DATA
        srcs = list(DATA_SOURCES.keys())
        n = max(1, int(len(srcs) * self.attack_ratio))
        self.compromised_ids = set(self._rng.sample(srcs, min(n, len(srcs))))
        self._dropped_queries = 0
        print(f"\n  Compromised: {sorted(self.compromised_ids)}, ratio: {self.attack_ratio:.0%}")
        print("\n  === PHASE 1 - Baseline ===")
        br = self._eval_live(eval_data, k_per_source, False)
        ba = sum(1 for r in br if r["correct"]) / len(br)
        print(f"  Baseline accuracy: {ba:.1%}")
        self._dropped_queries = 0
        print("\n  === PHASE 2 - SFA active ===")
        ar = self._eval_live(eval_data, k_per_source, True)
        aa = sum(1 for r in ar if r["correct"]) / len(ar)
        drop_pp = (ba - aa) * 100
        print(f"  Post-attack: {aa:.1%}  Drop: {drop_pp:.1f}pp  Dropped: {self._dropped_queries}")
        return {"compromised_sources": sorted(self.compromised_ids), "attack_ratio": self.attack_ratio,
                "baseline_accuracy": round(ba, 4), "attacked_accuracy": round(aa, 4),
                "accuracy_drop_pp": round(drop_pp, 2), "n_dropped": self._dropped_queries,
                "attack_success": drop_pp > 10, "baseline_per_q": br, "attacked_per_q": ar}

    def _http_src(self, sid, q, k):
        h = {"X-API-Key": self.api_key} if self.api_key else {}
        try:
            r = requests.post(f"{DATA_SOURCES[sid]}/query", json={"query": q, "k": k},
                              headers=h, timeout=30)
            if r.status_code == 200: return r.json().get("results", [])
        except: pass
        return []

    def _http_llm(self, q):
        h = {"X-API-Key": self.api_key} if self.api_key else {}
        try:
            r = requests.post(f"{LLM_SERVICE_URL}/query", json={"query": q},
                              headers=h, timeout=120)
            if r.status_code == 200:
                d = r.json(); return d.get("response", d.get("answer", ""))
        except: pass
        return ""

    def _eval_live(self, eval_data, k, attack_active):
        out = []; phase = "ATTACKED" if attack_active else "BASELINE"
        for i, item in enumerate(eval_data, 1):
            q = item["question"]; docs = []
            for sid in DATA_SOURCES:
                if (attack_active and sid in self.compromised_ids
                        and self._rng.random() < self.attack_ratio):
                    self._dropped_queries += 1
                    print(f"    [SFA] Dropped {sid} q{i}"); continue
                docs.extend(self._http_src(sid, q, k))
            resp = self._http_llm(q)
            ok = any(a.lower() in resp.lower() for a in item["answers"])
            print(f"    [{phase}] {i}/{len(eval_data)}  {'ok' if ok else '--'}  {q[:50]}")
            out.append({"question": q, "expected": item["answers"], "response": resp,
                        "correct": ok, "docs_retrieved": len(docs)})
        return out

def apply_selective_forwarding(network, attack_ratio, strategy="random", seed=42):
    """Mirrors DRAG apply_selective_forwarding() convenience helper."""
    sfa = SelectiveForwardingAttack(attack_ratio=attack_ratio, seed=seed)
    return sfa, sfa.apply(network, strategy=strategy)
