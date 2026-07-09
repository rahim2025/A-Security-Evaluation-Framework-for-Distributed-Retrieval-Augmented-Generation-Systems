"""
attack/selective_forward/run_attack.py  (fixed — real Docker sources)

Replaces the mock-only version. Connects to the live drag_data_source
Docker containers and uses SQuAD v1.1 validation questions as the workload.

Architecture
------------
  data-source-0   http://localhost:8001   (0 % polluted)
  data-source-20  http://localhost:8002   (20 % polluted)
  data-source-100 http://localhost:8003   (100 % polluted)

The SelectiveForwardingAttack monkey-patches the .query() method on
RealSource objects so some sources silently drop queries (gray-hole).
Accuracy = fraction of SQuAD questions for which at least one non-dropped
source returns a document containing the ground-truth answer string.

Usage
-----
  # Full real-Docker run — all modes
  python attack/selective_forward/run_attack.py --mode baseline
  python attack/selective_forward/run_attack.py --mode stealthy
  python attack/selective_forward/run_attack.py --mode detection
  python attack/selective_forward/run_attack.py --mode mitigation

  # Sweep all ratios and produce thesis table (default mode)
  python attack/selective_forward/run_attack.py --mode sweep --n_questions 50

  # Offline mock (no Docker needed)
  python attack/selective_forward/run_attack.py --mode mock

Prerequisites
-------------
  pip install datasets requests numpy scikit-learn sentence-transformers
  docker compose up -d   (from drag_data_source/)
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ══════════════════════════════════════════════════════════════════════════════
#  Local fallback ledger — used when _LEDGER cannot be imported from the
#  original selective_forward_attack module (it is an internal, not part of
#  the public API in all versions of the file).
# ══════════════════════════════════════════════════════════════════════════════

class _SimpleLedger:
    """
    Minimal stand-in for the SSMScoreBoard / _LEDGER object.
    Tracks per-node scores used to choose which nodes to compromise.
    """
    def __init__(self) -> None:
        self._scores: Dict[str, float] = {}
        self._chain:  List[dict]       = []

    def register(self, node_id: str, initial_score: float = 0.5) -> None:
        if node_id not in self._scores:
            self._scores[node_id] = initial_score
            self._chain.append({"node": node_id, "score": initial_score})

    def score(self, node_id: str) -> float:
        return self._scores.get(node_id, 0.5)

    def update(self, node_id: str, new_score: float) -> None:
        self._scores[node_id] = float(new_score)
        self._chain.append({"node": node_id, "score": new_score})

    @property
    def chain_length(self) -> int:
        return len(self._chain)

    def chain_valid(self) -> bool:
        return True  # trivially valid for a simple list


def _local_check_blockchain_status(ledger: "_SimpleLedger") -> Dict[str, Any]:
    return {
        "chain_valid":     ledger.chain_valid(),
        "chain_length":    ledger.chain_length,
        "num_blacklisted": 0,
    }


# ── Try to import from the real selective_forward_attack module ───────────────
_SFA_CLASS        = None
_SFA_DETECTOR     = None
_SFA_MITIGATION   = None
_REAL_LEDGER      = None
_REAL_BLOCKCHAIN  = None

def _load_sibling_module():
    """
    Load selective_forward_attack.py by absolute file path — works even when
    attack/ and attack/selective_forward/ have no __init__.py files.
    Falls back to package-style import if the sibling file is not found.
    """
    import importlib.util as _ilu

    sibling = os.path.join(_HERE, "selective_forward_attack.py")
    if os.path.exists(sibling):
        spec = _ilu.spec_from_file_location("selective_forward_attack", sibling)
        if spec and spec.loader:
            mod = _ilu.module_from_spec(spec)
            # Register in sys.modules *before* exec: the module defines a
            # @dataclass under `from __future__ import annotations`, and
            # dataclasses resolves string annotations via
            # sys.modules[cls.__module__].__dict__ — if the module isn't
            # registered yet, that lookup returns None and crashes with
            # "'NoneType' object has no attribute '__dict__'".
            sys.modules[spec.name] = mod
            try:
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                return mod
            except Exception as e:
                print(f"  [SFA] Could not exec selective_forward_attack.py: {e}")
                del sys.modules[spec.name]
                return None
    # Fallback: try package import (requires __init__.py in attack/ dirs)
    try:
        import importlib
        return importlib.import_module("attack.selective_forward.selective_forward_attack")
    except Exception:
        return None


def _get_attr(mod, *names):
    """Return the first attribute found on mod, or None."""
    if mod is None:
        return None
    for n in names:
        v = getattr(mod, n, None)
        if v is not None:
            return v
    return None


_sfa_mod         = _load_sibling_module()
_SFA_CLASS       = _get_attr(_sfa_mod, "SelectiveForwardingAttack")
_SFA_DETECTOR    = _get_attr(_sfa_mod, "SFADetector")
_SFA_MITIGATION  = _get_attr(_sfa_mod, "SFAMitigation")
_NAIVE_ROUTE     = _get_attr(_sfa_mod, "naive_route")
_REAL_LEDGER     = _get_attr(_sfa_mod, "_LEDGER", "LEDGER", "score_ledger")
_REAL_BLOCKCHAIN = _get_attr(_sfa_mod, "check_blockchain_status", "blockchain_status")

if _SFA_CLASS is not None:
    print("  [SFA] Loaded SelectiveForwardingAttack ✓"
          + ("  SFADetector ✓" if _SFA_DETECTOR else "")
          + ("  SFAMitigation ✓" if _SFA_MITIGATION else ""))
else:
    print("  [SFA] WARNING: SelectiveForwardingAttack not found.\n"
          f"        Looked for: {os.path.join(_HERE, 'selective_forward_attack.py')}\n"
          "        Ensure run_attack.py is placed in attack/selective_forward/ "
          "alongside selective_forward_attack.py")

# Use the real ledger if available, otherwise the local fallback
_ledger_instance = _REAL_LEDGER if _REAL_LEDGER is not None else _SimpleLedger()
_LEDGER          = _ledger_instance


def check_blockchain_status() -> Dict[str, Any]:
    if _REAL_BLOCKCHAIN is not None:
        try:
            return _REAL_BLOCKCHAIN()
        except Exception:
            pass
    # Safe fallback: use _SimpleLedger methods when available, otherwise
    # return static defaults so callers never crash on missing attributes.
    ledger = _ledger_instance
    try:
        chain_valid = ledger.chain_valid() if callable(getattr(ledger, "chain_valid", None)) else True
    except Exception:
        chain_valid = True
    try:
        chain_length = (ledger.chain_length if not callable(getattr(ledger, "chain_length", None))
                        else ledger.chain_length())
    except Exception:
        chain_length = 0
    return {
        "chain_valid":     chain_valid,
        "chain_length":    chain_length,
        "num_blacklisted": 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

DATA_SOURCE_URLS: Dict[str, str] = {
    "source_0":   os.getenv("DS0_URL",  "http://localhost:8001"),
    "source_20":  os.getenv("DS1_URL",  "http://localhost:8002"),
    "source_100": os.getenv("DS2_URL",  "http://localhost:8003"),
}
API_KEY         = os.getenv("API_KEY", "reliable-derag-secret-2026")
BLOCKCHAIN_URL  = os.getenv("BLOCKCHAIN_URL", "http://localhost:8545")
LOG_DIR  = os.path.join(_ROOT, "attack_logs")
os.makedirs(LOG_DIR, exist_ok=True)

SWEEP_RATIOS     = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SWEEP_STRATEGIES = ["random", "high_ssm_score"]


# ══════════════════════════════════════════════════════════════════════════════
#  Bridge to the real on-chain DragScores reliability scores
#
#  _LEDGER (from selective_forward_attack.py) is a self-contained, in-process
#  SHA-256-chained reputation simulation -- it starts every node at the same
#  INIT_SCORE and only diverges from local hit/miss observations *during this
#  process*. It is NOT connected to the real DragScores contract that
#  attack/ssm_score manipulates. That means "high_ssm_score" targeting, read
#  before any queries have run, sees every node tied at INIT_SCORE and
#  degenerates to a fixed tie-break (always the first source in
#  DATA_SOURCE_URLS) rather than genuinely picking "the most-trusted node" --
#  and can never reflect a real prior SSM-Score attack having inflated a
#  node's actual on-chain score. This bridge reads the real chain instead, so
#  SFA's targeting/routing baseline and SSM-Score's target ledger are the
#  same trust score, not two disconnected simulations.
# ══════════════════════════════════════════════════════════════════════════════

def get_onchain_reliability_scores(node_ids: List[str]) -> Dict[str, float]:
    """
    Read real reliability scores from DragScores for the given SFA node IDs
    (source_0/20/100), bridging to the chain's sources_0/20/100 naming.
    Returns {} (caller falls back to the local _SSMChain ledger) if the
    blockchain isn't reachable -- keeps --mode mock and non-Docker runs
    working.
    """
    try:
        sys.path.insert(0, os.path.join(_ROOT, 'drag_python_client'))
        from drag_python_client import DragScoresClient
        client = DragScoresClient(project_root=_ROOT, provider_url=BLOCKCHAIN_URL)
        chain_ids = [nid.replace("source_", "sources_", 1) for nid in node_ids]
        _, rel_scores, _ = client.get_scores_batch(chain_ids)
        return {nid: float(rel_scores[i]) for i, nid in enumerate(node_ids)}
    except Exception as e:
        print(f"  [warn] Could not read on-chain reliability scores ({e}); "
              "falling back to the local simulated ledger for high_ssm_score targeting.")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
#  Real HTTP source wrapper
# ══════════════════════════════════════════════════════════════════════════════

class RealSource:
    """
    Wraps one drag_data_source Docker node as a query-able source object.

    .query(question, k) → (results, top_score, hit)
      results   : list of {"id", "text", "score"} dicts from /query
      top_score : highest similarity score returned (0.0 if no results)
      hit       : True if at least one document was returned
    """

    def __init__(self, node_id: str, url: str, api_key: str = "") -> None:
        self.node_id  = node_id
        self.url      = url
        self._api_key = api_key

    def query(self, question: str, k: int = 5) -> Tuple[List[dict], float, bool]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        try:
            resp = requests.post(
                f"{self.url}/query",
                headers=headers,
                json={"query": question, "k": k},
                timeout=15,
            )
            resp.raise_for_status()
            data    = resp.json()
            results = (
                data.get("results") or
                data.get("documents") or
                data.get("chunks") or []
            )
            if not results:
                return [], 0.0, False
            top_score = max(float(r.get("score", 0)) for r in results)
            return results, top_score, True
        except requests.exceptions.ConnectionError:
            return [], 0.0, False
        except Exception:
            return [], 0.0, False

    def ping(self) -> bool:
        """Return True if the source node is reachable."""
        try:
            r = requests.get(f"{self.url}/health", timeout=5)
            return r.status_code < 500
        except Exception:
            pass
        try:
            r = requests.post(
                f"{self.url}/query",
                json={"query": "test", "k": 1},
                timeout=5,
            )
            return r.status_code < 500
        except Exception:
            return False


def _build_real_sources(api_key: str = API_KEY) -> List[RealSource]:
    sources = [
        RealSource(nid, url, api_key)
        for nid, url in DATA_SOURCE_URLS.items()
    ]
    for src in sources:
        try:
            _LEDGER.register(src.node_id)
        except Exception:
            pass

    # Seed the local ledger's starting score from the real on-chain value so
    # SFAMitigation's pre-suspicion routing tie-breaker (_prioritised(), sorts
    # by (-ledger.score(...))) reflects genuine blockchain trust rather than
    # every node starting tied at INIT_SCORE. Local hit/miss dynamics still
    # take over from there during the run.
    onchain = get_onchain_reliability_scores([s.node_id for s in sources])
    for src in sources:
        if src.node_id in onchain:
            try:
                _LEDGER._scores[src.node_id] = onchain[src.node_id]
            except Exception:
                pass
    return sources


def _check_sources_reachable(sources: List[RealSource]) -> bool:
    ok = [s for s in sources if s.ping()]
    if not ok:
        print("\n  [!] No Docker data-source nodes are reachable.")
        print("      Start them with:  docker compose up -d  (in drag_data_source/)")
        print("      Or run with --mode mock for an offline simulation.\n")
        return False
    print(f"  [+] Reachable nodes: {[s.node_id for s in ok]}")
    if len(ok) < len(sources):
        unreachable = [s.node_id for s in sources if not s.ping()]
        print(f"  [!] Unreachable     : {unreachable}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  SQuAD loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_corpus_contexts() -> set:
    """
    Return the set of context passages actually served by the Docker data
    sources. Every source (sources_0/20/100.jsonl) shares the same
    document indices — sources_0.jsonl is the 0%-polluted (clean) copy, so
    its "html" text is the ground-truth passage for each doc.
    """
    path = os.path.join(_ROOT, "data", "polluted_token", "sources_0.jsonl")
    contexts = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            html = rec.get("html", "")
            if html:
                contexts.add(html)
    return contexts


def load_squad_questions(n: int = 50, seed: int = 42) -> List[Dict[str, str]]:
    """
    Return n unique SQuAD QA pairs whose context passage is one of the
    documents actually loaded into the running Docker data sources.

    The data sources are seeded from the first N unique contexts of the
    SQuAD *train* split (see data/polluted_token/sources_0.jsonl). Pulling
    questions from the *validation* split — as this used to do — asks
    about articles that were never loaded into any source, so every
    question was unanswerable regardless of attack strength (baseline
    accuracy pinned near 0).
    """
    from datasets import load_dataset  # type: ignore
    print(f"  [SQuAD] Loading rajpurkar/squad train split ...")
    ds = load_dataset("rajpurkar/squad", split="train")

    corpus_contexts = _load_corpus_contexts()
    print(f"  [SQuAD] Matching questions against {len(corpus_contexts)} loaded source documents ...")

    seen, pairs = set(), []
    for item in ds:
        if item["context"] not in corpus_contexts:
            continue
        q = item["question"].strip()
        answers = item.get("answers", {})
        texts   = answers.get("text", []) if isinstance(answers, dict) else []
        a = texts[0].strip() if texts else ""
        if q not in seen and a:
            seen.add(q)
            pairs.append({"question": q, "answer": a})

    if not pairs:
        raise RuntimeError(
            "No SQuAD questions matched the loaded source documents — "
            "check data/polluted_token/sources_0.jsonl."
        )

    rng  = np.random.default_rng(seed)
    idxs = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
    selected = [pairs[i] for i in sorted(idxs)]
    print(f"  [SQuAD] Loaded {len(selected)} QA pairs answerable from the running corpus (seed={seed})")
    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  Accuracy evaluation helper
# ══════════════════════════════════════════════════════════════════════════════

def _answer_in_context(answer: str, results: List[dict]) -> bool:
    """
    True if the SQuAD ground-truth answer appears (case-insensitive substring)
    in any of the retrieved document texts.
    """
    ans_lower = answer.lower()
    for r in results:
        text = (r.get("text") or r.get("content") or
                r.get("page_content") or r.get("passage") or "")
        if ans_lower in text.lower():
            return True
    return False


def _evaluate(
    sources:  List[RealSource],
    qa_pairs: List[Dict[str, str]],
    k:        int = 5,
    max_hops: int = 3,
) -> Dict[str, Any]:
    """
    Query all (non-dropped) sources for each question; collect all results;
    check if ground-truth answer appears in the aggregated context.
    Returns accuracy metrics dict.
    """
    hits       = 0
    total_docs = 0
    per_q: List[dict] = []

    for qa in qa_pairs:
        q, ans    = qa["question"], qa["answer"]
        all_results: List[dict] = []

        for src in sources[:max_hops]:
            try:
                results, score, got_hit = src.query(q, k=k)
            except Exception:
                results, got_hit = [], False
            if got_hit:
                all_results.extend(results)

        found = _answer_in_context(ans, all_results)
        if found:
            hits += 1
        total_docs += len(all_results)
        per_q.append({"question": q, "answer": ans,
                       "docs_retrieved": len(all_results), "answer_found": found})

    accuracy = hits / len(qa_pairs) if qa_pairs else 0.0
    return {
        "accuracy":       round(accuracy, 4),
        "hits":           hits,
        "total":          len(qa_pairs),
        "avg_docs_per_q": round(total_docs / max(len(qa_pairs), 1), 2),
        "per_question":   per_q,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Safe wrappers around SelectiveForwardingAttack attributes
#  (attribute names differ across versions of selective_forward_attack.py)
# ══════════════════════════════════════════════════════════════════════════════

def _get_compromised_nodes(atk: Any) -> List[str]:
    """Return list of compromised node IDs regardless of attribute name."""
    for attr in ("compromised_nodes", "compromised", "attacked_nodes",
                 "malicious_nodes", "drop_nodes"):
        val = getattr(atk, attr, None)
        if val is not None:
            return list(val)
    return []


def _get_num_compromised(atk: Any) -> int:
    nodes = _get_compromised_nodes(atk)
    if nodes:
        return len(nodes)
    for attr in ("num_compromised", "n_compromised", "num_attacked"):
        val = getattr(atk, attr, None)
        if val is not None:
            return int(val)
    return 0


def _get_effective_drop_rate(atk: Any) -> float:
    for attr in ("effective_drop_rate", "drop_rate", "drop_prob",
                 "gray_hole_rate", "drop_fraction"):
        val = getattr(atk, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


def _get_attack_report(atk: Any) -> Dict[str, Any]:
    for mname in ("report", "summary", "get_report"):
        fn = getattr(atk, mname, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return {
        "compromised_nodes": _get_compromised_nodes(atk),
        "num_compromised":   _get_num_compromised(atk),
    }


def _apply_attack(atk: Any, sources: List[RealSource],
                  ssm_scores: Dict[str, float]) -> None:
    """
    Call atk.apply() — try different signatures used across versions.
    """
    apply_fn = getattr(atk, "apply", None)
    if apply_fn is None:
        print("  [warn] SelectiveForwardingAttack has no apply() method")
        return
    # Try apply(sources, ssm_scores)
    try:
        apply_fn(sources, ssm_scores)
        return
    except TypeError:
        pass
    # Try apply(sources)
    try:
        apply_fn(sources)
        return
    except TypeError:
        pass
    # Try apply(sources, scores=ssm_scores)
    try:
        apply_fn(sources, scores=ssm_scores)
        return
    except Exception as e:
        print(f"  [warn] apply() failed: {e}")


def _make_sfa_attack(ratio: float, strategy: str, seed: int) -> Optional[Any]:
    if _SFA_CLASS is None:
        print("  [!] SelectiveForwardingAttack not available — skipping attack.")
        return None
    # Try various constructor signatures
    for kwargs in [
        {"attack_ratio": ratio, "strategy": strategy, "drop_rate": "stealthy", "seed": seed},
        {"attack_ratio": ratio, "strategy": strategy, "seed": seed},
        {"ratio": ratio,        "strategy": strategy, "seed": seed},
        {"attack_ratio": ratio, "seed": seed},
    ]:
        try:
            return _SFA_CLASS(**kwargs)
        except TypeError:
            continue
    print("  [warn] Could not construct SelectiveForwardingAttack — check __init__ signature")
    return None


def _make_detector(sources: List[RealSource]) -> Optional[Any]:
    if _SFA_DETECTOR is None:
        return None
    try:
        det = _SFA_DETECTOR()
        for src in sources:
            for mname in ("register", "add_node", "track"):
                fn = getattr(det, mname, None)
                if callable(fn):
                    try:
                        fn(src.node_id)
                    except Exception:
                        pass
                    break
        return det
    except Exception as e:
        print(f"  [warn] SFADetector init failed: {e}")
        return None


def _detector_observe(det: Any, node_id: str, got_hit: bool) -> None:
    for mname in ("observe", "record", "update", "log_hit"):
        fn = getattr(det, mname, None)
        if callable(fn):
            try:
                fn(node_id, got_hit)
                return
            except Exception:
                pass


def _detector_report(det: Any) -> Dict[str, Any]:
    for mname in ("report", "summary", "get_report"):
        fn = getattr(det, mname, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return {}


def _mitigation_route(mit: Any, question: str, max_hops: int) -> Tuple[bool, int]:
    """
    Call mitigation.route() and return (hit, hops_used).
    Handles (hit, hops, log) and (hit,) and just bool -- hops defaults to
    max_hops when the underlying route() doesn't report it, so TTL
    exhaustion accounting still degrades gracefully instead of silently
    reporting 0 hops on a miss.
    """
    route_fn = getattr(mit, "route", None)
    if route_fn is None:
        return False, max_hops
    try:
        result = route_fn(question, max_hops=max_hops)
    except TypeError:
        try:
            result = route_fn(question)
        except Exception:
            return False, max_hops
    except Exception:
        return False, max_hops
    # Unpack result
    if isinstance(result, (tuple, list)):
        hit  = bool(result[0])
        hops = int(result[1]) if len(result) > 1 else max_hops
        return hit, hops
    return bool(result), max_hops


def _make_mitigation(sources: List[RealSource], detector: Any) -> Optional[Any]:
    if _SFA_MITIGATION is None:
        return None
    for args, kwargs in [
        ((sources, detector, _LEDGER), {"redundancy_k": 2}),
        ((sources, detector),          {"redundancy_k": 2}),
        ((sources,),                   {}),
    ]:
        try:
            return _SFA_MITIGATION(*args, **kwargs)
        except TypeError:
            continue
    print("  [warn] SFAMitigation constructor signature not matched")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Mock source (offline fallback)
# ══════════════════════════════════════════════════════════════════════════════

class _MockSource:
    def __init__(self, node_id: str, rng: np.random.Generator) -> None:
        self.node_id = node_id
        self._rng    = rng

    def query(self, question: str, k: int = 5):
        score = float(self._rng.beta(2, 3))
        hit   = score >= 0.45
        results = [{"id": "mock", "text": "mock answer context", "score": score}] if hit else []
        return results, score, hit


def _make_mock_sources(n: int, seed: int) -> List[_MockSource]:
    rng     = np.random.default_rng(seed)
    sources = [
        _MockSource(f"node_{i:02d}", np.random.default_rng(rng.integers(0, 2**31)))
        for i in range(n)
    ]
    for src in sources:
        try:
            _LEDGER.register(src.node_id)
        except Exception:
            pass
    return sources


# ══════════════════════════════════════════════════════════════════════════════
#  Run modes
# ══════════════════════════════════════════════════════════════════════════════

def run_mock(args) -> dict:
    """Offline baseline — no Docker, synthetic sources."""
    print("\n[mock] Offline baseline (MockSource — no Docker)")
    sources  = _make_mock_sources(args.nodes, args.seed)
    queries  = [{"question": f"mock q {i}", "answer": "mock answer context"}
                for i in range(args.n_questions)]
    baseline = _evaluate(sources, queries, k=args.k, max_hops=args.max_hops)
    print(f"  Accuracy (baseline): {baseline['accuracy']:.4f}")
    bc = check_blockchain_status()
    print(f"  Blockchain         : {bc}")
    return {"mode": "mock", "baseline": baseline}


def run_baseline(args) -> dict:
    """Real Docker baseline — no attack."""
    print("\n[baseline] No attack — measuring context retrieval accuracy")
    sources = _build_real_sources(args.api_key)
    if not _check_sources_reachable(sources):
        return {}
    qa_pairs = load_squad_questions(args.n_questions, args.seed)
    result   = _evaluate(sources, qa_pairs, k=args.k, max_hops=args.max_hops)
    print(f"  Accuracy : {result['accuracy']:.4f}  ({result['hits']}/{result['total']})")
    print(f"  Avg docs : {result['avg_docs_per_q']:.1f} per question")
    return {"mode": "baseline", "metrics": result}


def run_stealthy(args) -> dict:
    """Stealthy gray-hole SFA."""
    print(f"\n[stealthy] SFA ratio={args.ratio}  strategy={args.strategy}")
    sources  = _build_real_sources(args.api_key)
    if not _check_sources_reachable(sources):
        return {}

    qa_pairs = load_squad_questions(args.n_questions, args.seed)
    print("  Running baseline ...")
    baseline = _evaluate(sources, qa_pairs, k=args.k, max_hops=args.max_hops)

    atk = _make_sfa_attack(args.ratio, args.strategy, args.seed + 999)
    if atk is None:
        return {"mode": "stealthy", "error": "SFA class unavailable"}

    ssm_scores = (get_onchain_reliability_scores([s.node_id for s in sources])
                  or {s.node_id: _LEDGER.score(s.node_id) for s in sources})
    _apply_attack(atk, sources, ssm_scores)

    print("  Running under attack ...")
    attacked = _evaluate(sources, qa_pairs, k=args.k, max_hops=args.max_hops)

    compromised     = _get_compromised_nodes(atk)
    acc_drop        = round(baseline["accuracy"] - attacked["accuracy"], 4)
    eff_drop        = _get_effective_drop_rate(atk)
    dropped_queries = _get_attack_report(atk).get("total_dropped", 0)
    # _evaluate() has no early-stop routing -- every query spends the full
    # max_hops budget, so avg_hops_per_query is fixed and TTL exhaustion is
    # exactly "no hit after using the whole budget" (i.e. a miss).
    avg_hops_per_query  = float(args.max_hops)
    ttl_exhaustion_rate = round(1.0 - attacked["accuracy"], 4)

    print(f"\n  Compromised nodes : {compromised}")
    print(f"  Baseline accuracy : {baseline['accuracy']:.4f}")
    print(f"  Attacked accuracy (hit_rate) : {attacked['accuracy']:.4f}")
    print(f"  Accuracy drop     : {acc_drop:+.4f}")
    print(f"  Effective drop %  : {eff_drop*100:.1f}%")
    print(f"  Dropped queries   : {dropped_queries}")
    print(f"  TTL exhaustion rate : {ttl_exhaustion_rate:.4f}")
    bc = check_blockchain_status()
    print(f"  Blockchain        : chain_valid={bc.get('chain_valid')}  "
          f"len={bc.get('chain_length')}")

    return {
        "mode":                 "stealthy",
        "attack_config":        _get_attack_report(atk),
        "baseline":             baseline["accuracy"],
        "attacked":             attacked["accuracy"],
        "hit_rate":             round(attacked["accuracy"], 4),
        "acc_drop":             acc_drop,
        "avg_hops_per_query":   avg_hops_per_query,
        "ttl_exhaustion_rate":  ttl_exhaustion_rate,
        "dropped_queries":      dropped_queries,
    }


def run_detection(args) -> dict:
    """Stealthy SFA + EWMA anomaly detection."""
    print(f"\n[detection] SFA ratio={args.ratio}  strategy={args.strategy}")
    sources  = _build_real_sources(args.api_key)
    if not _check_sources_reachable(sources):
        return {}

    detector = _make_detector(sources)

    atk = _make_sfa_attack(args.ratio, args.strategy, args.seed + 999)
    if atk is None:
        return {"mode": "detection", "error": "SFA class unavailable"}

    ssm_scores = (get_onchain_reliability_scores([s.node_id for s in sources])
                  or {s.node_id: _LEDGER.score(s.node_id) for s in sources})
    _apply_attack(atk, sources, ssm_scores)

    qa_pairs = load_squad_questions(args.n_questions, args.seed)
    hits = 0
    for qa in qa_pairs:
        q = qa["question"]
        all_results: List[dict] = []
        for src in sources[:args.max_hops]:
            try:
                results, score, got_hit = src.query(q, k=args.k)
            except Exception:
                results, got_hit = [], False
            if detector is not None:
                _detector_observe(detector, src.node_id, got_hit)
            if got_hit:
                all_results.extend(results)
        if _answer_in_context(qa["answer"], all_results):
            hits += 1

    accuracy    = hits / len(qa_pairs)
    det_report  = _detector_report(detector) if detector else {}
    compromised = _get_compromised_nodes(atk)
    suspected   = [nid for nid, v in det_report.items()
                   if isinstance(v, dict) and v.get("suspected_drop")]
    caught      = [n for n in suspected if n in compromised]
    n_comp      = _get_num_compromised(atk)

    print(f"\n  Accuracy      : {accuracy:.4f}")
    print(f"  Compromised   : {compromised}")
    print(f"  Suspected     : {suspected}")
    print(f"  Detection     : {len(caught)}/{n_comp} caught")
    if det_report:
        print("\n  Per-node report:")
        for nid, v in sorted(det_report.items()):
            if isinstance(v, dict):
                flag     = " *** SUSPECTED" if v.get("suspected_drop") else ""
                miss     = v.get("miss_rate", v.get("drop_rate", "?"))
                suspicion = v.get("suspicion", v.get("suspicion_level", "?"))
                print(f"    {nid}  miss={miss}  susp={suspicion}{flag}")

    return {
        "mode":           "detection",
        "accuracy":       round(accuracy, 4),
        "compromised":    compromised,
        "suspected":      suspected,
        "detection_rate": len(caught) / max(n_comp, 1),
    }


def run_mitigation(args) -> dict:
    """Full pipeline — attack + detection + SSM-aware re-routing."""
    print(f"\n[mitigation] SFA ratio={args.ratio}  strategy={args.strategy}")
    sources  = _build_real_sources(args.api_key)
    if not _check_sources_reachable(sources):
        return {}

    detector = _make_detector(sources)

    atk = _make_sfa_attack(args.ratio, args.strategy, args.seed + 999)
    if atk is None:
        return {"mode": "mitigation", "error": "SFA class unavailable"}

    ssm_scores = (get_onchain_reliability_scores([s.node_id for s in sources])
                  or {s.node_id: _LEDGER.score(s.node_id) for s in sources})
    _apply_attack(atk, sources, ssm_scores)

    mitigation = _make_mitigation(sources, detector)
    qa_pairs   = load_squad_questions(args.n_questions, args.seed)

    if mitigation is not None:
        hits, hops_used, exhausted = 0, [], 0
        for qa in qa_pairs:
            hit, hops = _mitigation_route(mitigation, qa["question"], args.max_hops)
            hops_used.append(hops)
            if hit:
                hits += 1
            elif hops >= args.max_hops:
                exhausted += 1
        accuracy             = hits / len(qa_pairs)
        avg_hops_per_query   = sum(hops_used) / len(hops_used) if hops_used else 0.0
        ttl_exhaustion_rate  = exhausted / len(qa_pairs) if qa_pairs else 0.0

        blacklisted = []
        suspected   = []
        active      = len(sources)
        for mname, dest in [("blacklisted_ids", blacklisted),
                            ("suspected_ids",   suspected)]:
            fn = getattr(mitigation, mname, None)
            if callable(fn):
                try:
                    dest.extend(fn())
                except Exception:
                    pass
        fn = getattr(mitigation, "active_count", None)
        if callable(fn):
            try:
                active = fn()
            except Exception:
                pass
    else:
        # Fallback: plain evaluation (no re-routing)
        result              = _evaluate(sources, qa_pairs, k=args.k, max_hops=args.max_hops)
        accuracy            = result["accuracy"]
        avg_hops_per_query  = float(args.max_hops)  # _evaluate() always queries all max_hops sources
        ttl_exhaustion_rate = 1.0 - accuracy         # no early-stop routing => "exhausted" iff missed
        blacklisted, suspected, active = [], [], len(sources)

    dropped_queries = _get_attack_report(atk).get("total_dropped", 0)

    # Fair comparison point: naive_route() uses the identical first-hit
    # routing mechanism and hop budget as SFAMitigation.route(), just
    # without suspicion-awareness/blacklisting -- run against a *fresh*
    # source set with an equivalent (same ratio/strategy/seed) attack
    # instance applied, so this isolates what the suspicion-aware routing
    # itself contributes, rather than comparing against _evaluate()'s
    # unrelated aggregate-all-sources accuracy (see defense/sfa_defense/README.md).
    naive_accuracy = None
    naive_avg_hops_per_query = None
    naive_ttl_exhaustion_rate = None
    naive_dropped_queries = None
    if _NAIVE_ROUTE is not None:
        naive_sources = _build_real_sources(args.api_key)
        naive_atk = _make_sfa_attack(args.ratio, args.strategy, args.seed + 999)
        if naive_atk is not None:
            naive_ssm = (get_onchain_reliability_scores([s.node_id for s in naive_sources])
                         or {s.node_id: _LEDGER.score(s.node_id) for s in naive_sources})
            _apply_attack(naive_atk, naive_sources, naive_ssm)
            naive_hits, naive_hops_used, naive_exhausted = 0, [], 0
            for qa in qa_pairs:
                hit, hops, _ = _NAIVE_ROUTE(naive_sources, qa["question"], max_hops=args.max_hops)
                naive_hops_used.append(hops)
                if hit:
                    naive_hits += 1
                elif hops >= args.max_hops:
                    naive_exhausted += 1
            naive_accuracy = naive_hits / len(qa_pairs)
            naive_avg_hops_per_query = sum(naive_hops_used) / len(naive_hops_used) if naive_hops_used else 0.0
            naive_ttl_exhaustion_rate = naive_exhausted / len(qa_pairs) if qa_pairs else 0.0
            naive_dropped_queries = _get_attack_report(naive_atk).get("total_dropped", 0)

    bc = check_blockchain_status()
    print(f"\n  Naive routing accuracy (same attack, no mitigation) : "
          f"{naive_accuracy if naive_accuracy is None else f'{naive_accuracy:.4f}'}")
    print(f"  Mitigated routing accuracy                          : {accuracy:.4f}")
    print(f"  Mitigated hit_rate / avg_hops / ttl_exhaustion      : "
          f"{accuracy:.4f} / {avg_hops_per_query:.2f} / {ttl_exhaustion_rate:.4f}")
    if naive_accuracy is not None:
        print(f"  Naive     hit_rate / avg_hops / ttl_exhaustion      : "
              f"{naive_accuracy:.4f} / {naive_avg_hops_per_query:.2f} / {naive_ttl_exhaustion_rate:.4f}")
    print(f"  Dropped queries (mitigated run / naive run)         : "
          f"{dropped_queries} / {naive_dropped_queries}")
    print(f"  Blacklisted                : {blacklisted}")
    print(f"  Suspected                  : {suspected}")
    print(f"  Active nodes               : {active}/{len(sources)}")
    print(f"  Blockchain                 : valid={bc.get('chain_valid')}  "
          f"len={bc.get('chain_length')}  blacklisted={bc.get('num_blacklisted')}")

    return {
        "mode":                        "mitigation",
        "accuracy":                    round(accuracy, 4),
        "hit_rate":                    round(accuracy, 4),
        "avg_hops_per_query":          round(avg_hops_per_query, 4),
        "ttl_exhaustion_rate":         round(ttl_exhaustion_rate, 4),
        "dropped_queries":             dropped_queries,
        "naive_routing_accuracy":      round(naive_accuracy, 4) if naive_accuracy is not None else None,
        "naive_hit_rate":              round(naive_accuracy, 4) if naive_accuracy is not None else None,
        "naive_avg_hops_per_query":    round(naive_avg_hops_per_query, 4) if naive_avg_hops_per_query is not None else None,
        "naive_ttl_exhaustion_rate":   round(naive_ttl_exhaustion_rate, 4) if naive_ttl_exhaustion_rate is not None else None,
        "naive_dropped_queries":       naive_dropped_queries,
        "blacklisted":                 blacklisted,
        "suspected":                   suspected,
    }


def run_sweep(args) -> dict:
    """
    Sweep all ratios × strategies — produces the thesis accuracy-vs-ratio table.
    Saves a JSON log and prints a formatted table.
    """
    print("\n[sweep] Ratio × Strategy sweep (thesis table)")
    sources = _build_real_sources(args.api_key)
    if not _check_sources_reachable(sources):
        return {}

    qa_pairs = load_squad_questions(args.n_questions, args.seed)

    print("\n  Running baseline ...")
    baseline_acc = _evaluate(sources, qa_pairs, k=args.k, max_hops=args.max_hops)["accuracy"]
    print(f"  Baseline accuracy: {baseline_acc:.4f}")

    rows: List[dict] = []
    for strategy in SWEEP_STRATEGIES:
        for ratio in SWEEP_RATIOS:
            if ratio == 0.0:
                rows.append({
                    "strategy":      "baseline",
                    "ratio":         0.0,
                    "n_compromised": 0,
                    "accuracy":      baseline_acc,
                    "acc_drop":      0.0,
                })
                continue

            src_fresh  = _build_real_sources(args.api_key)
            atk        = _make_sfa_attack(ratio, strategy, args.seed + 999)
            n_comp     = 0
            if atk is not None:
                ssm_s = (get_onchain_reliability_scores([s.node_id for s in src_fresh])
                         or {s.node_id: _LEDGER.score(s.node_id) for s in src_fresh})
                _apply_attack(atk, src_fresh, ssm_s)
                n_comp = _get_num_compromised(atk)

            res  = _evaluate(src_fresh, qa_pairs, k=args.k, max_hops=args.max_hops)
            drop = round(baseline_acc - res["accuracy"], 4)
            rows.append({
                "strategy":      strategy,
                "ratio":         ratio,
                "n_compromised": n_comp,
                "accuracy":      res["accuracy"],
                "acc_drop":      drop,
            })
            print(f"  {strategy:20s}  ratio={ratio:.1f}  "
                  f"compromised={n_comp}  "
                  f"acc={res['accuracy']:.4f}  drop={drop:+.4f}")

    print("\n" + "=" * 72)
    print(f"  {'Strategy':<22} {'Ratio':>6} {'Compromised':>12} "
          f"{'Accuracy':>10} {'Acc Drop':>10}")
    print("=" * 72)
    for r in rows:
        print(f"  {r['strategy']:<22} {r['ratio']:>6.1f} {r['n_compromised']:>12} "
              f"{r['accuracy']:>10.4f} {r['acc_drop']:>+10.4f}")
    print("=" * 72)

    ts       = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOG_DIR, f"attack_{ts}_sfa_sweep.json")
    log = {
        "attack":       "selective_forwarding",
        "timestamp":    ts,
        "dataset":      "rajpurkar/squad",
        "n_questions":  args.n_questions,
        "seed":         args.seed,
        "baseline_acc": baseline_acc,
        "rows":         rows,
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Log saved -> {log_path}")
    return log


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Selective Forwarding Attack — Reliable-dRAG (real Docker)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", default="sweep",
                   choices=["mock", "baseline", "stealthy", "detection",
                            "mitigation", "sweep"],
                   help="Execution mode (default: sweep)")
    p.add_argument("--ratio",        type=float, default=0.34,
                   help="Attack ratio 0-1 (stealthy/detection/mitigation modes)")
    p.add_argument("--strategy",     default="random",
                   choices=["random", "high_ssm_score"],
                   help="Node selection strategy")
    p.add_argument("--n_questions",  type=int, default=50,
                   help="SQuAD questions per evaluation (default: 50)")
    p.add_argument("--k",            type=int, default=5,
                   help="Top-k docs per source query (default: 5)")
    p.add_argument("--max_hops",     type=int, default=3,
                   help="Max sources to query per question (default: 3)")
    p.add_argument("--seed",         type=int, default=42,
                   help="RNG seed (default: 42). Use 0,1,2 for thesis.")
    p.add_argument("--api_key",      default=API_KEY,
                   help="X-API-Key for authenticated source queries")
    p.add_argument("--nodes",        type=int, default=12,
                   help="Number of mock nodes (mock mode only)")
    args = p.parse_args()

    print("=" * 62)
    print("  Selective Forwarding Attack  |  Reliable-dRAG")
    print("=" * 62)
    print(f"  Mode        : {args.mode}")
    print(f"  Sources     : {list(DATA_SOURCE_URLS.keys())}")
    print(f"  Questions   : {args.n_questions}")
    print(f"  Seed        : {args.seed}")

    t0 = time.perf_counter()
    dispatch = {
        "mock":       run_mock,
        "baseline":   run_baseline,
        "stealthy":   run_stealthy,
        "detection":  run_detection,
        "mitigation": run_mitigation,
        "sweep":      run_sweep,
    }
    dispatch[args.mode](args)
    print(f"\n  Elapsed: {time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    main()
