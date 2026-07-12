"""
attack/selective_forward_sim/live_network.py

Live counterpart to network_sim.MockRAGNetwork: wraps the 3 real
drag_data_source Docker containers (see docker-compose.yml at repo
root) as peers in a fully-connected overlay, routed through the same
TTL-bounded BFS loop as the mock simulation -- so
SelectiveForwardingAttack / SelectiveForwardingDefense drive the real
system with no mode-specific branching.

"Blockchain maintain": this module does not forge on-chain score
updates. DragScores.feedbackAndUpdateScoreRecords() can only be called
by the llmService address and requires a per-source ECDSA signature
(see drag_contract/contracts/drag_scores.sol) -- that authority
legitimately belongs to drag_llm_service's real query pipeline, not a
standalone attack/defense script. Instead this module *reads* the real
on-chain reliability ledger (view calls only, no private key needed)
via the same bridge pattern already used by
attack/selective_forward/run_attack.py's get_onchain_reliability_scores,
and uses it for degree-analogous "high_connectivity" targeting and for
before/after reporting. The chain keeps getting genuinely maintained by
running real queries through drag_llm_service (docker compose's
llm-service) alongside these attack/defense runs.

Requires `docker compose up -d` at the repo root. On-chain reads
degrade gracefully (empty dict) if the Hardhat node / DragScores
contract isn't reachable or `web3` isn't installed.
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx
import requests

from .network_sim import QueryResult

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

DEFAULT_SOURCE_URLS: Dict[str, str] = {
    "source_0": os.getenv("DS0_URL", "http://localhost:8001"),
    "source_20": os.getenv("DS1_URL", "http://localhost:8002"),
    "source_100": os.getenv("DS2_URL", "http://localhost:8003"),
}
DEFAULT_API_KEY = os.getenv("API_KEY", "reliable-derag-secret-2026")
DEFAULT_BLOCKCHAIN_URL = os.getenv("BLOCKCHAIN_URL", "http://localhost:8545")


def get_onchain_reliability_scores(
    node_ids: List[str], blockchain_url: str = DEFAULT_BLOCKCHAIN_URL
) -> Dict[str, float]:
    """
    Read real reliability scores from the deployed DragScores contract
    (view call only). Returns {} if the chain isn't reachable, `web3`
    isn't installed, or the contract hasn't been deployed yet -- callers
    fall back to unweighted/equal-priority targeting in that case.
    """
    try:
        sys.path.insert(0, os.path.join(ROOT, "drag_python_client"))
        from drag_python_client import DragScoresClient  # noqa: PLC0415

        client = DragScoresClient(project_root=ROOT, provider_url=blockchain_url)
        chain_ids = [nid.replace("source_", "sources_", 1) for nid in node_ids]
        _, rel_scores, _ = client.get_scores_batch(chain_ids)
        return {nid: float(rel_scores[i]) for i, nid in enumerate(node_ids)}
    except Exception as e:
        print(
            f"  [warn] Could not read on-chain reliability scores ({e}); "
            "falling back to unweighted targeting/reporting."
        )
        return {}


class LivePeer:
    """
    Wraps one drag_data_source Docker node with the 4-tuple
    `(answer, knowledge, score, is_hit)` query convention used by
    network_sim.MockPeer, so the same attack/defense classes work
    against both mock and live networks.

    Self-throttled, 429-aware, and failure-mode-tracking. Confirmed live
    (see attack/selective_forward_sim/README.md's "Rate limiting"
    section): each drag_data_source Flask server enforces 60 requests/
    minute per source. A single SFA sweep easily exceeds that in seconds
    (up to ~3 HTTP calls per top-level question -- one per BFS hop --
    times dozens of questions times up to ~11 sweep scenarios), and the
    resulting 429 responses were previously swallowed into the same
    `(None, None, 0.0, False)` "miss" tuple as a genuine no-relevant-
    content response -- silently corrupting every live-mode metric in a
    sweep with no indication in the aggregate JSON/CSV logs that
    anything had gone wrong.

    `error_count` is the general counter to watch, not just
    `rate_limited_count`: raising the server-side rate limit and lowering
    `min_request_interval_s` together (to make a large `--n_questions
    --trials` sweep practical) increases real request concurrency against
    the shared retrieval/embedding backend, which can trade 429s for
    timeouts or 5xx errors instead -- a different failure mode that would
    corrupt `hit_rate` exactly as silently if only `rate_limited_count`
    were checked. `error_count` (a strict superset including
    `rate_limited_count`) covers all of: timeouts, connection errors,
    non-200/non-429 HTTP statuses, and malformed response bodies. A clean
    live run requires `error_count == 0` across every peer, not just
    `rate_limited_count == 0`.
    """

    #: Minimum seconds between requests to this instance -- default keeps
    #: throughput safely under the confirmed 60/min (1.0s) server limit.
    MIN_REQUEST_INTERVAL_S = 1.1

    def __init__(
        self, peer_id: int, node_id: str, url: str, api_key: str = DEFAULT_API_KEY,
        min_request_interval_s: float = MIN_REQUEST_INTERVAL_S,
    ):
        self.peer_id = peer_id
        self.node_id = node_id
        self.url = url
        self._api_key = api_key
        self.min_request_interval_s = min_request_interval_s
        self._last_request_time = 0.0
        self.rate_limited_count = 0  # subset of error_count: specifically HTTP 429
        self.timeout_count = 0
        self.connection_error_count = 0
        self.error_count = 0  # superset: every non-clean-200 outcome, of any kind

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        wait = self.min_request_interval_s - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    def query(
        self, question: str, query_confidence_threshold: float = 0.5, k: int = 5
    ) -> Tuple[Optional[str], Optional[str], float, bool]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key

        for attempt in range(2):  # one retry after respecting a 429's backoff
            self._throttle()
            try:
                resp = requests.post(
                    f"{self.url}/query",
                    headers=headers,
                    json={"query": question, "k": k},
                    timeout=15,
                )
            except requests.exceptions.Timeout:
                self.timeout_count += 1
                self.error_count += 1
                return None, None, 0.0, False
            except requests.exceptions.ConnectionError:
                self.connection_error_count += 1
                self.error_count += 1
                return None, None, 0.0, False
            except Exception:
                self.error_count += 1
                return None, None, 0.0, False

            if resp.status_code == 429:
                self.rate_limited_count += 1
                self.error_count += 1
                if attempt == 0:
                    retry_after = float(resp.headers.get("Retry-After", 2.0))
                    time.sleep(retry_after)
                    continue
                return None, None, 0.0, False
            if resp.status_code != 200:
                self.error_count += 1
                return None, None, 0.0, False

            try:
                data = resp.json()
                results = data.get("results") or data.get("documents") or data.get("chunks") or []
                if not results:
                    return None, None, 0.0, False  # genuine miss, not an error -- a clean 200 with nothing relevant
                top = max(results, key=lambda r: float(r.get("score", 0)))
                score = float(top.get("score", 0))
                text = top.get("text")
                return text, text, score, True
            except Exception:
                self.error_count += 1  # 200 status but an unparseable/malformed body
                return None, None, 0.0, False
        return None, None, 0.0, False
        return None, None, 0.0, False

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.url}/health", timeout=5)
            return r.status_code < 500
        except Exception:
            pass
        try:
            r = requests.post(f"{self.url}/query", json={"query": "test", "k": 1}, timeout=5)
            return r.status_code < 500
        except Exception:
            return False


class LiveRAGNetwork:
    """
    Fully-connected overlay over the real docker-compose data sources
    (source_0/20/100), driven through the same BFS routing loop as
    MockRAGNetwork.topic_aware_query(). Real similarity scores from
    RealSource-style responses aren't calibrated to any particular
    [0, 1] confidence threshold the way the mock simulation's synthetic
    scores are, so -- matching how the rest of this repo already treats
    a RealSource result (see SFADetector's MISS_THRESH comment in
    attack/selective_forward/selective_forward_attack.py) -- a live hit
    is defined as "the source returned at least one document" rather
    than gating on `query_confidence_threshold`.
    """

    def __init__(
        self,
        source_urls: Optional[Dict[str, str]] = None,
        num_query_neighbor: int = 2,
        query_ttl: int = 3,
        api_key: str = DEFAULT_API_KEY,
        blockchain_url: str = DEFAULT_BLOCKCHAIN_URL,
        use_onchain_scores: bool = True,
        seed: int = 0,
        min_request_interval_s: float = LivePeer.MIN_REQUEST_INTERVAL_S,
    ):
        self.source_urls = source_urls or DEFAULT_SOURCE_URLS
        node_ids = list(self.source_urls.keys())
        self.num_peers = len(node_ids)
        self.num_query_neighbor = num_query_neighbor
        self.query_ttl = query_ttl
        self._rng_seed = seed

        self.peers: List[LivePeer] = [
            LivePeer(i, nid, self.source_urls[nid], api_key, min_request_interval_s)
            for i, nid in enumerate(node_ids)
        ]
        self.network = nx.complete_graph(self.num_peers)

        self.onchain_reliability: Dict[str, float] = {}
        if use_onchain_scores:
            self.onchain_reliability = get_onchain_reliability_scores(node_ids, blockchain_url)

        self._sfa_defense = None

    def ping_all(self) -> Dict[str, bool]:
        return {p.node_id: p.ping() for p in self.peers}

    def rate_limited_total(self) -> int:
        """Total 429s observed across all peers -- a *subset* of
        error_total(). Report both, not just this one: raising the
        server-side rate limit and lowering min_request_interval_s
        together (to make a large sweep practical) trades 429s for
        timeouts/5xx under real concurrency, which this alone won't catch
        (see reports/SFA_Security_Analysis_Report.md §12.9)."""
        return sum(p.rate_limited_count for p in self.peers)

    def error_total(self) -> int:
        """Total non-clean-200 outcomes across all peers -- timeouts,
        connection errors, non-200/non-429 HTTP statuses, and malformed
        bodies, in addition to 429s. This is the check that actually
        certifies a live run is clean; rate_limited_total() alone can
        read 0 while a different failure mode is silently corrupting
        hit_rate the same way 429s did (§12.9)."""
        return sum(p.error_count for p in self.peers)

    def topic_aware_query(
        self,
        question: str,
        query_confidence_threshold: float = 0.5,
        start: Optional[int] = None,
    ) -> QueryResult:
        order = sorted(
            range(self.num_peers),
            key=lambda i: -self.onchain_reliability.get(self.peers[i].node_id, 0),
        )
        start_id = start if start is not None else order[0]
        visited = {start_id}   # discovered/enqueued -- controls BFS traversal, NOT "actually queried"
        queried: set = set()   # peers whose .query() was actually invoked in phase 1
        queue = deque([start_id])
        log: List[str] = []
        defense = self._sfa_defense
        hops = 0

        while queue and hops < self.query_ttl:
            current_id = queue.popleft()
            peer = self.peers[current_id]

            if defense is not None and defense.is_peer_blacklisted(current_id):
                defense.record_bypass()
                log.append(f"{peer.node_id}: BYPASS[blacklisted]")
                neighbours = [n for n in self.network.neighbors(current_id) if n not in visited]
                for nid in neighbours[: self.num_query_neighbor]:
                    visited.add(nid)
                    queue.append(nid)
                continue

            hops += 1
            queried.add(current_id)
            answer, knowledge, score, is_hit = peer.query(question, query_confidence_threshold)
            log.append(f"{peer.node_id}: {'HIT' if is_hit else 'MISS'} hop={hops} score={score:.3f}")
            if is_hit:
                return QueryResult(question, answer, True, hops, log)

            neighbours = [n for n in self.network.neighbors(current_id) if n not in visited]
            for nid in neighbours[: self.num_query_neighbor]:
                visited.add(nid)
                queue.append(nid)

        # Phase 2 -- bounded redundant backup probe (defense only). See the
        # identical block in network_sim.MockRAGNetwork.topic_aware_query()
        # for the full rationale on why this excludes by `queried`, not
        # `visited` -- on the live 3-node fully-connected graph, a single
        # failed hop discovers both other peers as neighbours and marks
        # them `visited` immediately, even though query_ttl=1 means they
        # never actually get popped and queried. Excluding by `visited`
        # made Phase 2 see zero candidates in exactly that case.
        if defense is not None and getattr(defense, "redundancy_k", 0) > 0:
            for bid in defense.backup_candidates(queried, list(range(self.num_peers))):
                peer = self.peers[bid]
                if peer is None:
                    continue
                hops += 1
                answer, knowledge, score, is_hit = peer.query(question, query_confidence_threshold)
                defense.record_redundant_probe(is_hit)
                log.append(f"{peer.node_id}: REDUNDANT {'HIT' if is_hit else 'MISS'} hop={hops} score={score:.3f}")
                if is_hit:
                    return QueryResult(question, answer, True, hops, log)

        return QueryResult(question, None, False, hops if hops else self.query_ttl, log)
