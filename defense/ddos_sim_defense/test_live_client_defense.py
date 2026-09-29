"""
defense/ddos_sim_defense/test_live_client_defense.py

Offline unit tests for LiveClientDefense (live_client_defense.py) -- no
Docker/live service needed. Uses a tiny fake network/peer, matching this
project's existing plain-function `test_*` convention (see
drag_llm_service/test_similarity_noise_defense.py).

Usage
-----
  python defense/ddos_sim_defense/test_live_client_defense.py
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from defense.ddos_sim_defense.live_client_defense import LiveClientDefense  # noqa: E402


class _FakePeer:
    """Always succeeds unless `.fail` is set -- lets tests force circuit
    breaker trips deterministically instead of relying on randomness."""

    def __init__(self, peer_id: int):
        self.peer_id = peer_id
        self.fail = False
        self.call_count = 0

    def query(self, question, query_confidence_threshold=0.5, *args, **kwargs):
        self.call_count += 1
        if self.fail:
            return None, None, 0.0, False
        return f"answer[{self.peer_id}]", f"knowledge[{self.peer_id}]", 0.9, True


class _FakeNetwork:
    def __init__(self, n_peers: int):
        self.peers = [_FakePeer(i) for i in range(n_peers)]
        self._sfa_defense = None


def test_disabled_passthrough_when_not_applied():
    net = _FakeNetwork(1)
    result = net.peers[0].query("q")
    assert result[3] is True


def test_apply_wraps_and_remove_restores():
    # remove() re-assigns peer.query = original_fn (a captured bound
    # method) rather than `del`-ing the instance attribute -- the exact
    # same pattern defense/sfa_sim_defense's already-validated
    # SelectiveForwardingDefense.remove() uses, confirmed by inspection.
    # So "query" stays an instance-level attribute in __dict__ even after
    # remove(); what actually matters, and what this checks, is that the
    # BEHAVIOR is fully restored (unwrapped -- no rate limiting/circuit
    # breaking applied anymore) and the network's defense slot is cleared.
    net = _FakeNetwork(1)
    peer = net.peers[0]
    assert "query" not in peer.__dict__, "query should start as a class method, no instance override"

    defense = LiveClientDefense()
    defense.apply(net)
    assert "query" in peer.__dict__, "apply() must install an instance-level wrapper"
    assert net._sfa_defense is defense

    defense.remove(net)
    assert net._sfa_defense is None
    result = peer.query("q")
    assert result[3] is True
    # The restored query must be genuinely unwrapped: draining well past
    # the (now-removed) rate limiter's default capacity (20) must still
    # succeed every time.
    for _ in range(25):
        result = peer.query("q")
    assert result[3] is True, "restored query must not still be rate-limited after remove()"


def test_normal_call_succeeds_and_updates_stats():
    net = _FakeNetwork(1)
    defense = LiveClientDefense({"cache_ttl_seconds": 0})  # disable cache for this test
    defense.apply(net)
    result = net.peers[0].query("q1")
    assert result[3] is True
    stats = defense.get_stats()
    assert stats["total_rate_limited"] == 0
    assert stats["total_circuit_blocked"] == 0


def test_rate_limiter_blocks_after_capacity_exhausted():
    net = _FakeNetwork(1)
    defense = LiveClientDefense({
        "rate_limit_capacity": 3, "rate_limit_refill_per_sec": 0.0,  # no refill during the test
        "cache_ttl_seconds": 0, "circuit_failure_threshold": 1000,  # disable circuit for this test
    })
    defense.apply(net)
    outcomes = [net.peers[0].query(f"q{i}") for i in range(6)]
    successes = sum(1 for o in outcomes if o[3])
    assert successes == 3, f"expected exactly 3 successes (capacity=3, no refill), got {successes}"
    assert defense.get_stats()["total_rate_limited"] == 3


def test_circuit_breaker_opens_after_failures_and_blocks():
    net = _FakeNetwork(1)
    net.peers[0].fail = True
    defense = LiveClientDefense({
        "circuit_failure_threshold": 3, "circuit_window_seconds": 60.0, "circuit_open_seconds": 60.0,
        "rate_limit_capacity": 1000, "cache_ttl_seconds": 0,
    })
    defense.apply(net)
    for _ in range(3):
        net.peers[0].query("q")  # 3 failures -> should open the circuit
    stats_before = defense.get_stats()
    assert stats_before["circuit_states"][0] == "open"

    calls_before = net.peers[0].call_count
    result = net.peers[0].query("q_after_open")
    assert result[3] is False
    assert net.peers[0].call_count == calls_before, "circuit-open call must NOT reach the underlying peer"
    assert defense.get_stats()["total_circuit_blocked"] >= 1


def test_circuit_breaker_half_open_recovers_on_success():
    net = _FakeNetwork(1)
    net.peers[0].fail = True
    defense = LiveClientDefense({
        "circuit_failure_threshold": 2, "circuit_window_seconds": 60.0, "circuit_open_seconds": 0.05,
        "rate_limit_capacity": 1000, "cache_ttl_seconds": 0,
    })
    defense.apply(net)
    for _ in range(2):
        net.peers[0].query("q")
    assert defense.get_stats()["circuit_states"][0] == "open"

    time.sleep(0.08)  # let circuit_open_seconds elapse
    net.peers[0].fail = False  # peer recovers
    result = net.peers[0].query("trial")  # half-open trial request
    assert result[3] is True, "half-open trial should reach the now-healthy peer"
    assert defense.get_stats()["circuit_states"][0] == "closed", "success in half-open must close the circuit"


def test_queue_limit_blocks_over_concurrent_calls():
    """Simulates concurrency by manually incrementing _in_flight (no real
    threads needed for a deterministic check)."""
    net = _FakeNetwork(1)
    defense = LiveClientDefense({
        "max_concurrent_per_peer": 2, "rate_limit_capacity": 1000, "cache_ttl_seconds": 0,
        "circuit_failure_threshold": 1000,
    })
    defense.apply(net)
    defense._in_flight[0] = 2  # already at the concurrency bound
    result = net.peers[0].query("q")
    assert result[3] is False
    assert defense.get_stats()["total_queue_full"] == 1


def test_cache_serves_repeated_identical_query_without_calling_peer():
    net = _FakeNetwork(1)
    defense = LiveClientDefense({"cache_ttl_seconds": 5.0, "rate_limit_capacity": 1000})
    defense.apply(net)
    net.peers[0].query("same question")
    calls_after_first = net.peers[0].call_count
    result = net.peers[0].query("same question")
    assert result[3] is True
    assert net.peers[0].call_count == calls_after_first, "cache hit must not reach the underlying peer"
    assert defense.get_stats()["total_cache_hits"] == 1


def test_cache_expires_after_ttl():
    net = _FakeNetwork(1)
    defense = LiveClientDefense({"cache_ttl_seconds": 0.05, "rate_limit_capacity": 1000})
    defense.apply(net)
    net.peers[0].query("q")
    time.sleep(0.08)
    calls_before = net.peers[0].call_count
    net.peers[0].query("q")
    assert net.peers[0].call_count == calls_before + 1, "expired cache entry must hit the peer again"


def test_is_peer_blacklisted_reflects_rate_and_circuit_state():
    net = _FakeNetwork(1)
    defense = LiveClientDefense({
        "rate_limit_capacity": 1, "rate_limit_refill_per_sec": 0.0, "cache_ttl_seconds": 0,
        "circuit_failure_threshold": 1000,
    })
    defense.apply(net)
    assert defense.is_peer_blacklisted(0) is False
    net.peers[0].query("q1")  # consumes the only token
    assert defense.is_peer_blacklisted(0) is True


def test_backup_candidates_excludes_blacklisted_and_tried():
    net = _FakeNetwork(3)
    defense = LiveClientDefense({"redundancy_k": 2, "rate_limit_capacity": 0, "cache_ttl_seconds": 0})
    defense.apply(net)  # capacity=0 -> every peer immediately blacklisted (no tokens)
    candidates = defense.backup_candidates(tried_ids=[0], all_peer_ids=[0, 1, 2])
    assert candidates == [], "all peers rate-limited to zero capacity -> no viable backup candidates"


_ALL_TESTS = [
    test_disabled_passthrough_when_not_applied,
    test_apply_wraps_and_remove_restores,
    test_normal_call_succeeds_and_updates_stats,
    test_rate_limiter_blocks_after_capacity_exhausted,
    test_circuit_breaker_opens_after_failures_and_blocks,
    test_circuit_breaker_half_open_recovers_on_success,
    test_queue_limit_blocks_over_concurrent_calls,
    test_cache_serves_repeated_identical_query_without_calling_peer,
    test_cache_expires_after_ttl,
    test_is_peer_blacklisted_reflects_rate_and_circuit_state,
    test_backup_candidates_excludes_blacklisted_and_tried,
]

if __name__ == "__main__":
    failures = 0
    for fn in _ALL_TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(_ALL_TESTS) - failures}/{len(_ALL_TESTS)} passed")
    sys.exit(1 if failures else 0)
