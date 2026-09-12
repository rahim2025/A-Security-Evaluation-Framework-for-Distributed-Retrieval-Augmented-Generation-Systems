"""
defense/ddos_sim_defense/live_client_defense.py

A second, independent DDoS countermeasure -- responds to the NAACL
improvement proposal's DDoS item 3: "Implement and test a real live
defense, not only a simulation defense. Evaluate authenticated quotas,
global rate limits, queue limits, circuit breakers, load-aware failover,
caching, and source-health routing."

Relationship to the existing defense/ddos_sim_defense/ddos_defense.py
------------------------------------------------------------------------
`DDoSDefense` (existing, unmodified by this file) is a reputation/
blacklist/bypass/redundant-probe mechanism -- the same shape as
defense/sfa_sim_defense's SFA countermeasure, adapted for congestion
instead of silent-drop. It is duck-type-compatible with
`LiveRAGNetwork` (attack/selective_forward_sim/live_network.py) and so
CAN run against the real 3-node deployment, but reports/ddos_attack.md
(section 13) notes it has never actually been evaluated together with a
genuine concurrent flood (`attack/ddos_sim/live_flood.TrafficFlood`) --
only in the in-process `MockRAGNetwork` simulation. That gap -- combining
the existing defense with real live congestion -- is closed by
`run_live_defense_eval.py`, not by this file.

This file adds a GENUINELY DIFFERENT defense, implementing mechanisms
`DDoSDefense` does not have at all: rate limiting, a circuit-breaker state
machine, a bounded concurrency/queue limit, and a short-TTL response
cache. These are the specific mechanisms the improvement proposal names
that a reputation-EMA approach does not by itself provide -- a
reputation score answers "should I trust this peer," while these four
answer "how much load am I allowed to put on it, right now, regardless
of trust."

Same duck-typed interface as DDoSDefense/SelectiveForwardingDefense
-----------------------------------------------------------------------
`apply(network)` / `remove(network)` / `is_peer_blacklisted(pid)` /
`record_bypass()` / `backup_candidates(tried, all_ids)` /
`record_redundant_probe(hit)` / `get_stats()` -- the exact same slot
`network_sim.MockRAGNetwork.topic_aware_query()` and
`live_network.LiveRAGNetwork.topic_aware_query()` already consult via
`network._sfa_defense`. No changes to either network module, or to
DDoSDefense, were needed -- this is a second, alternative implementation
of the same interface, installable in exactly the same way, so the two
defenses (and "no defense") can be compared head-to-head by an evaluator
that just swaps which one it installs.

The four mechanisms
--------------------
1. Rate limiting (token bucket, per peer) -- "authenticated quotas" /
   "global rate limits". Each peer has a bucket of `rate_limit_capacity`
   tokens, refilling at `rate_limit_refill_per_sec`. A query attempt
   consumes one token; an empty bucket makes `is_peer_blacklisted()`
   return True for that peer until a token regenerates -- the router
   bypasses it for free, exactly like an SFA-blacklisted peer, without
   ever making the call. This is a client-side quota (this client's own
   view of how much load it is willing to put on a peer), not the
   server's own Flask-Limiter (drag_data_source/app/server.py's existing
   60/min per-IP limit) -- complementary to, not a replacement for, that
   server-side control.
2. Circuit breaker -- "circuit breakers". Three states per peer, the
   standard pattern (Nygard, *Release It!*, 2007): CLOSED (normal,
   requests flow through) -> OPEN (too many recent failures; every
   request short-circuited without even attempting the network call,
   for `circuit_open_seconds`) -> HALF_OPEN (one trial request allowed
   after the open window elapses; success closes the circuit again,
   failure re-opens it). Distinct from rate limiting: rate limiting caps
   *volume* regardless of outcome, the circuit breaker reacts to
   *failure rate* regardless of volume.
3. Queue / concurrency limit -- "queue limits". A bounded semaphore per
   peer (`max_concurrent_per_peer`) so this client never has more than
   that many requests in flight against one peer at once -- a call that
   would exceed the bound is treated as an immediate miss (queue full)
   rather than blocking indefinitely, protecting both this client's own
   resources and the (possibly already-struggling) peer from additional
   concurrent pressure.
4. Short-TTL response cache -- "caching". Identical `(peer_id, question)`
   pairs within `cache_ttl_seconds` are served from cache instead of
   hitting the network again -- genuinely reduces load on a peer during
   congestion (fewer real requests reach it at all for repeated/similar
   questions), not just a latency optimization for this client.

What this does NOT include: "load-aware failover" and "source-health
routing" are not separate mechanisms here -- they fall out of
`is_peer_blacklisted()` (rate-limited or circuit-open peers are bypassed,
i.e. failed over away from) and `backup_candidates()` (ranks untried
peers by a combined health score for the bounded redundant-probe fallback)
exactly the way both existing sibling defenses already provide them, reused
rather than reimplemented.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple


class _TokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.tokens = capacity
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
        self.last_refill = now

    def try_consume(self) -> bool:
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class _CircuitBreaker:
    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, failure_threshold: int, window_seconds: float, open_seconds: float):
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.open_seconds = open_seconds
        self.state = self.CLOSED
        self._failure_timestamps: List[float] = []
        self._opened_at: Optional[float] = None

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._failure_timestamps = [t for t in self._failure_timestamps if t >= cutoff]

    def allow_request(self) -> bool:
        now = time.monotonic()
        if self.state == self.OPEN:
            if self._opened_at is not None and now - self._opened_at >= self.open_seconds:
                self.state = self.HALF_OPEN
                return True  # one trial request
            return False
        return True  # CLOSED or HALF_OPEN (trial already granted this call)

    def record_result(self, success: bool) -> None:
        now = time.monotonic()
        if success:
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                self._failure_timestamps.clear()
            return
        self._failure_timestamps.append(now)
        self._prune(now)
        if self.state == self.HALF_OPEN:
            self.state = self.OPEN
            self._opened_at = now
        elif len(self._failure_timestamps) >= self.failure_threshold:
            self.state = self.OPEN
            self._opened_at = now


class LiveClientDefense:
    """
    Parameters
    ----------
    rate_limit_capacity / rate_limit_refill_per_sec : token-bucket sizing,
        per peer.
    circuit_failure_threshold : consecutive-window failures before the
        circuit opens.
    circuit_window_seconds : rolling window the failure count is measured
        over.
    circuit_open_seconds : how long the circuit stays OPEN before allowing
        one HALF_OPEN trial request.
    max_concurrent_per_peer : queue/concurrency bound per peer.
    cache_ttl_seconds : how long an identical (peer, question) response is
        served from cache instead of hitting the network again. 0 disables
        caching.
    redundancy_k : bounded extra backup-probe attempts, same role as the
        sibling defenses' `redundancy_k` (see their docstrings for why this
        must be bounded).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.rate_limit_capacity = float(cfg.get("rate_limit_capacity", 20))
        self.rate_limit_refill_per_sec = float(cfg.get("rate_limit_refill_per_sec", 5.0))
        self.circuit_failure_threshold = int(cfg.get("circuit_failure_threshold", 5))
        self.circuit_window_seconds = float(cfg.get("circuit_window_seconds", 10.0))
        self.circuit_open_seconds = float(cfg.get("circuit_open_seconds", 15.0))
        self.max_concurrent_per_peer = int(cfg.get("max_concurrent_per_peer", 3))
        self.cache_ttl_seconds = float(cfg.get("cache_ttl_seconds", 5.0))
        self.redundancy_k = int(cfg.get("redundancy_k", 2))
        self.max_blacklist_fraction = float(cfg.get("max_blacklist_fraction", 0.5))

        self._buckets: Dict[int, _TokenBucket] = {}
        self._breakers: Dict[int, _CircuitBreaker] = {}
        self._in_flight: Dict[int, int] = {}
        self._cache: Dict[Tuple[int, str], Tuple[float, Any]] = {}
        self._original_query_fns: Dict[int, Any] = {}
        self._num_peers: Optional[int] = None

        # Stats -- feed run_live_defense_eval.py's "defense cost" / "false
        # blocks" metrics directly.
        self._total_bypasses = 0
        self._total_rate_limited = 0
        self._total_circuit_blocked = 0
        self._total_queue_full = 0
        self._total_cache_hits = 0
        self._total_redundant_probes = 0
        self._total_redundant_hits = 0
        self._total_defense_overhead_seconds = 0.0

    # ── health scoring, for is_peer_blacklisted / backup_candidates ──────
    def _peer_health_score(self, peer_id: int) -> float:
        """Higher is healthier. Used only for ranking untried peers in
        backup_candidates() -- routing bypass itself is a hard boolean via
        is_peer_blacklisted(), not this score."""
        breaker = self._breakers.get(peer_id)
        bucket = self._buckets.get(peer_id)
        circuit_penalty = {
            None: 0.0, _CircuitBreaker.CLOSED: 0.0,
            _CircuitBreaker.HALF_OPEN: 0.5, _CircuitBreaker.OPEN: 1.0,
        }[breaker.state if breaker else None]
        tokens_frac = (bucket.tokens / bucket.capacity) if bucket else 1.0
        return tokens_frac - circuit_penalty

    def is_peer_blacklisted(self, peer_id: int) -> bool:
        breaker = self._breakers.get(peer_id)
        if breaker is not None and breaker.state == _CircuitBreaker.OPEN:
            now = time.monotonic()
            if breaker._opened_at is not None and now - breaker._opened_at < breaker.open_seconds:
                return True
        bucket = self._buckets.get(peer_id)
        if bucket is not None:
            bucket._refill()
            if bucket.tokens < 1.0:
                return True
        return False

    def record_bypass(self) -> None:
        self._total_bypasses += 1

    def backup_candidates(self, tried_ids, all_peer_ids: List[int]) -> List[int]:
        tried = set(tried_ids)
        candidates = [
            p for p in all_peer_ids
            if p not in tried and not self.is_peer_blacklisted(p)
        ]
        candidates.sort(key=lambda p: -self._peer_health_score(p))
        return candidates[: self.redundancy_k]

    def record_redundant_probe(self, hit: bool) -> None:
        self._total_redundant_probes += 1
        if hit:
            self._total_redundant_hits += 1

    # ── installation ──────────────────────────────────────────────────
    def apply(self, network) -> None:
        self._num_peers = sum(1 for p in network.peers if p is not None)
        for pid, peer in enumerate(network.peers):
            if peer is None:
                continue
            self._buckets[pid] = _TokenBucket(self.rate_limit_capacity, self.rate_limit_refill_per_sec)
            self._breakers[pid] = _CircuitBreaker(
                self.circuit_failure_threshold, self.circuit_window_seconds, self.circuit_open_seconds,
            )
            self._in_flight[pid] = 0
            original_fn = peer.query
            self._original_query_fns[pid] = original_fn
            peer.query = self._make_wrapper(pid, original_fn)
        network._sfa_defense = self

    def _make_wrapper(self, peer_id: int, orig):
        defense = self

        def _wrapped_query(question, query_confidence_threshold=0.5, *args, **kwargs):
            t0 = time.monotonic()

            # 1. Cache -- served without touching the network, breaker, or
            #    concurrency bound at all (a cache hit costs this peer
            #    nothing).
            if defense.cache_ttl_seconds > 0:
                key = (peer_id, question)
                cached = defense._cache.get(key)
                if cached is not None:
                    cached_at, result = cached
                    if time.monotonic() - cached_at < defense.cache_ttl_seconds:
                        defense._total_cache_hits += 1
                        defense._total_defense_overhead_seconds += time.monotonic() - t0
                        return result

            # 2. Rate limit.
            bucket = defense._buckets[peer_id]
            if not bucket.try_consume():
                defense._total_rate_limited += 1
                defense._total_defense_overhead_seconds += time.monotonic() - t0
                return None, None, 0.0, False

            # 3. Circuit breaker.
            breaker = defense._breakers[peer_id]
            if not breaker.allow_request():
                defense._total_circuit_blocked += 1
                defense._total_defense_overhead_seconds += time.monotonic() - t0
                return None, None, 0.0, False

            # 4. Concurrency / queue limit.
            if defense._in_flight[peer_id] >= defense.max_concurrent_per_peer:
                defense._total_queue_full += 1
                defense._total_defense_overhead_seconds += time.monotonic() - t0
                return None, None, 0.0, False
            defense._in_flight[peer_id] += 1
            defense._total_defense_overhead_seconds += time.monotonic() - t0

            try:
                result = orig(question, query_confidence_threshold, *args, **kwargs)
                succeeded = result[0] is not None
                breaker.record_result(succeeded)
                if defense.cache_ttl_seconds > 0 and succeeded:
                    defense._cache[(peer_id, question)] = (time.monotonic(), result)
                return result
            except Exception:
                breaker.record_result(False)
                raise
            finally:
                defense._in_flight[peer_id] -= 1

        return _wrapped_query

    def remove(self, network) -> None:
        for pid, original_fn in self._original_query_fns.items():
            peer = network.peers[pid]
            if peer is not None:
                peer.query = original_fn
        self._original_query_fns.clear()
        if getattr(network, "_sfa_defense", None) is self:
            network._sfa_defense = None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "defense": "live_client_defense",
            "total_bypasses": self._total_bypasses,
            "total_rate_limited": self._total_rate_limited,
            "total_circuit_blocked": self._total_circuit_blocked,
            "total_queue_full": self._total_queue_full,
            "total_cache_hits": self._total_cache_hits,
            "total_redundant_probes": self._total_redundant_probes,
            "total_redundant_probe_hits": self._total_redundant_hits,
            "total_defense_overhead_seconds": self._total_defense_overhead_seconds,
            "circuit_states": {pid: b.state for pid, b in self._breakers.items()},
            "tokens_remaining": {pid: round(b.tokens, 2) for pid, b in self._buckets.items()},
        }
