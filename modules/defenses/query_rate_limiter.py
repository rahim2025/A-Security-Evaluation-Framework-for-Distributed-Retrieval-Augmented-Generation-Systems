import time
from collections import defaultdict, deque
from typing import Dict, Any, Tuple
from loguru import logger


class QueryRateLimiter:
    """
    Rate-limiting defense against Knowledge Base Extraction attacks.

    Tracks queries per peer within a sliding time window and temporarily blocks
    peers that exceed the threshold, limiting how many datapoints an attacker
    can extract in a single session.

    Attack vector mitigated: high query_hit_rate (1.0) and total_queries_sent (44)
    leading to extraction_rate of 0.52.  By capping allowed queries to
    max_queries_per_window the attacker can extract at most that many unique
    datapoints before being blocked.
    """

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get('enabled', True)
        self.max_queries_per_window = config.get('max_queries_per_window', 20)
        self.time_window_seconds = config.get('time_window_seconds', 300)
        self.block_duration_seconds = config.get('block_duration_seconds', 600)

        # Sliding-window query log per peer
        self._query_log: Dict[int, deque] = defaultdict(deque)
        # Blocked peers → unblock timestamp
        self._blocked_peers: Dict[int, float] = {}

        # Statistics
        self._total_checks = 0
        self._blocked_count = 0
        self._allowed_count = 0
        self._peers_ever_blocked: set = set()

        logger.info("QueryRateLimiter initialized:")
        logger.info(f"  - Max queries/window : {self.max_queries_per_window}")
        logger.info(f"  - Time window        : {self.time_window_seconds}s")
        logger.info(f"  - Block duration     : {self.block_duration_seconds}s")

    def check_and_record(self, peer_id: int) -> Tuple[bool, str]:
        """
        Check whether peer_id may send another query and record the attempt.

        Returns:
            (is_allowed, reason_string)
        """
        if not self.enabled:
            return True, "rate_limiting_disabled"

        self._total_checks += 1
        now = time.time()

        # ── Check active block ────────────────────────────────────────────
        if peer_id in self._blocked_peers:
            if now < self._blocked_peers[peer_id]:
                self._blocked_count += 1
                remaining = self._blocked_peers[peer_id] - now
                logger.debug(f"[RateLimiter] Peer {peer_id} blocked ({remaining:.0f}s remaining)")
                return False, f"peer_blocked_remaining_{remaining:.0f}s"
            else:
                del self._blocked_peers[peer_id]
                logger.info(f"[RateLimiter] Block expired for peer {peer_id}, re-allowing")

        # ── Slide the window ──────────────────────────────────────────────
        window_start = now - self.time_window_seconds
        peer_log = self._query_log[peer_id]
        while peer_log and peer_log[0] < window_start:
            peer_log.popleft()

        # ── Enforce limit ─────────────────────────────────────────────────
        if len(peer_log) >= self.max_queries_per_window:
            block_until = now + self.block_duration_seconds
            self._blocked_peers[peer_id] = block_until
            self._peers_ever_blocked.add(peer_id)
            self._blocked_count += 1
            logger.warning(
                f"[RateLimiter] LIMIT EXCEEDED for peer {peer_id}: "
                f"{len(peer_log)}/{self.max_queries_per_window} queries in window. "
                f"Blocking for {self.block_duration_seconds}s."
            )
            return False, f"rate_limit_exceeded_{len(peer_log)}_queries"

        # ── Allow and record ──────────────────────────────────────────────
        peer_log.append(now)
        self._allowed_count += 1
        remaining_budget = self.max_queries_per_window - len(peer_log)
        logger.debug(
            f"[RateLimiter] Peer {peer_id} allowed "
            f"({len(peer_log)}/{self.max_queries_per_window}, {remaining_budget} remaining)"
        )
        return True, "allowed"

    def get_stats(self) -> Dict[str, Any]:
        return {
            'name': 'QueryRateLimiter',
            'enabled': self.enabled,
            'total_checks': self._total_checks,
            'allowed_count': self._allowed_count,
            'blocked_count': self._blocked_count,
            'block_rate': self._blocked_count / max(1, self._total_checks),
            'currently_blocked_peers': list(self._blocked_peers.keys()),
            'peers_ever_blocked': list(self._peers_ever_blocked),
        }
