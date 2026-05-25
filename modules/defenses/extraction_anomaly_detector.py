from collections import defaultdict
from typing import Dict, Any, Optional, Tuple
from loguru import logger


class ExtractionAnomalyDetector:
    """
    Behavioral anomaly detection defense against Knowledge Base Extraction attacks.

    Legitimate users ask a handful of questions on related topics.  An attacker
    performing systematic KB extraction exhibits a distinctive pattern:
        - High total query volume from a single peer
        - Broad topic coverage (queries span many unrelated topics)
        - High topic-diversity ratio  (unique_topics / total_queries)

    When a peer's behaviour crosses the configured thresholds it is flagged and
    all subsequent queries from that peer are silently dropped, preventing further
    extraction even if the rate limiter has not yet triggered.

    Attack metrics targeted:
        - successful_queries  : 44 → reduced after flagging
        - query_hit_rate      : 1.0 → drops post-flag
        - topic_coverage      : 0.586 → cannot be raised after flag
        - extraction_rate     : 0.52 → bounded by queries allowed before flag
    """

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get('enabled', True)
        self.max_queries_threshold = config.get('max_queries_threshold', 20)
        self.max_unique_topics_threshold = config.get('max_unique_topics_threshold', 6)
        self.topic_diversity_threshold = float(config.get('topic_diversity_threshold', 0.6))
        self.min_queries_for_detection = config.get('min_queries_for_detection', 10)
        self.action = config.get('action', 'block')   # 'block' | 'log_only'

        # Per-peer state: {peer_id: {'total': int, 'topics': set}}
        self._peer_state: Dict[int, Dict] = defaultdict(
            lambda: {'total': 0, 'topics': set()}
        )
        # Flagged peers: {peer_id: reason}
        self._flagged_peers: Dict[int, str] = {}

        # Statistics
        self._total_checks = 0
        self._flagged_count = 0   # queries blocked due to already-flagged peer
        self._new_flags = 0       # number of new peers flagged

        logger.info("ExtractionAnomalyDetector initialized:")
        logger.info(f"  - Max queries threshold      : {self.max_queries_threshold}")
        logger.info(f"  - Max unique topics threshold: {self.max_unique_topics_threshold}")
        logger.info(f"  - Topic diversity threshold  : {self.topic_diversity_threshold:.0%}")
        logger.info(f"  - Min queries for detection  : {self.min_queries_for_detection}")
        logger.info(f"  - Action on detection        : {self.action}")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def check_and_record(
        self, peer_id: int, topic: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Record a query from peer_id for the given topic and decide whether to allow it.

        Args:
            peer_id: ID of the peer initiating the query.
            topic  : Topic inferred for the query (may be None).

        Returns:
            (is_allowed, reason_string)
        """
        if not self.enabled:
            return True, "anomaly_detection_disabled"

        self._total_checks += 1

        # ── Already flagged ───────────────────────────────────────────────
        if peer_id in self._flagged_peers:
            self._flagged_count += 1
            reason = self._flagged_peers[peer_id]
            logger.debug(f"[AnomalyDetector] Peer {peer_id} already flagged: {reason}")
            if self.action == 'block':
                return False, f"flagged_extractor:{reason}"
            # log_only – record but allow
            return True, f"flagged_but_allowed:{reason}"

        # ── Update state ──────────────────────────────────────────────────
        state = self._peer_state[peer_id]
        state['total'] += 1
        if topic:
            state['topics'].add(topic)

        total = state['total']
        unique_topics = len(state['topics'])
        diversity = unique_topics / total if total > 0 else 0.0

        logger.debug(
            f"[AnomalyDetector] Peer {peer_id}: total={total}, "
            f"unique_topics={unique_topics}, diversity={diversity:.2f}"
        )

        # ── Detection (only after min_queries_for_detection) ──────────────
        if total >= self.min_queries_for_detection:
            flag_reason = self._check_thresholds(peer_id, total, unique_topics, diversity)
            if flag_reason:
                self._flagged_peers[peer_id] = flag_reason
                self._new_flags += 1
                logger.warning(
                    f"[AnomalyDetector] EXTRACTION DETECTED – peer {peer_id} FLAGGED: {flag_reason} "
                    f"(total={total}, topics={unique_topics}, diversity={diversity:.2f})"
                )
                if self.action == 'block':
                    self._flagged_count += 1
                    return False, f"newly_flagged:{flag_reason}"
                return True, f"flagged_but_allowed:{flag_reason}"

        return True, "allowed"

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _check_thresholds(
        self, peer_id: int, total: int, unique_topics: int, diversity: float
    ) -> Optional[str]:
        """Return a flag reason string if any threshold is crossed, else None."""
        if total > self.max_queries_threshold:
            return f"high_query_volume({total}>{self.max_queries_threshold})"

        if unique_topics > self.max_unique_topics_threshold:
            return f"broad_topic_coverage({unique_topics}>{self.max_unique_topics_threshold})"

        if diversity >= self.topic_diversity_threshold:
            return (
                f"high_topic_diversity("
                f"{diversity:.2f}>={self.topic_diversity_threshold:.2f})"
            )

        return None

    # ──────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            'name': 'ExtractionAnomalyDetector',
            'enabled': self.enabled,
            'action': self.action,
            'total_checks': self._total_checks,
            'flagged_query_count': self._flagged_count,
            'new_peers_flagged': self._new_flags,
            'flagged_peers': dict(self._flagged_peers),
            'peer_state_summary': {
                pid: {
                    'total_queries': s['total'],
                    'unique_topics': len(s['topics']),
                    'diversity': len(s['topics']) / max(1, s['total']),
                }
                for pid, s in self._peer_state.items()
            },
        }
