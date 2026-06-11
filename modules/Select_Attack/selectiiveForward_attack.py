import random
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from loguru import logger


class NodeAttack:

    def __init__(self, attack_type='node_removal', attack_ratio=0.3,
                 attack_iterations=5, seed=42):

        self.attack_type = attack_type
        self.attack_ratio = attack_ratio
        self.attack_iterations = attack_iterations
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        self.attacked_nodes: Set[int] = set()
        self.attack_history: List[Dict] = []

        logger.info(f"NodeAttack initialized: type={attack_type}, ratio={attack_ratio}")

    def select_target_nodes(self, total_nodes: int, node_data: Dict = None,
                            strategy: str = "random") -> List[int]:
        num_attack = max(1, int(total_nodes * self.attack_ratio))

        if strategy == "high_connectivity" and node_data and 'connectivity' in node_data:
            connectivity = node_data['connectivity']
            sorted_nodes = sorted(range(total_nodes),
                                  key=lambda i: connectivity.get(i, 0), reverse=True)
            target_nodes = sorted_nodes[:num_attack]
            logger.info(f"HIGH-CONNECTIVITY attack: targeting nodes {target_nodes}")

        elif strategy == "high_data" and node_data and 'data_sizes' in node_data:
            data_sizes = node_data['data_sizes']
            sorted_nodes = sorted(range(total_nodes),
                                  key=lambda i: data_sizes.get(i, 0), reverse=True)
            target_nodes = sorted_nodes[:num_attack]
            logger.info(f"HIGH-DATA attack: targeting nodes {target_nodes}")

        elif strategy == "specific" and node_data and 'target_peers' in node_data:
            target_nodes = node_data['target_peers'][:num_attack]
            logger.info(f"SPECIFIC attack: targeting nodes {target_nodes}")

        else:
            target_nodes = random.sample(range(total_nodes), min(num_attack, total_nodes))
            logger.info(f"RANDOM attack: targeting nodes {target_nodes}")

        self.attacked_nodes.update(target_nodes)
        return target_nodes

    def node_removal_attack(self, nodes: List, node_data: Dict, strategy: str = "random") -> Dict:

        total_nodes = len(nodes)
        target_nodes = self.select_target_nodes(total_nodes, node_data, strategy)

        if 'disabled_nodes' not in node_data:
            node_data['disabled_nodes'] = set()

        removed_count = 0
        for idx in target_nodes:
            if idx < len(nodes) and nodes[idx] is not None:
                node_data['disabled_nodes'].add(idx)
                removed_count += 1

        result = {
            'attack_type': 'node_removal',
            'total_nodes': total_nodes,
            'removed_count': removed_count,
            'removed_indices': target_nodes,
            'disabled_nodes': list(node_data['disabled_nodes']),
            'availability_ratio': (total_nodes - len(node_data['disabled_nodes'])) / total_nodes if total_nodes > 0 else 0,
            'targeting_strategy': strategy
        }

        self.attack_history.append(result)
        logger.info(f"Node removal ({strategy}): {removed_count}/{total_nodes} nodes disabled")

        return result

    def ddos_attack(self, nodes: List, node_data: Dict, strategy: str = "random") -> Dict:

        total_nodes = len(nodes)
        target_nodes = self.select_target_nodes(total_nodes, node_data, strategy)

        if 'ddos_targets' not in node_data:
            node_data['ddos_targets'] = set()

        overloaded_count = 0
        for idx in target_nodes:
            if idx < len(nodes) and nodes[idx] is not None:
                node_data['ddos_targets'].add(idx)
                overloaded_count += 1

        result = {
            'attack_type': 'ddos',
            'total_nodes': total_nodes,
            'overloaded_count': overloaded_count,
            'target_indices': list(target_nodes),
            'overload_ratio': overloaded_count / total_nodes if total_nodes > 0 else 0,
            'targeting_strategy': strategy
        }

        self.attack_history.append(result)
        logger.info(f"DDoS attack ({strategy}): {overloaded_count}/{total_nodes} nodes overloaded")

        return result


# ---------------------------------------------------------------------------
# Selective Forwarding Attack
# ---------------------------------------------------------------------------

class SelectiveForwardingAttack:
    """
    Selective Forwarding Attack on a DRAG P2P RAG network.

    A compromised peer stays connected to the network and remains visible in
    routing tables. When a query is routed to it, it silently drops the query
    — returning (None, None, 0.0, False) instead of knowledge. The hop is
    consumed, burning through the query's TTL budget, but no answer is produced.

    This is distinct from node_removal:
      - node_removal  -> peer is set to None; the network knows it is gone.
      - selective fwd -> peer object stays intact and reachable in the overlay
                         graph; only its query() response is suppressed.
                         Health checks pass. The network cannot detect it is
                         malicious.

    Targeting strategies
    --------------------
    random            : compromise a uniformly random fraction of peers.
    high_connectivity : compromise the highest-degree hub peers first.
                        Hub peers receive the most routed queries, so
                        compromising them maximises hit-rate degradation
                        per compromised peer — the adversarially optimal choice.

    Metrics tracked
    ---------------
    hit_rate           : fraction of queries that returned a non-empty answer.
    avg_hops_per_query : mean hops consumed per query (including wasted hops on
                         compromised peers that produced nothing).
    ttl_exhaustion_rate: fraction of queries that hit max_ttl without an answer.
    dropped_queries    : total query calls silently dropped by compromised peers.
    """

    def __init__(self, attack_ratio: float = 0.3, seed: int = 42):
        """
        Args:
            attack_ratio : Fraction of peers to compromise (0.0 – 1.0).
            seed         : RNG seed for reproducibility.
        """
        self.attack_ratio = attack_ratio
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # peer_id -> original query method (stored so attack can be reverted)
        self._patched_peers: Dict[int, object] = {}
        self.compromised_ids: Set[int] = set()
        self._dropped_queries: int = 0

        logger.info(
            f"SelectiveForwardingAttack initialized: ratio={attack_ratio}, seed={seed}"
        )

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    def select_targets(self, rag_network, strategy: str = "random") -> List[int]:
        """
        Choose which peer IDs to compromise.

        Args:
            rag_network : A DRAG RAGNetwork instance (has .peers and .network).
            strategy    : 'random' | 'high_connectivity'

        Returns:
            List of peer IDs selected for compromise.
        """
        num_peers = rag_network.num_peers
        num_compromise = max(1, int(num_peers * self.attack_ratio))

        if strategy == "high_connectivity":
            # Sort peers by overlay-graph degree (descending).
            # Hub peers receive the most forwarded queries; compromising them
            # wastes the most hop budget per compromised node.
            degree_map = dict(rag_network.network.degree())
            sorted_peers = sorted(degree_map.keys(),
                                  key=lambda p: degree_map[p], reverse=True)
            targets = sorted_peers[:num_compromise]
            logger.info(
                f"HIGH-CONNECTIVITY targeting: top-{num_compromise} hub peers {targets} "
                f"(degrees {[degree_map[p] for p in targets]})"
            )
        else:
            # Uniform random
            targets = random.sample(range(num_peers), min(num_compromise, num_peers))
            logger.info(f"RANDOM targeting: compromised peers {targets}")

        return targets

    # ------------------------------------------------------------------
    # Apply / revert the attack on a live RAGNetwork
    # ------------------------------------------------------------------

    def apply(self, rag_network, strategy: str = "random") -> Dict:
        """
        Monkey-patch query() on selected peers so they silently drop every
        incoming query. The peer object stays in rag_network.peers and in the
        NetworkX overlay graph — it looks healthy from the outside.

        Args:
            rag_network : RAGNetwork instance to attack.
            strategy    : 'random' | 'high_connectivity'

        Returns:
            Summary dict describing the compromised peers.
        """
        if self._patched_peers:
            logger.warning("Attack already applied. Call revert() first to reset.")
            return {}

        targets = self.select_targets(rag_network, strategy)
        self.compromised_ids = set(targets)
        self._dropped_queries = 0

        for peer_id in targets:
            peer = rag_network.peers[peer_id]
            if peer is None:
                logger.warning(f"Peer {peer_id} is None — skipping (already removed).")
                continue

            # Save original query method
            self._patched_peers[peer_id] = peer.query

            # Close over peer_id and self for the black-hole replacement
            attack_ref = self

            def _silent_drop(question, query_confidence_threshold,
                             _pid=peer_id, _atk=attack_ref):
                _atk._dropped_queries += 1
                logger.debug(
                    f"[SFA] Peer {_pid} silently drops query: "
                    f"'{str(question)[:60]}'"
                )
                # Mimic a peer with no relevant knowledge:
                # returns (answer, knowledge, score, is_hit) = all empty/False
                return None, None, 0.0, False

            peer.query = _silent_drop

        degree_map = dict(rag_network.network.degree())
        result = {
            'attack': 'selective_forwarding',
            'strategy': strategy,
            'attack_ratio': self.attack_ratio,
            'num_compromised': len(self._patched_peers),
            'compromised_ids': sorted(self._patched_peers.keys()),
            'compromised_degrees': {
                p: degree_map.get(p, 0) for p in self._patched_peers
            },
        }
        logger.info(
            f"[SFA] Applied: {len(self._patched_peers)}/{rag_network.num_peers} "
            f"peers compromised (strategy={strategy})"
        )
        return result

    def revert(self, rag_network) -> None:
        """
        Restore all patched peers to their original query() methods.
        Call this between experimental runs to reuse the same RAGNetwork
        instance without reloading knowledge bases.
        """
        for peer_id, original_query in self._patched_peers.items():
            peer = rag_network.peers[peer_id]
            if peer is not None:
                peer.query = original_query
                logger.debug(f"[SFA] Peer {peer_id} query() restored.")

        restored = len(self._patched_peers)
        self._patched_peers.clear()
        self.compromised_ids.clear()
        logger.info(f"[SFA] Reverted: {restored} peers restored.")

    # ------------------------------------------------------------------
    # Metric collection
    # ------------------------------------------------------------------

    @property
    def dropped_queries(self) -> int:
        """Total query calls silently dropped since the last apply()."""
        return self._dropped_queries

    def collect_metrics(self, rag_answers: list, max_ttl: int) -> Dict:
        """
        Compute the three primary attack-impact metrics from a list of
        RAGAnswer objects returned by the network after the attack was applied.

        Args:
            rag_answers : List of data_types.RAGAnswer objects.
            max_ttl     : TTL cap used during the run (queries that consumed
                          >= max_ttl hops without a hit are TTL-exhausted).

        Returns:
            Dict with:
              hit_rate            — fraction of answered queries
              avg_hops_per_query  — mean hops consumed (wasted + useful)
              ttl_exhaustion_rate — fraction of queries that hit the TTL wall
              dropped_queries     — raw drop count across all compromised peers
              total_queries       — total queries issued
              answered_queries    — queries that got an answer
              exhausted_queries   — queries that hit max_ttl without an answer
        """
        if not rag_answers:
            return {
                'hit_rate': 0.0,
                'avg_hops_per_query': 0.0,
                'ttl_exhaustion_rate': 0.0,
                'dropped_queries': self._dropped_queries,
                'total_queries': 0,
                'answered_queries': 0,
                'exhausted_queries': 0,
            }

        total = len(rag_answers)
        answered = sum(1 for r in rag_answers if r.answer and r.is_query_hit)
        exhausted = sum(
            1 for r in rag_answers if r.num_hops >= max_ttl and not r.is_query_hit
        )
        avg_hops = float(np.mean([r.num_hops for r in rag_answers]))

        return {
            'hit_rate': answered / total,
            'avg_hops_per_query': avg_hops,
            'ttl_exhaustion_rate': exhausted / total,
            'dropped_queries': self._dropped_queries,
            'total_queries': total,
            'answered_queries': answered,
            'exhausted_queries': exhausted,
        }


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def apply_selective_forwarding(
    rag_network,
    attack_ratio: float,
    strategy: str = "random",
    seed: int = 42,
) -> Tuple["SelectiveForwardingAttack", Dict]:
    """
    One-shot helper: create a SelectiveForwardingAttack, apply it, and return
    both the attack object (needed for revert/metrics) and the apply summary.

    Example
    -------
    attack, info = apply_selective_forwarding(rag_net, 0.3, 'high_connectivity')
    answers = [rag_net.topic_aware_query(q) for q in questions]
    metrics = attack.collect_metrics(answers, max_ttl=6)
    attack.revert(rag_net)
    """
    sfa = SelectiveForwardingAttack(attack_ratio=attack_ratio, seed=seed)
    summary = sfa.apply(rag_network, strategy=strategy)
    return sfa, summary
