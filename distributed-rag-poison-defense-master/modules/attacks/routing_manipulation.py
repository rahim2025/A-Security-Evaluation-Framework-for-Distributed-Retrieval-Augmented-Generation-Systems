import random
from typing import Dict, Any, List

from loguru import logger

from modules.attacks.base_attack import BaseAttack
from modules.rag_network import DRAGNetwork
from modules.data_types import Datapoint


class RoutingManipulationAttack(BaseAttack):
    """
    Source Selection Manipulation Attack on Distributed RAG Systems.

    Malicious peers falsely advertise topic expertise in TARW's routing tables
    (peer_topics / topic_peers), causing queries to be preferentially routed to
    them. Once selected, attackers serve either empty responses (knowledge base
    wiped) or fabricated responses (garbage datapoints injected), degrading
    answer quality without ever accessing honest peers' knowledge bases.

    Attack surface: TARW trusts peers to self-report their topics with no
    authentication or verification, making the routing table trivially spoofable.
    """

    def __init__(
        self,
        attacker_peer_id: int = None,
        attacker_ratio: float = 0.1,
        attack_strategy: str = "random",
        response_type: str = "fabricated",
        topic_claim_ratio: float = 1.0,
    ):
        """
        Initialize the routing manipulation attack.

        Args:
            attacker_peer_id: Specific peer to use as attacker (None = use attacker_ratio)
            attacker_ratio: Fraction of peers turned into attackers when no specific ID given
            attack_strategy: Attacker selection — 'random' or 'high_degree'
            response_type: What attackers serve — 'empty' or 'fabricated'
            topic_claim_ratio: Fraction of all network topics each attacker falsely claims (0–1)
        """
        super().__init__("RoutingManipulation")
        self.attacker_peer_id = attacker_peer_id
        self.attacker_ratio = attacker_ratio
        self.attack_strategy = attack_strategy
        self.response_type = response_type
        self.topic_claim_ratio = min(1.0, max(0.0, topic_claim_ratio))

        self.attacker_peer_ids: List[int] = []
        self.hijacked_topics: Dict[int, List[str]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_attackers(self, network: DRAGNetwork) -> List[int]:
        if self.attacker_peer_id is not None:
            return [self.attacker_peer_id]

        num_attackers = max(1, int(network.num_peers * self.attacker_ratio))

        if self.attack_strategy == "high_degree":
            degree_dict = dict(network.network.degree())
            sorted_peers = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
            return [peer_id for peer_id, _ in sorted_peers[:num_attackers]]

        return random.sample(range(network.num_peers), num_attackers)

    def _generate_garbage_datapoints(
        self,
        topics: List[str],
        data_points: List[Datapoint],
    ) -> List[Datapoint]:
        """
        Generate datapoints whose questions embed close to legitimate queries (high
        retrieval score) but carry a wrong answer from a different datapoint in the
        same topic (answer swap).

        Using a real answer keeps the response semantically coherent so it looks
        plausible to both the LLM and any downstream evaluator — but it is incorrect
        for the actual question. The generate_answer.tmpl instructs the LLM to return
        ref_answer verbatim, so the swapped answer is what the user receives.
        """
        questions_by_topic: Dict[str, List[Datapoint]] = {}
        for dp in data_points:
            questions_by_topic.setdefault(dp.topic, []).append(dp)

        garbage = []
        for topic in topics:
            topic_dps = questions_by_topic.get(topic, [])
            if not topic_dps:
                continue

            sample = random.choice(topic_dps)

            # Prefer a wrong answer from the same topic (maximally confusing)
            wrong_candidates = [dp.answer for dp in topic_dps if dp.answer != sample.answer]
            if not wrong_candidates:
                # Fall back to a different topic
                other_dps = [dp for dp in data_points if dp.topic != topic]
                wrong_candidates = [dp.answer for dp in other_dps] if other_dps else []

            wrong_answer = random.choice(wrong_candidates) if wrong_candidates else sample.answer

            garbage.append(Datapoint(
                topic=topic,
                question=sample.question,
                answer=wrong_answer,
            ))
        return garbage

    # ------------------------------------------------------------------
    # BaseAttack interface
    # ------------------------------------------------------------------

    def execute(self, network: DRAGNetwork, data_points: List[Datapoint]) -> Dict[str, Any]:
        logger.info(f"Executing {self.attack_name} attack...")
        logger.info(f"Strategy: {self.attack_strategy} | Response: {self.response_type} | "
                    f"Topic claim ratio: {self.topic_claim_ratio:.0%}")

        if not network.all_topics:
            logger.warning("No topics registered in network — routing manipulation has no effect.")
            return {
                "num_attacker_peers": 0,
                "attacker_peer_ids": [],
                "topics_hijacked": 0,
                "response_type": self.response_type,
                "topic_claim_ratio": self.topic_claim_ratio,
            }

        self.attacker_peer_ids = self._select_attackers(network)
        logger.info(f"Selected {len(self.attacker_peer_ids)} attacker peer(s): {self.attacker_peer_ids}")

        all_topics = list(network.all_topics)
        num_to_claim = max(1, int(len(all_topics) * self.topic_claim_ratio))
        topics_to_claim = random.sample(all_topics, num_to_claim)

        total_injections = 0
        for peer_id in self.attacker_peer_ids:
            self.hijacked_topics[peer_id] = []

            # --- Inject false topic advertisements ---
            for topic in topics_to_claim:
                if topic not in network.peer_topics[peer_id]:
                    network.peer_topics[peer_id].append(topic)
                    self.hijacked_topics[peer_id].append(topic)
                    total_injections += 1

                if topic not in network.topic_peers:
                    network.topic_peers[topic] = []
                if peer_id not in network.topic_peers[topic]:
                    network.topic_peers[topic].append(peer_id)

            # --- Degrade attacker's response capability ---
            if self.response_type == "empty":
                network.peers[peer_id].knowledge_base.data_points.clear()
                network.peers[peer_id].knowledge_base.embeddings = None
                logger.debug(f"Peer {peer_id}: knowledge base cleared (empty response mode)")

            elif self.response_type == "fabricated":
                # Use only the attacker's own KB — a real malicious peer has no access
                # to data distributed to other peers.
                own_data = list(network.peers[peer_id].knowledge_base.data_points)
                garbage_dps = self._generate_garbage_datapoints(topics_to_claim, own_data)
                for dp in garbage_dps:
                    network.peers[peer_id].add_knowledge(dp)
                logger.debug(f"Peer {peer_id}: injected {len(garbage_dps)} garbage datapoints")

        logger.info(
            f"Routing manipulation complete — {len(self.attacker_peer_ids)} attacker(s), "
            f"{num_to_claim}/{len(all_topics)} topics hijacked, "
            f"{total_injections} false topic registrations injected."
        )

        return {
            "num_attacker_peers": len(self.attacker_peer_ids),
            "attacker_peer_ids": self.attacker_peer_ids,
            "topics_hijacked": num_to_claim,
            "total_topic_registrations_injected": total_injections,
            "response_type": self.response_type,
            "attack_strategy": self.attack_strategy,
            "topic_claim_ratio": self.topic_claim_ratio,
            "hijacked_topics_per_peer": {
                str(pid): topics for pid, topics in self.hijacked_topics.items()
            },
        }

    def evaluate_success(
        self,
        original_metrics: Dict[str, float],
        attacked_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Derive misdirection impact entirely from existing QA metrics — no new
        instrumentation required in the network layer.

        Key indicators:
        - Hit rate drop  → fewer queries resolving against legitimate knowledge
        - F1 / exact-match drop → fabricated or empty answers being served
        - Avg hops increase → queries traversing longer before landing on an honest peer
        """
        def _drop(key):
            orig = original_metrics.get(key, 0.0)
            att = attacked_metrics.get(key, 0.0)
            abs_drop = orig - att
            pct_drop = ((abs_drop / orig) * 100) if orig > 0 else 0.0
            return round(abs_drop, 4), round(pct_drop, 2)

        hit_drop, hit_drop_pct = _drop("avg_query_hit")
        f1_drop, f1_drop_pct = _drop("f1")
        em_drop, em_drop_pct = _drop("exact_match")
        sem_drop, sem_drop_pct = _drop("semantic_similarity")

        orig_hops = original_metrics.get("avg_hops", 0.0)
        att_hops = attacked_metrics.get("avg_hops", 0.0)
        hop_increase = round(att_hops - orig_hops, 4)

        avg_quality_degradation = round((f1_drop_pct + em_drop_pct + sem_drop_pct) / 3.0, 2)

        return {
            "hit_rate_drop": hit_drop,
            "hit_rate_drop_pct": hit_drop_pct,
            "f1_drop": f1_drop,
            "f1_drop_pct": f1_drop_pct,
            "exact_match_drop": em_drop,
            "exact_match_drop_pct": em_drop_pct,
            "semantic_similarity_drop": sem_drop,
            "semantic_similarity_drop_pct": sem_drop_pct,
            "avg_hops_increase": hop_increase,
            "avg_quality_degradation_pct": avg_quality_degradation,
            "original_metrics": original_metrics,
            "attacked_metrics": attacked_metrics,
            "is_successful": avg_quality_degradation > 5.0,
        }
