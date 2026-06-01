import random
from typing import Dict, Any, List
from loguru import logger
from modules.attacks.base_attack import BaseAttack
from modules.rag_network import DRAGNetwork
from modules.data_types import Datapoint


class DataPoisoningAttack(BaseAttack):
    """
    Data Poisoning Attack on Distributed RAG Systems.
    
    This attack injects malicious/incorrect data points into randomly selected peers
    to degrade the quality of answers.
    """
    
    def __init__(
        self,
        poisoning_ratio: float = 0.1,
        attack_strategy: str = "random",
        poison_type: str = "wrong_answer",
        target_peer_ids: List[int] = None, # Add this parameter
        amplification_factor: int = 3,  # NEW
        question_variants: int = 2      # NEW
    ):
        """
        Initialize the data poisoning attack.
        
        Args:
            poisoning_ratio: Ratio of peers to compromise (0.0 to 1.0)
            attack_strategy: Strategy for selecting peers to poison
                - "random": Random peer selection
                - "high_degree": Target high-degree nodes (hubs)
                - "topic_specific": Target peers with specific topics
            poison_type: Type of poisoning to apply
                - "wrong_answer": Replace answers with incorrect ones
                - "misleading": Replace with plausible but wrong answers
                - "noise": Add random noise to answers
            target_peer_ids: List of specific peer IDs to target (used with "targeted" strategy)
        """
        super().__init__("DataPoisoning")
        self.poisoning_ratio = poisoning_ratio
        self.attack_strategy = attack_strategy
        self.poison_type = poison_type
        self.target_peer_ids = target_peer_ids or []
        self.amplification_factor = amplification_factor
        self.question_variants = question_variants
        self.poisoned_peer_ids = []
        self.poisoned_datapoints = []
    
    def execute(self, network: DRAGNetwork, data_points: List[Datapoint]) -> Dict[str, Any]:
        """
        Execute data poisoning attack on the network.
        
        Args:
            network: The DRAG network to attack
            data_points: Original clean data points
            
        Returns:
            Dictionary with attack execution details
        """
        logger.info(f"Executing aggresive {self.attack_name} attack...")
        logger.info(f"Amplification: {self.amplification_factor}x, Variants: {self.question_variants}")
        logger.info(f"Poisoning ratio: {self.poisoning_ratio}")
        logger.info(f"Attack strategy: {self.attack_strategy}")
        logger.info(f"Poison type: {self.poison_type}")
        
        # Select peers to poison based on strategy
        self.poisoned_peer_ids = self._select_peers_to_poison(network)
        
        logger.info(f"Selected {len(self.poisoned_peer_ids)} peers to poison: {self.poisoned_peer_ids}")
        
        # Generate poisoned data points
        num_poison_per_peer = max(10, len(data_points) // (2 * len(self.poisoned_peer_ids)))
        logger.info(f"Injecting {num_poison_per_peer} poisoned data points per malicious peer")
        
        total_poisoned = 0
        for peer_id in self.poisoned_peer_ids:
            # Sample data points to poison
            samples = random.sample(data_points, min(num_poison_per_peer, len(data_points)))
            
            for original_dp in samples:
                # Strategy 1: Create question variants (if enabled)
                if self.question_variants > 1:
                    variants = self._create_question_variants(
                        original_dp, 
                        self.question_variants, 
                        data_points
                    )
                    for variant_dp in variants:
                        # Strategy 2: Amplify each variant
                        for _ in range(self.amplification_factor):
                            network.peers[peer_id].add_knowledge(variant_dp)
                            self.poisoned_datapoints.append(variant_dp)
                            total_poisoned += 1
                else:
                    # Single poisoned datapoint with amplification
                    if self.poison_type == "answer_swap":
                        poisoned_dp = self._create_poisoned_with_confusing_answer(
                            original_dp, 
                            data_points
                        )
                    else:
                        poisoned_dp = self._create_poisoned_datapoint(original_dp)
                    
                    # Amplify: Add multiple copies
                    for _ in range(self.amplification_factor):
                        network.peers[peer_id].add_knowledge(poisoned_dp)
                        self.poisoned_datapoints.append(poisoned_dp)
                        total_poisoned += 1
        
        logger.info(f"Successfully poisoned {total_poisoned} data points across {len(self.poisoned_peer_ids)} peers")
        
        return {
            "num_poisoned_peers": len(self.poisoned_peer_ids),
            "poisoned_peer_ids": self.poisoned_peer_ids,
            "total_poisoned_datapoints": total_poisoned,
            "poisoning_ratio": self.poisoning_ratio,
            "attack_strategy": self.attack_strategy,
            "poison_type": self.poison_type,
            "amplification_factor": self.amplification_factor,  # NEW
            "question_variants": self.question_variants          # NEW
        }
    
    def _create_question_variants(self, original_dp: Datapoint, num_variants: int, 
                                data_points: List[Datapoint]) -> List[Datapoint]:
        """Create semantically similar questions with wrong answers."""
        variants = []
        for i in range(num_variants):
            perturbed_q = self._perturb_question(original_dp.question, i)
            wrong_answer = self._get_confusing_answer(original_dp, data_points)
            
            variants.append(Datapoint(
                topic=original_dp.topic,
                question=perturbed_q,
                answer=wrong_answer
            ))
        return variants
    
    def _perturb_question(self, question: str, variant_id: int) -> str:
        """Perturb question slightly."""
        perturbations = [
            question,
            f"{question} ?",
            question.replace("?", ""),
            question.replace("What", "what").replace("Which", "which"),
        ]
        return perturbations[variant_id % len(perturbations)]

    def _get_confusing_answer(self, original_dp: Datapoint, 
                            data_points: List[Datapoint]) -> str:
        """Get answer from same topic for maximum confusion."""
        same_topic = [dp.answer for dp in data_points 
                    if dp.topic == original_dp.topic and dp.answer != original_dp.answer]
        
        if same_topic:
            return random.choice(same_topic)
        
        all_wrong = [dp.answer for dp in data_points if dp.answer != original_dp.answer]
        return random.choice(all_wrong)

    def _create_poisoned_with_confusing_answer(self, original_dp: Datapoint,
                                            data_points: List[Datapoint]) -> Datapoint:
        """Create poisoned datapoint with confusing answer."""
        return Datapoint(
            topic=original_dp.topic,
            question=original_dp.question,
            answer=self._get_confusing_answer(original_dp, data_points)
        )

    def _select_peers_to_poison(self, network: DRAGNetwork) -> List[int]:
        """Select peers to poison based on attack strategy."""
        num_malicious = max(1, int(network.num_peers * self.poisoning_ratio))
        
        if self.attack_strategy == "random":
            return random.sample(range(network.num_peers), num_malicious)
        
        elif self.attack_strategy == "high_degree":
            # Target high-degree nodes (network hubs)
            degree_dict = dict(network.network.degree())
            sorted_peers = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
            return [peer_id for peer_id, _ in sorted_peers[:num_malicious]]
        
        elif self.attack_strategy == "topic_specific":
            # Target peers with the most topics
            peer_topic_counts = [(peer_id, len(topics)) 
                                for peer_id, topics in network.peer_topics.items()]
            sorted_peers = sorted(peer_topic_counts, key=lambda x: x[1], reverse=True)
            return [peer_id for peer_id, _ in sorted_peers[:num_malicious]]
        
        elif self.attack_strategy == "targeted":
            # Use specified target peers
            target_peers = self.target_peer_ids.copy()
            
            # If we need more peers, add high-degree ones
            if num_malicious > len(target_peers):
                degree_dict = dict(network.network.degree())
                sorted_peers = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
                for peer_id, _ in sorted_peers:
                    if peer_id not in target_peers:
                        target_peers.append(peer_id)
                        if len(target_peers) >= num_malicious:
                            break
            
            logger.info(f"Targeted attack on specific peers: {target_peers[:num_malicious]}")
            return target_peers[:num_malicious]
        
        elif self.attack_strategy == "data_rich":
            # NEW: Target peers with most data points (highest query probability)
            peer_data_counts = []
            for peer_id in range(network.num_peers):
                num_datapoints = len(network.peers[peer_id].knowledge_base.data_points)
                peer_data_counts.append((peer_id, num_datapoints))
            
            # Sort by data count (descending)
            sorted_peers = sorted(peer_data_counts, key=lambda x: x[1], reverse=True)
            target_peers = [peer_id for peer_id, _ in sorted_peers[:num_malicious]]
            
            logger.info(f"Targeting peers with most data: {target_peers}")
            return target_peers
        
        elif self.attack_strategy == "replica_aware":
            # Intelligently poison multiple replicas
            return self._select_peers_replica_aware_poisoning(network)

        else:
            logger.warning(f"Unknown strategy {self.attack_strategy}, using random")
            return random.sample(range(network.num_peers), num_malicious)
        

    def _select_peers_replica_aware_poisoning(self, network: DRAGNetwork) -> List[int]:
        """
        Replica-aware poisoning: Target ALL replicas of high-value topics.
        This ensures complete compromise of selected topics.
        """
        num_peers_to_poison = max(1, int(network.num_peers * self.poisoning_ratio))
        logger.info(f"Replica-aware poisoning with budget: {num_peers_to_poison} peers")
        
        # Strategy: Select topics and poison ALL their replicas
        peers_to_poison = set()
        fully_compromised_topics = []
        
        # Count data points per topic to prioritize high-value targets
        topic_data_counts = {}
        for topic in network.all_topics:
            topic_data_counts[topic] = sum(
                1 for peer_id in network.topic_peers[topic]
                for dp in network.peers[peer_id].knowledge_base.data_points
                if dp.topic == topic
            )
        
        # Sort topics by number of data points (descending)
        sorted_topics = sorted(topic_data_counts.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"Available topics: {len(sorted_topics)}")
        logger.info(f"Top 5 topics by data count: {sorted_topics[:5]}")
        
        # Greedily select topics and poison ALL their replicas
        for topic, data_count in sorted_topics:
            topic_peers = set(network.topic_peers[topic])
            
            # Check if we can afford to poison all replicas of this topic
            if len(peers_to_poison) + len(topic_peers) <= num_peers_to_poison:
                peers_to_poison.update(topic_peers)
                fully_compromised_topics.append(topic)
                logger.info(f"✓ Fully compromised topic '{topic}': {len(topic_peers)} replicas, {data_count} datapoints")
            else:
                # Partial poisoning: poison as many replicas as budget allows
                remaining_budget = num_peers_to_poison - len(peers_to_poison)
                if remaining_budget > 0:
                    available_peers = topic_peers - peers_to_poison
                    partial_peers = list(available_peers)[:remaining_budget]
                    peers_to_poison.update(partial_peers)
                    logger.info(f"⚠ Partially compromised topic '{topic}': {len(partial_peers)}/{len(topic_peers)} replicas")
                break
        
        logger.info(f"Replica-aware poisoning results:")
        logger.info(f"  Peers poisoned: {len(peers_to_poison)}/{num_peers_to_poison}")
        logger.info(f"  Topics fully compromised: {len(fully_compromised_topics)}")
        logger.info(f"  Compromised topics: {fully_compromised_topics}")
        
        return list(peers_to_poison)
    
    def _create_poisoned_datapoint(self, original_dp: Datapoint) -> Datapoint:
        """Create a poisoned version of a data point."""
        if self.poison_type == "wrong_answer":
            # Simply prefix the answer with "POISONED"
            poisoned_answer = f"POISONED: This is incorrect information."
            
        elif self.poison_type == "misleading":
            # Create plausible but wrong answer
            poisoned_answer = f"The correct answer is the opposite of {original_dp.answer}"
            
        elif self.poison_type == "noise":
            # Add random characters to the answer
            noise = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))
            poisoned_answer = f"{original_dp.answer} {noise}"
            
        else:
            poisoned_answer = "POISONED: Invalid response"
        
        return Datapoint(
            topic=original_dp.topic,
            question=original_dp.question,
            answer=poisoned_answer
        )
    
    def evaluate_success(self, original_metrics: Dict[str, float], 
                        attacked_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate attack success by comparing metrics.
        
        Args:
            original_metrics: Metrics from clean network
            attacked_metrics: Metrics from attacked network
            
        Returns:
            Dictionary with success evaluation
        """
        # Calculate performance degradation
        degradation = {}
        for metric in ['f1', 'exact_match', 'bleu', 'semantic_similarity']:
            if metric in original_metrics and metric in attacked_metrics:
                original_val = original_metrics[metric]
                attacked_val = attacked_metrics[metric]
                
                if original_val > 0:
                    degradation[f"{metric}_degradation"] = \
                        ((original_val - attacked_val) / original_val) * 100
                else:
                    degradation[f"{metric}_degradation"] = 0.0
        
        # Calculate overall attack success rate
        avg_degradation = sum(degradation.values()) / len(degradation) if degradation else 0.0
        
        return {
            "attack_success_rate": avg_degradation,
            "metric_degradations": degradation,
            "original_metrics": original_metrics,
            "attacked_metrics": attacked_metrics,
            "is_successful": avg_degradation > 10.0  # Consider successful if >10% degradation
        }