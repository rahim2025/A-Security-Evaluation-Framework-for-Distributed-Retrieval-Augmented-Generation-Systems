from typing import Dict, Any, List, Tuple
from collections import Counter
from loguru import logger
from modules.defenses.base_defense import BaseDefense
from difflib import SequenceMatcher


class CrossPeerValidation(BaseDefense):
    """
    Cross-peer validation using majority voting.
    
    This defense mechanism validates answers by comparing responses from multiple
    peers. An answer is considered valid if a sufficient number of peers agree
    on the same or similar answer.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("CrossPeerValidation", config)
        self.min_agreement_ratio = config.get('min_agreement_ratio', 0.6)
        self.voting_method = config.get('voting_method', 'majority')
        self.min_peers = config.get('min_peers_for_validation', 3)
        self.use_similarity = config.get('use_similarity_matching', True)
        self.similarity_threshold = config.get('similarity_threshold', 0.85)
        
        logger.info(f"CrossPeerValidation initialized with:")
        logger.info(f"  - Min agreement ratio: {self.min_agreement_ratio}")
        logger.info(f"  - Voting method: {self.voting_method}")
        logger.info(f"  - Min peers: {self.min_peers}")
        logger.info(f"  - Use similarity matching: {self.use_similarity}")
    
    def validate(self, question: str, candidate_answer: str, 
                 peer_responses: List[Dict[str, Any]]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Validate answer using cross-peer voting.
        
        Args:
            question: The input question
            candidate_answer: The candidate answer to validate
            peer_responses: List of peer responses with answers
            
        Returns:
            Tuple of (is_valid, confidence, details)
        """
        if not self.enabled:
            return True, 1.0, {"reason": "Defense disabled"}
        
        # Extract answers from peer responses
        peer_answers = [resp.get('answer', '') for resp in peer_responses]
        
        if len(peer_answers) < self.min_peers:
            logger.debug(f"Insufficient peers for validation ({len(peer_answers)} < {self.min_peers})")
            # Not enough peers for validation, accept by default
            return True, 0.5, {
                "reason": f"Insufficient peers ({len(peer_answers)} < {self.min_peers})",
                "num_peers": len(peer_answers),
                "validation_skipped": True
            }
        
        # Perform voting
        is_valid, confidence, details = self._perform_voting(candidate_answer, peer_answers)
        
        # Update statistics
        self.update_stats(is_valid, confidence)
        
        # Add additional context to details
        details['num_peers'] = len(peer_answers)
        details['min_agreement_ratio'] = self.min_agreement_ratio
        
        if logger.level("DEBUG").no <= logger._core.min_level:
            logger.debug(f"Validation result: valid={is_valid}, confidence={confidence:.2f}")
            logger.debug(f"Details: {details}")
        
        return is_valid, confidence, details
    
    def _perform_voting(self, candidate_answer: str, peer_answers: List[str]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Perform voting among peer answers.
        
        Args:
            candidate_answer: The answer to validate
            peer_answers: List of answers from different peers
            
        Returns:
            Tuple of (is_valid, confidence, details)
        """
        if self.voting_method == "majority":
            return self._majority_voting(candidate_answer, peer_answers)
        elif self.voting_method == "weighted":
            return self._weighted_voting(candidate_answer, peer_answers)
        else:
            logger.warning(f"Unknown voting method: {self.voting_method}, using majority")
            return self._majority_voting(candidate_answer, peer_answers)
    
    def _majority_voting(self, candidate_answer: str, peer_answers: List[str]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Simple majority voting with optional fuzzy matching.
        
        Args:
            candidate_answer: The answer to validate
            peer_answers: List of answers from peers
            
        Returns:
            Tuple of (is_valid, confidence, details)
        """
        if not peer_answers:
            return False, 0.0, {"reason": "No peer answers available"}
        
        # Group similar answers if similarity matching is enabled
        if self.use_similarity:
            answer_groups = self._group_similar_answers(peer_answers)
        else:
            # Exact matching only
            answer_groups = {ans: [ans] for ans in set(peer_answers)}
        
        # Count occurrences in each group
        group_counts = {}
        for group_key, group_members in answer_groups.items():
            group_counts[group_key] = len(group_members)
        
        # Find the most common answer group
        if not group_counts:
            return False, 0.0, {"reason": "No valid answer groups"}
        
        most_common_answer = max(group_counts, key=group_counts.get)
        most_common_count = group_counts[most_common_answer]
        
        # Calculate agreement ratio
        agreement_ratio = most_common_count / len(peer_answers)
        
        # Check if candidate answer belongs to the majority group
        candidate_in_majority = self._answer_matches_group(
            candidate_answer, most_common_answer, answer_groups
        )

        # Check if there's an actual majority (more than 50% OR clear winner with sufficient support)
        has_clear_majority = (
            agreement_ratio > 0.5 or  # Absolute majority
            (most_common_count >= 2 and agreement_ratio >= self.min_agreement_ratio)  # At least 2 peers agree
        )

        # If no clear majority, default to accepting candidate answer
        if not has_clear_majority:
            logger.debug(f"No clear majority found (max agreement: {agreement_ratio:.2f}), accepting candidate answer")
            return True, 0.5, {
                "reason": "No clear majority, accepting candidate",
                "agreement_ratio": agreement_ratio,
                "answer_distribution": dict(group_counts),
                "candidate_accepted_by_default": True
            }
        
        # Validation logic
        if candidate_in_majority and agreement_ratio >= self.min_agreement_ratio:
            return True, agreement_ratio, {
                "reason": "Majority agreement",
                "agreement_ratio": agreement_ratio,
                "majority_answer": most_common_answer,
                "supporting_peers": most_common_count,
                "answer_distribution": dict(group_counts)
            }
        elif not candidate_in_majority:
            return False, 1.0 - agreement_ratio, {
                "reason": "Answer differs from majority",
                "agreement_ratio": agreement_ratio,
                "majority_answer": most_common_answer,
                "candidate_answer": candidate_answer,
                "answer_distribution": dict(group_counts)
            }
        else:
            return False, agreement_ratio, {
                "reason": f"Low agreement ratio ({agreement_ratio:.2f} < {self.min_agreement_ratio})",
                "agreement_ratio": agreement_ratio,
                "majority_answer": most_common_answer,
                "answer_distribution": dict(group_counts)
            }
    
    def _weighted_voting(self, candidate_answer: str, peer_answers: List[str]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Weighted voting (can be extended to include peer reputation).
        Currently falls back to majority voting.
        """
        # For now, use majority voting
        # In future, this can incorporate peer reputation scores
        return self._majority_voting(candidate_answer, peer_answers)
    
    def _group_similar_answers(self, answers: List[str]) -> Dict[str, List[str]]:
        """
        Group similar answers together using fuzzy string matching.
        
        Args:
            answers: List of answer strings
            
        Returns:
            Dictionary mapping representative answer to list of similar answers
        """
        groups = {}
        
        for answer in answers:
            # Try to find an existing group this answer belongs to
            found_group = False
            for group_key in groups.keys():
                if self._are_answers_similar(answer, group_key):
                    groups[group_key].append(answer)
                    found_group = True
                    break
            
            # If no similar group found, create a new group
            if not found_group:
                groups[answer] = [answer]
        
        return groups
    
    def _are_answers_similar(self, answer1: str, answer2: str) -> bool:
        """
        Check if two answers are similar using SequenceMatcher.
        
        Args:
            answer1: First answer string
            answer2: Second answer string
            
        Returns:
            True if answers are similar, False otherwise
        """
        # Normalize answers
        a1 = answer1.lower().strip()
        a2 = answer2.lower().strip()
        
        # Exact match
        if a1 == a2:
            return True
        
        # Calculate similarity ratio
        similarity = SequenceMatcher(None, a1, a2).ratio()
        
        return similarity >= self.similarity_threshold
    
    def _answer_matches_group(self, candidate_answer: str, group_key: str, 
                             answer_groups: Dict[str, List[str]]) -> bool:
        """
        Check if candidate answer matches any answer in the group.
        
        Args:
            candidate_answer: The answer to check
            group_key: The representative answer of the group
            answer_groups: Dictionary of answer groups
            
        Returns:
            True if candidate matches the group, False otherwise
        """
        if self.use_similarity:
            return self._are_answers_similar(candidate_answer, group_key)
        else:
            # Exact match
            return candidate_answer.lower().strip() == group_key.lower().strip()