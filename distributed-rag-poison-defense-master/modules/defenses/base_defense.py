from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from loguru import logger


class BaseDefense(ABC):
    """
    Base class for all defense mechanisms in DRAG.
    
    All defense mechanisms should inherit from this class and implement
    the validate method.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize defense mechanism.
        
        Args:
            name: Name of the defense mechanism
            config: Configuration dictionary for this defense
        """
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.stats = {
            'total_validations': 0,
            'blocked_answers': 0,
            'passed_answers': 0,
            'avg_confidence': 0.0
        }
        logger.info(f"Initialized defense: {self.name} (enabled={self.enabled})")
    
    @abstractmethod
    def validate(self, question: str, candidate_answer: str, 
                 peer_responses: List[Dict[str, Any]]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Validate if an answer should be accepted.
        
        Args:
            question: The input question
            candidate_answer: The candidate answer to validate
            peer_responses: List of responses from different peers
                           Each dict contains: {'peer_id': int, 'answer': str, 'documents': List}
            
        Returns:
            Tuple of (is_valid, confidence_score, validation_details)
            - is_valid: Boolean indicating if answer passes validation
            - confidence_score: Float between 0 and 1 indicating confidence
            - validation_details: Dictionary with detailed validation information
        """
        pass
    
    def update_stats(self, is_valid: bool, confidence: float):
        """Update defense statistics."""
        self.stats['total_validations'] += 1
        if is_valid:
            self.stats['passed_answers'] += 1
        else:
            self.stats['blocked_answers'] += 1
        
        # Update running average of confidence
        n = self.stats['total_validations']
        current_avg = self.stats['avg_confidence']
        self.stats['avg_confidence'] = (current_avg * (n - 1) + confidence) / n
    
    def get_stats(self) -> Dict[str, Any]:
        """Get defense statistics."""
        total = self.stats['total_validations']
        return {
            'name': self.name,
            'enabled': self.enabled,
            'total_validations': total,
            'blocked_answers': self.stats['blocked_answers'],
            'passed_answers': self.stats['passed_answers'],
            'block_rate': self.stats['blocked_answers'] / max(1, total),
            'avg_confidence': self.stats['avg_confidence']
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_validations': 0,
            'blocked_answers': 0,
            'passed_answers': 0,
            'avg_confidence': 0.0
        }