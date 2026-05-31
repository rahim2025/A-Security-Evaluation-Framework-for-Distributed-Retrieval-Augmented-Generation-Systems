from abc import ABC, abstractmethod
from typing import Dict, Any, List
from modules.rag_network import DRAGNetwork
from modules.data_types import Datapoint


class BaseAttack(ABC):
    """Base class for all attacks on distributed RAG systems."""
    
    def __init__(self, attack_name: str):
        self.attack_name = attack_name
        self.attack_results = {}
    
    @abstractmethod
    def execute(self, network: DRAGNetwork, data_points: List[Datapoint]) -> Dict[str, Any]:
        """
        Execute the attack on the network.
        
        Args:
            network: The DRAG network to attack
            data_points: Original data points (for reference)
            
        Returns:
            Dictionary containing attack metrics and results
        """
        pass
    
    @abstractmethod
    def evaluate_success(self, original_metrics: Dict[str, float], 
                        attacked_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate the success of the attack.
        
        Args:
            original_metrics: Metrics from clean network
            attacked_metrics: Metrics from attacked network
            
        Returns:
            Dictionary containing success metrics
        """
        pass