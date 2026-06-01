

import random
import numpy as np
from typing import List, Dict, Set
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
        """
        Select nodes to attack based on strategy
        
        Args:
            total_nodes: Total number of nodes
            node_data: Optional dict with 'connectivity', 'data_sizes', or 'target_peers'
            strategy: 'random', 'high_connectivity', 'high_data', or 'specific'
        """
        num_attack = max(1, int(total_nodes * self.attack_ratio))
        
        if strategy == "high_connectivity" and node_data and 'connectivity' in node_data:
            # Attack well-connected nodes (hub nodes)
            connectivity = node_data['connectivity']
            sorted_nodes = sorted(range(total_nodes), 
                                key=lambda i: connectivity.get(i, 0), reverse=True)
            target_nodes = sorted_nodes[:num_attack]
            logger.info(f"HIGH-CONNECTIVITY attack: targeting nodes {target_nodes}")
        
        elif strategy == "high_data" and node_data and 'data_sizes' in node_data:
            # Attack nodes with most knowledge
            data_sizes = node_data['data_sizes']
            sorted_nodes = sorted(range(total_nodes), 
                                key=lambda i: data_sizes.get(i, 0), reverse=True)
            target_nodes = sorted_nodes[:num_attack]
            logger.info(f"HIGH-DATA attack: targeting nodes {target_nodes}")
        
        elif strategy == "specific" and node_data and 'target_peers' in node_data:
            # Attack specific peers
            target_nodes = node_data['target_peers'][:num_attack]
            logger.info(f"SPECIFIC attack: targeting nodes {target_nodes}")
        
        else:
            # Random selection (default)
            target_nodes = random.sample(range(total_nodes), min(num_attack, total_nodes))
            logger.info(f"RANDOM attack: targeting nodes {target_nodes}")
        
        self.attacked_nodes.update(target_nodes)
        return target_nodes
    
    def node_removal_attack(self, nodes: List, node_data: Dict, strategy: str = "random") -> Dict:
        
        total_nodes = len(nodes)
        target_nodes = self.select_target_nodes(total_nodes, node_data, strategy)
        
        # Track which nodes are disabled in node_data
        if 'disabled_nodes' not in node_data:
            node_data['disabled_nodes'] = set()
        
        removed_count = 0
        for idx in target_nodes:
            if idx < len(nodes) and nodes[idx] is not None:
                # Mark as disabled in tracking dict
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
    
    def byzantine_attack(self, nodes: List, node_data: Dict, strategy: str = "random") -> Dict:
        
        total_nodes = len(nodes)
        target_nodes = self.select_target_nodes(total_nodes, node_data, strategy)
        
        if 'byzantine' not in node_data:
            node_data['byzantine'] = set()
        
        corrupted_count = 0
        for idx in target_nodes:
            if idx < len(nodes) and nodes[idx] is not None:
                node_data['byzantine'].add(idx)
                corrupted_count += 1
        
        result = {
            'attack_type': 'byzantine',
            'total_nodes': total_nodes,
            'byzantine_count': corrupted_count,
            'byzantine_indices': list(target_nodes),
            'byzantine_ratio': corrupted_count / total_nodes if total_nodes > 0 else 0,
            'targeting_strategy': strategy
        }
        
        self.attack_history.append(result)
        logger.info(f"Byzantine attack ({strategy}): {corrupted_count}/{total_nodes} nodes corrupted")
        
        return result
    
    def network_partition_attack(self, nodes: List, node_data: Dict) -> Dict:
        
        total_nodes = len(nodes)
        partition_point = int(total_nodes * self.attack_ratio)
        partition_point = max(1, min(partition_point, total_nodes - 1))
        
        partition_1 = list(range(partition_point))
        partition_2 = list(range(partition_point, total_nodes))
        
        node_data['partitions'] = [partition_1, partition_2]
        
        result = {
            'attack_type': 'partition',
            'total_nodes': total_nodes,
            'partition_sizes': [len(partition_1), len(partition_2)],
            'partition_1': partition_1,
            'partition_2': partition_2
        }
        
        self.attack_history.append(result)
        logger.info(f"Network partition: {len(partition_1)} vs {len(partition_2)} nodes")
        
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
    
    def sybil_attack(self, nodes: List, node_data: Dict) -> Dict:
        
        total_nodes = len(nodes)
        num_sybil = int(total_nodes * self.attack_ratio)
        num_sybil = max(1, num_sybil)
        
        # Add fake nodes
        sybil_nodes = [f"sybil_{i}" for i in range(num_sybil)]
        
        if 'sybil_nodes' not in node_data:
            node_data['sybil_nodes'] = []
        node_data['sybil_nodes'].extend(sybil_nodes)
        
        result = {
            'attack_type': 'sybil',
            'original_nodes': total_nodes,
            'sybil_count': num_sybil,
            'total_with_sybil': total_nodes + num_sybil,
            'sybil_ratio': num_sybil / (total_nodes + num_sybil) if (total_nodes + num_sybil) > 0 else 0
        }
        
        self.attack_history.append(result)
        logger.info(f"Sybil attack: {num_sybil} fake nodes injected")
        
        return result
    
    def execute_attack(self, nodes: List, node_data: Dict = None, strategy: str = "random") -> Dict:
        """
        Execute the configured attack type
        
        Args:
            nodes: List of active nodes
            node_data: Dictionary containing node information
            strategy: Targeting strategy ('random', 'high_connectivity', 'high_data', 'specific')
            
        Returns:
            Attack results
        """
        if node_data is None:
            node_data = {}
        
        attack_methods = {
            'node_removal': lambda n, nd: self.node_removal_attack(n, nd, strategy),
            'byzantine': lambda n, nd: self.byzantine_attack(n, nd, strategy),
            'partition': self.network_partition_attack,
            'ddos': lambda n, nd: self.ddos_attack(n, nd, strategy),
            'sybil': self.sybil_attack
        }
        
        if self.attack_type not in attack_methods:
            logger.error(f"Unknown attack type: {self.attack_type}")
            return {'error': f"Unknown attack type: {self.attack_type}"}
        
        return attack_methods[self.attack_type](nodes, node_data)
    
    def get_attack_summary(self) -> Dict:
        
        if not self.attack_history:
            return {'message': 'No attacks executed yet'}
        
        summary = {
            'total_attacks': len(self.attack_history),
            'attack_type': self.attack_type,
            'attack_ratio': self.attack_ratio,
            'attacked_nodes': list(self.attacked_nodes),
            'attack_history': self.attack_history
        }
        
        return summary
    
    def reset(self):
        """Reset attack state"""
        self.attacked_nodes.clear()
        self.attack_history.clear()
        logger.info("Attack state reset")


def evaluate_system_availability(nodes: List, node_data: Dict) -> Dict:
    
    total_nodes = len(nodes)
    
    # Count based on disabled_nodes tracking
    disabled_nodes = node_data.get('disabled_nodes', set())
    active_nodes = total_nodes - len(disabled_nodes)
    
    metrics = {
        'total_nodes': total_nodes,
        'active_nodes': active_nodes,
        'unavailable_nodes': len(disabled_nodes),
        'availability_percentage': (active_nodes / total_nodes * 100) if total_nodes > 0 else 0,
        'byzantine_nodes': len(node_data.get('byzantine', set())),
        'ddos_targets': len(node_data.get('ddos_targets', set())),
        'sybil_nodes': len(node_data.get('sybil_nodes', []))
    }
    
    return metrics


def apply_attack_to_network(rag_network, node_attack_results: dict):
    """
    Actually apply the attack effects to the RAG network by setting nodes to None.
    Call this AFTER evaluation phases to physically disable nodes.
    
    Args:
        rag_network: The RAG network to modify
        node_attack_results: Results containing disabled_nodes list
    """
    if not hasattr(rag_network, 'peers'):
        logger.warning("Network has no peers attribute")
        return
    
    # Get disabled nodes from attack results
    disabled_indices = set()
    for iteration_result in node_attack_results.get('iterations', []):
        attack_result = iteration_result.get('attack_result', {})
        disabled_nodes = attack_result.get('disabled_nodes', [])
        disabled_indices.update(disabled_nodes)
    
    # Actually disable the nodes
    for idx in disabled_indices:
        if idx < len(rag_network.peers):
            rag_network.peers[idx] = None
    
    logger.info(f"Applied attack: {len(disabled_indices)} nodes set to None")