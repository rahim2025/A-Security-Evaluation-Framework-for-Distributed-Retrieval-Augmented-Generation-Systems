from collections import deque
import random
from typing import Dict, List, Optional
from typing import Optional
import yaml
from pathlib import Path

from loguru import logger
import networkx as nx
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from modules.data_types import RAGAnswer, Datapoint
from modules.peer import Peer


class DRAGNetwork:
    def __init__(
            self, 
            num_peers: int, 
            num_peer_attachments: int, 
            llm_url: str, 
            llm_name: str, 
            llm_num_ctx: int,
            llm_seed: int, 
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
        ):
        self.num_peers = num_peers
        self.network = nx.barabasi_albert_graph(num_peers, num_peer_attachments)
        self.text_embedding_model = SentenceTransformer(embedding_model)
        self.peers = [Peer(peer_id, llm_url, llm_name, llm_num_ctx, llm_seed, self.text_embedding_model) 
                      for peer_id in range(num_peers)]

        self.peer_topics: Dict[int, List[str]] = {peer_id: [] for peer_id in range(self.num_peers)}
        self.topic_peers: Dict[str, List[int]] = {}
        self.all_topics: List[str] = []

        # Initialize defense mechanism
        self.defense_enabled = False
        self.defense_mechanism = None
        self._initialize_defense()

    # NEW
    def _initialize_defense(self):
        """Initialize defense mechanism if configured."""
        try:
            from modules.defenses import CrossPeerValidation
            
            defense_config_path = Path("config/defense.yaml")
            
            if not defense_config_path.exists():
                logger.info("No defense configuration file found, defenses disabled")
                self.defense_enabled = False
                self.defense_mechanism = None
                self.max_additional_hops = 0
                return
            
            # Load defense configuration
            with open(defense_config_path, 'r') as f:
                defense_config = yaml.safe_load(f)
            
            # Check if defense is enabled in config
            if defense_config and defense_config.get('defense', {}).get('enabled', False):
                self.defense_enabled = True
                cpv_config = defense_config['defense'].get('cross_peer_validation', {})
                self.defense_mechanism = CrossPeerValidation(cpv_config)
                self.max_additional_hops = cpv_config.get('max_additional_hops', 0)

                # Show defense banner - THIS IS THE NEW PART
                logger.info("=" * 60)
                logger.info("DEFENSE MECHANISM ENABLED")
                logger.info("=" * 60)
                logger.info(f"Defense Type: Cross-Peer Validation")
                logger.info(f"Min Agreement Ratio: {cpv_config.get('min_agreement_ratio', 0.6)}")
                logger.info(f"Min Peers Required: {cpv_config.get('min_peers_for_validation', 3)}")
                logger.info(f"Max Additional Hops: {self.max_additional_hops}")
                logger.info(f"Voting Method: {cpv_config.get('voting_method', 'majority')}")
                logger.info(f"Similarity Matching: {cpv_config.get('use_similarity_matching', True)}")
                logger.info(f"Similarity Threshold: {cpv_config.get('similarity_threshold', 0.85)}")
                logger.info("=" * 60)
            else:
                self.defense_enabled = False
                self.defense_mechanism = None
                self.max_additional_hops = 0
                logger.info("Defense mechanism disabled in configuration")
                
        except Exception as e:
            logger.error(f"Failed to initialize defense mechanism: {e}")
            logger.exception(e)
            self.defense_enabled = False
            self.defense_mechanism = None
            self.max_additional_hops = 0


    # NEW
    def _validate_with_defense(self, question: str, candidate_answer: str, 
                        candidate_peer_id: int, peer_answers_cache: Dict[int, str],
                        query_confidence_threshold: float = 0.5) -> tuple:
        """
        Validate answer using defense mechanism if enabled.
        
        Args:
            question: The question being asked
            candidate_answer: The answer to validate
            candidate_peer_id: ID of peer that provided the answer
            peer_answers_cache: Dictionary mapping peer_id -> answer (from routing)
            query_confidence_threshold: Confidence threshold for queries
            
        Returns:
            Tuple of (final_answer, defense_info)
        """
        if not self.defense_enabled or not self.defense_mechanism:
            return candidate_answer, {'defense_enabled': False}
        
        # Build peer_responses from cached answers (no re-querying!)
        peer_responses = []
        for peer_id, answer in peer_answers_cache.items():
            peer_responses.append({
                'peer_id': peer_id,
                'answer': answer,
                'documents': []
            })
        
        # Filter out empty answers for validation
        valid_responses = [r for r in peer_responses if r['answer']]
        
        logger.debug(f"Defense validation: {len(peer_responses)} peers queried, {len(valid_responses)} have answers")
        
        # If we don't have enough peers with actual answers, issue warning
        if len(valid_responses) < self.defense_mechanism.min_peers:
            logger.warning(
                f"INSUFFICIENT PEERS FOR VALIDATION: Only {len(valid_responses)}/{self.defense_mechanism.min_peers} "
                f"peers have answers. Accepting candidate answer with LIMITED CONFIDENCE."
            )
            return candidate_answer, {
                'defense_enabled': True,
                'validation_skipped': True,
                'reason': f'Insufficient peers with answers ({len(valid_responses)} < {self.defense_mechanism.min_peers})',
                'num_peers_visited': len(peer_responses),
                'num_peers_with_answers': len(valid_responses),
                'warning': 'VALIDATION_INSUFFICIENT_PEERS'
            }
        
        # Validate using defense mechanism
        is_valid, confidence, details = self.defense_mechanism.validate(
            question, candidate_answer, valid_responses
        )
        
        if not is_valid:
            logger.warning(f"Defense blocked answer from peer {candidate_peer_id}: {details.get('reason', 'Unknown')}")
            # Use majority answer if available
            if 'majority_answer' in details:
                final_answer = details['majority_answer']
                logger.info(f"Using majority answer instead")
            else:
                final_answer = candidate_answer
                logger.warning("No majority answer available, using candidate anyway")
        else:
            final_answer = candidate_answer
        
        # Prepare defense info
        defense_info = {
            'defense_enabled': True,
            'is_valid': is_valid,
            'confidence': confidence,
            'details': details,
            'num_peers_validated': len(valid_responses),
            'num_peers_visited': len(peer_responses)
        }
        
        return final_answer, defense_info


    # Modified with replication_factor
    def init_knowledge(self, data_points: List[Datapoint], replication_factor: int = 1):
        """
        Distributes data points to peers based on their topics with replication.
        
        Args:
            data_points: A list of Datapoint objects.
            replication_factor: Number of peers that should have each topic (default: 1)
        """
        # Count all topics
        topic_check: Dict[str, bool] = {}
        for data_point in data_points:
            topic_check[data_point.topic] = True
        self.all_topics = list(topic_check.keys())

        # Assign topics to MULTIPLE peers for redundancy
        self.topic_peers = {topic: [] for topic in self.all_topics}
        for topic in self.all_topics:
            # Select multiple random peers for each topic
            selected_peers = random.sample(range(self.num_peers), replication_factor)
            
            for peer_id in selected_peers:
                self.topic_peers[topic].append(peer_id)
                if topic not in self.peer_topics[peer_id]:
                    self.peer_topics[peer_id].append(topic)

        # Distribute data points to peers based on assigned topics
        for data_point in tqdm(data_points, desc=f"Distributing data points to peers"):
            for peer_id in self.topic_peers[data_point.topic]:
                self.peers[peer_id].add_knowledge(data_point)
        
        # Log distribution statistics
        topics_per_peer = [len(topics) for topics in self.peer_topics.values()]
        logger.info(f"Data distribution complete:")
        logger.info(f"  Topics: {len(self.all_topics)}")
        logger.info(f"  Replication factor: {replication_factor}")
        logger.info(f"  Avg topics per peer: {sum(topics_per_peer)/len(topics_per_peer):.2f}")
        logger.info(f"  Min/Max topics per peer: {min(topics_per_peer)}/{max(topics_per_peer)}")


    def topic_query(
        self, 
        question: str, 
        query_peer_id: Optional[int] = None, 
        num_query_neighbor: int = 2,
        query_confidence_threshold: float = 0.5,
        max_ttl: int = 6,
        skip_query_peer: bool = False  # NEW - Skip querying the initial peer
    ) -> RAGAnswer:
        """
        Topic-based network query with extended routing for defense validation.

        Args:
            question: The question to ask.
            query_peer_id: The ID of the peer initiating the query. If None, a random peer is selected.
            num_query_neighbor: The maximum number of neighbors to query at each hop.
            query_confidence_threshold: The confidence threshold required for an answer to be accepted.
            max_ttl: The time-to-live for the query (maximum number of hops).

        Returns:
            The answer to the question if found, otherwise None.
        """
        # Track number of messages (queries) sent
        num_messages = 0

        # Initialize query peer
        if query_peer_id is None:
            query_peer_id = random.choice(range(self.num_peers))
            logger.debug(f"Randomly selected starting peer: {query_peer_id}")
        
        logger.debug(f"Starting topic-based search from peer {query_peer_id}")
        logger.debug(f"Parameters: num_neighbors={num_query_neighbor}, max_ttl={max_ttl}")

        # Determine the topic of the question first
        question_topic = self.peers[query_peer_id].parse_topic(question, self.all_topics)
        logger.debug(f"Parsed question topic: {question_topic}")

        # Keep track of visited peers and their answers
        visited_ids = {query_peer_id}
        peer_answers_cache = {}  # Maps peer_id -> answer_text

        # Queue for BFS: (peer_id, hop)
        queue = deque([(query_peer_id, 0)])
        
        # Track first answer found
        candidate_answer = None
        candidate_peer_id = None
        candidate_data = None  # (relevant_knowledge, relevant_score, is_query_hit)
        first_answer_hop = None

        # Perform topic-based search
        while queue:
            current_peer_id, hop = queue.popleft()

            # Check termination conditions
            if candidate_answer is not None and self.defense_enabled:
                # Extended routing for defense: continue until we have enough peers OR exceeded additional hops
                peers_with_answers = sum(1 for ans in peer_answers_cache.values() if ans)
                additional_hops_used = hop - first_answer_hop
                
                if peers_with_answers >= self.defense_mechanism.min_peers:
                    logger.debug(f"Sufficient peers collected: {peers_with_answers}/{self.defense_mechanism.min_peers}")
                    break
                elif additional_hops_used >= self.max_additional_hops:
                    logger.debug(f"Max additional hops reached: {additional_hops_used}/{self.max_additional_hops}")
                    break
            elif candidate_answer is not None:
                # Defense disabled: stop immediately after finding answer
                break

            if hop >= max_ttl:
                continue

            logger.debug(f"Hop {hop}: Querying peer {current_peer_id}")

            # Skip querying the initial peer if requested (for extraction attacks)
            if hop == 0 and current_peer_id == query_peer_id and skip_query_peer:
                logger.debug(f"  Skipping query at starting peer {query_peer_id} (skip_query_peer=True)")
                # Don't query, just continue to add neighbors to queue
                current_answer = None
                relevant_knowledge = None
                relevant_score = 0.0
                is_query_hit = False
            else:
                # Query the current peer and track message
                current_answer, relevant_knowledge, relevant_score, is_query_hit = \
                    self.peers[current_peer_id].query(question, query_confidence_threshold)
                num_messages += 1

                # Cache the answer (even if None)
                answer_text = str(current_answer) if current_answer is not None else ""
                peer_answers_cache[current_peer_id] = answer_text

                # Store first answer found
                if current_answer is not None and candidate_answer is None:
                    candidate_answer = current_answer
                    candidate_peer_id = current_peer_id
                    candidate_data = (relevant_knowledge, relevant_score, is_query_hit)
                    first_answer_hop = hop
                    logger.debug(f"First answer found at peer {current_peer_id}, hop {hop}")
                    
                    if self.defense_enabled:
                        logger.debug(f"Defense enabled: continuing to collect peers for validation")

            # Get and prioritize neighbors
            current_neighbor_ids = list(self.network.neighbors(current_peer_id))
            picked_neighbor_ids = []

            if len(current_neighbor_ids) > num_query_neighbor:
                # Find topic-matched neighbors
                topic_matched_neighbors = []
                for neighbor_id in current_neighbor_ids:
                    if question_topic in self.peer_topics[neighbor_id]:
                        topic_matched_neighbors.append(neighbor_id)

                # Add topic-matched peers to the neighbor list
                logger.debug(f"Found {len(topic_matched_neighbors)} topic-matched neighbors, update neighbor list")
                for neighbor_id in topic_matched_neighbors:
                    self.network.add_edge(query_peer_id, neighbor_id)

                # Select neighbors based on topic matching
                if len(topic_matched_neighbors) > num_query_neighbor:
                    picked_neighbor_ids = random.sample(topic_matched_neighbors, num_query_neighbor)
                else:
                    picked_neighbor_ids = topic_matched_neighbors
                    # Fill remaining slots with random neighbors
                    remaining_neighbor_ids = list(set(current_neighbor_ids) - set(picked_neighbor_ids))
                    remaining_num = min(num_query_neighbor - len(picked_neighbor_ids), len(remaining_neighbor_ids))
                    picked_neighbor_ids += random.sample(remaining_neighbor_ids, remaining_num)
            else:
                picked_neighbor_ids = current_neighbor_ids

            logger.debug(f"Selected neighbors for next hop: {picked_neighbor_ids}")

            # Add picked neighbors to the queue
            for neighbor_id in picked_neighbor_ids:
                if neighbor_id not in visited_ids:
                    visited_ids.add(neighbor_id)
                    queue.append((neighbor_id, hop + 1))
                    logger.debug(f"Added neighbor {neighbor_id} to queue at Hop {hop + 1}")

        # Process results
        if candidate_answer is not None:
            logger.debug(f"Answer found at peer {candidate_peer_id} after {first_answer_hop} hops")
            logger.debug(f"Total messages sent: {num_messages}")
            
            # Apply defense validation using cached answers
            final_answer, defense_info = self._validate_with_defense(
                question, str(candidate_answer), candidate_peer_id, peer_answers_cache,
                query_confidence_threshold
            )
            
            return RAGAnswer(
                answer=final_answer,
                relevant_knowledge=candidate_data[0],
                relevant_score=candidate_data[1],
                num_hops=first_answer_hop,
                num_messages=num_messages,
                is_query_hit=candidate_data[2],
                defense_info=defense_info
            )

        logger.debug(f"Search failed after {max_ttl} hops")
        logger.debug(f"Total messages sent: {num_messages}")

        # Return empty answer if no result found
        return RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=max_ttl,
            num_messages=num_messages,
            is_query_hit=False,
            defense_info={'defense_enabled': False, 'reason': 'No answer found'}
        )


    def random_walk_query(
            self,
            question: str,
            query_peer_id: Optional[int] = None,
            query_confidence_threshold: float = 0.5,
            max_ttl: int = 6,
            restart_probability: float = 0.1
    ) -> RAGAnswer:
        """
        Queries the network using a random walk algorithm with restart probability.
        Extended routing for defense validation when enabled.
        
        Args:
            question: The question to ask
            query_peer_id: Starting peer ID (random if None)
            query_confidence_threshold: Minimum confidence required for an answer
            max_ttl: Maximum number of steps in the walk
            restart_probability: Probability of restarting from the initial peer
        
        Returns:
            RAGAnswer object containing the response
        """
        # Track number of messages (queries) sent
        num_messages = 0

        # Initialize starting peer
        if query_peer_id is None:
            query_peer_id = random.choice(range(self.num_peers))
            logger.debug(f"Randomly selected starting peer: {query_peer_id}")
        
        current_peer_id = query_peer_id
        initial_peer_id = query_peer_id
        
        # Keep track of visited peers and their answers
        visited_ids = set()
        peer_answers_cache = {}
        
        # Track first answer
        candidate_answer = None
        candidate_peer_id = None
        candidate_data = None
        first_answer_hop = None

        logger.debug(f"Starting random walk search from peer {initial_peer_id}")
        logger.debug(f"Parameters: max_ttl={max_ttl}, restart_prob={restart_probability}")
        
        # Perform random walk
        for hop in range(max_ttl):
            # Check termination for defense
            if candidate_answer is not None and self.defense_enabled:
                peers_with_answers = sum(1 for ans in peer_answers_cache.values() if ans)
                additional_hops_used = hop - first_answer_hop
                
                if peers_with_answers >= self.defense_mechanism.min_peers:
                    logger.debug(f"Sufficient peers collected: {peers_with_answers}/{self.defense_mechanism.min_peers}")
                    break
                elif additional_hops_used >= self.max_additional_hops:
                    logger.debug(f"Max additional hops reached: {additional_hops_used}/{self.max_additional_hops}")
                    break
            elif candidate_answer is not None:
                break
            
            logger.debug(f"Hop {hop}: Querying peer {current_peer_id}")
            
            # Query current peer (even if visited before - random walk allows revisits)
            current_answer, relevant_knowledge, relevant_score, is_query_hit = \
                self.peers[current_peer_id].query(question, query_confidence_threshold)
            num_messages += 1
            
            # Cache answer and mark as visited
            answer_text = str(current_answer) if current_answer is not None else ""
            peer_answers_cache[current_peer_id] = answer_text
            visited_ids.add(current_peer_id)
            
            # Store first answer
            if current_answer is not None and candidate_answer is None:
                candidate_answer = current_answer
                candidate_peer_id = current_peer_id
                candidate_data = (relevant_knowledge, relevant_score, is_query_hit)
                first_answer_hop = hop
                logger.debug(f"First answer found at peer {current_peer_id}, hop {hop}")
                
                if self.defense_enabled:
                    logger.debug(f"Defense enabled: continuing to collect peers for validation")
            
            # Decide next move
            if random.random() < restart_probability:
                current_peer_id = initial_peer_id
                logger.debug(f"Random restart to initial peer: {initial_peer_id}")
            else:
                neighbors = list(self.network.neighbors(current_peer_id))
                if not neighbors:
                    logger.debug(f"Dead end at peer {current_peer_id}, restarting")
                    current_peer_id = initial_peer_id
                else:
                    current_peer_id = random.choice(neighbors)
                    logger.debug(f"Moving to neighbor peer {current_peer_id}")
        
        # Process results
        if candidate_answer is not None:
            logger.debug(f"Answer found at peer {candidate_peer_id} after {first_answer_hop} hops")
            logger.debug(f"Total messages sent: {num_messages}")
            
            # Apply defense validation
            final_answer, defense_info = self._validate_with_defense(
                question, str(candidate_answer), candidate_peer_id, peer_answers_cache,
                query_confidence_threshold
            )
            
            return RAGAnswer(
                answer=final_answer,
                relevant_knowledge=candidate_data[0],
                relevant_score=candidate_data[1],
                num_hops=first_answer_hop,
                num_messages=num_messages,
                is_query_hit=candidate_data[2],
                defense_info=defense_info
            )
        
        logger.debug(f"Search failed after {max_ttl} hops")
        logger.debug(f"Total messages sent: {num_messages}")
        
        # Return empty answer if no result found
        return RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=max_ttl,
            num_messages=num_messages,
            is_query_hit=False,
            defense_info={'defense_enabled': False, 'reason': 'No answer found'}
        )


    def flooding_query(
            self,
            question: str,
            query_peer_id: Optional[int] = None,
            query_confidence_threshold: float = 0.5,
            max_ttl: int = 6
    ) -> RAGAnswer:
        """
        Queries the network using a flooding algorithm.
        Extended routing for defense validation when enabled.
        
        Args:
            question: The question to ask
            query_peer_id: Starting peer ID (random if None)
            query_confidence_threshold: Minimum confidence required for an answer
            max_ttl: Maximum time-to-live (network depth to explore)
        
        Returns:
            RAGAnswer object containing the response
        """
        # Track number of messages (queries) sent
        num_messages = 0

        # Initialize starting peer
        if query_peer_id is None:
            query_peer_id = random.choice(range(self.num_peers))
            logger.debug(f"Randomly selected starting peer: {query_peer_id}")
        
        logger.debug(f"Starting flooding search from peer {query_peer_id}")
        logger.debug(f"Parameters: max_ttl={max_ttl}")
        
        # Use set for visited nodes and cache for answers
        visited_ids = {query_peer_id}
        peer_answers_cache = {}
        
        # Queue for BFS: (peer_id, hop)
        queue = deque([(query_peer_id, 0)])
        
        # Track first answer
        candidate_answer = None
        candidate_peer_id = None
        candidate_data = None
        first_answer_hop = None
        
        while queue:
            current_peer_id, hop = queue.popleft()
            
            # Check termination for defense
            if candidate_answer is not None and self.defense_enabled:
                peers_with_answers = sum(1 for ans in peer_answers_cache.values() if ans)
                additional_hops_used = hop - first_answer_hop
                
                if peers_with_answers >= self.defense_mechanism.min_peers:
                    logger.debug(f"Sufficient peers collected: {peers_with_answers}/{self.defense_mechanism.min_peers}")
                    break
                elif additional_hops_used >= self.max_additional_hops:
                    logger.debug(f"Max additional hops reached: {additional_hops_used}/{self.max_additional_hops}")
                    break
            elif candidate_answer is not None:
                break
            
            if hop >= max_ttl:
                continue

            logger.debug(f"Flooding at Hop {hop}, peer: {current_peer_id}")
            
            # Query current peer and track message
            current_answer, relevant_knowledge, relevant_score, is_query_hit = \
                self.peers[current_peer_id].query(question, query_confidence_threshold)
            num_messages += 1
            
            # Cache answer
            answer_text = str(current_answer) if current_answer is not None else ""
            peer_answers_cache[current_peer_id] = answer_text
            
            # Store first answer
            if current_answer is not None and candidate_answer is None:
                candidate_answer = current_answer
                candidate_peer_id = current_peer_id
                candidate_data = (relevant_knowledge, relevant_score, is_query_hit)
                first_answer_hop = hop
                logger.debug(f"First answer found at peer {current_peer_id}, hop {hop}")
                
                if self.defense_enabled:
                    logger.debug(f"Defense enabled: continuing to collect peers for validation")
            
            # Add all unvisited neighbors to queue
            neighbors = self.network.neighbors(current_peer_id)
            for neighbor_id in neighbors:
                if neighbor_id not in visited_ids:
                    visited_ids.add(neighbor_id)
                    queue.append((neighbor_id, hop + 1))
                    logger.debug(f"Added neighbor {neighbor_id} to queue at Hop {hop + 1}")
        
        # Process results
        if candidate_answer is not None:
            logger.debug(f"Answer found at peer {candidate_peer_id}, hop {first_answer_hop}")
            logger.debug(f"Total messages sent: {num_messages}")
            
            # Apply defense validation
            final_answer, defense_info = self._validate_with_defense(
                question, str(candidate_answer), candidate_peer_id, peer_answers_cache,
                query_confidence_threshold
            )

            return RAGAnswer(
                answer=final_answer,
                relevant_knowledge=candidate_data[0],
                relevant_score=candidate_data[1],
                num_hops=first_answer_hop,
                num_messages=num_messages,
                is_query_hit=candidate_data[2],
                defense_info=defense_info
            )
        
        # Log search statistics
        logger.debug("Search failed to find answer")
        logger.debug(f"Total messages sent: {num_messages}")
        
        # Return empty answer if no result found
        return RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=max_ttl,
            num_messages=num_messages,
            is_query_hit=False,
            defense_info={'defense_enabled': False, 'reason': 'No answer found'}
        )


class CRAGNetwork:
    def __init__(
            self, 
            llm_url: str, 
            llm_name: str, 
            llm_num_ctx: int,
            llm_seed: int, 
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
        ):
        self.text_embedding_model = SentenceTransformer(embedding_model)
        self.peer = Peer(0, llm_url, llm_name, llm_num_ctx, llm_seed, self.text_embedding_model)

    def init_knowledge(self, data_points: List[Datapoint]):
        """
        Distributes data points to peers based on their topics (Uniform).

        Args:
            data_points: A list of Datapoint objects.
        """
        # Distribute data points to peers based on assigned topics
        for data_point in tqdm(data_points, desc=f"Distributing data points to peers"):
            self.peer.add_knowledge(data_point)

    def query(
            self,
            question: str,
            query_confidence_threshold: float = 0.5
    ) -> RAGAnswer:
        """
        Queries the network using a random walk algorithm with restart probability.
        
        Args:
            question: The question to ask
            query_confidence_threshold: Minimum confidence required for an answer
        
        Returns:
            RAGAnswer object containing the response
        """
        # Query current peer
        current_answer, relevant_knowledge, relevant_score, is_query_hit = \
            self.peer.query(question, query_confidence_threshold)

        # Return if answer found
        if current_answer is not None:
            logger.debug(f"Answer found at peer 0")
            return RAGAnswer(
                answer=str(current_answer),
                relevant_knowledge=relevant_knowledge,
                relevant_score=relevant_score,
                num_hops=0,
                num_messages=0,
                is_query_hit=is_query_hit,
                defense_info={'defense_enabled': False}  # NEW
            )
            
        logger.debug(f"Search failed")
        
        # Return empty answer if no result found
        return RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=0,
            num_messages=0,
            is_query_hit=False,
            defense_info={'defense_enabled': False}  # NEW
        )


class NoRAGNetwork:
    def __init__(
            self, 
            llm_url: str, 
            llm_name: str, 
            llm_num_ctx: int,
            llm_seed: int
        ):
        self.peer = Peer(0, llm_url, llm_name, llm_num_ctx, llm_seed)

    def init_knowledge(self, data_points: List[Datapoint]):
        """
        No need to initialize knowledge for No RAG mode

        Args:
            data_points: A list of Datapoint objects.
        """
        pass

    def query(
            self,
            question: str
    ) -> RAGAnswer:
        """
        Queries the network using a random walk algorithm with restart probability.

        Args:
            question: The question to ask
            query_confidence_threshold: Minimum confidence required for an answer

        Returns:
            RAGAnswer object containing the response
        """
        # Query current peer
        generated_answer, relevant_knowledge, relevant_score, is_query_hit = \
            self.peer.query_no_rag(question)

        # Return empty answer if no result found
        return RAGAnswer(
            answer=str(generated_answer),
            relevant_knowledge=relevant_knowledge,
            relevant_score=relevant_score,
            num_hops=0,
            num_messages=0,
            is_query_hit=is_query_hit,
            defense_info={'defense_enabled': False}  # NEW
        )
