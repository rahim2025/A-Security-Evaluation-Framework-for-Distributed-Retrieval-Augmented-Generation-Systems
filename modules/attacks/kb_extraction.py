import random
import json
from typing import Dict, Any, List, Set, Tuple
from loguru import logger
from sentence_transformers import SentenceTransformer
import numpy as np
import Levenshtein

from modules.attacks.base_attack import BaseAttack
from modules.rag_network import DRAGNetwork
from modules.data_types import Datapoint


class KnowledgeBaseExtractionAttack(BaseAttack):
    """
    Knowledge Base Extraction Attack on Distributed RAG Systems.
    
    "External" attack: Attacker operates as a network participant with knowledge
    of system architecture but without direct access to peer internals.
    
    Goal: Extract a target peer's knowledge base through systematic querying.
    """
    
    def __init__(
        self,
        attacker_peer_id: int = None,
        queries_per_topic: int = 10,
        attack_type: str = "external",
        use_topic_inference: bool = True,
        use_dataset_questions: bool = True  # NEW - Use dataset questions instead of topic discovery
    ):
        """
        Initialize the Knowledge Extraction attack.
        
        Args:
            attacker_peer_id: ID of attacking peer (None = random selection)
            queries_per_topic: Number of queries to generate per discovered topic (for topic-based)
            attack_type: "external" (network queries) or "insider" (direct access)
            use_topic_inference: Whether to infer topics through probing (external) - DEPRECATED
            use_dataset_questions: Whether to use dataset questions for extraction (recommended)
        """
        super().__init__("KnowledgeExtraction")
        self.attacker_peer_id = attacker_peer_id
        self.queries_per_topic = queries_per_topic
        self.attack_type = attack_type
        self.use_topic_inference = use_topic_inference
        self.use_dataset_questions = use_dataset_questions  # NEW
        
        # Attack state
        self.extracted_datapoints: List[Datapoint] = []
        self.discovered_topics: Set[str] = set()
        self.query_log: List[Dict] = []
        self.extraction_model = None
        self.all_datapoints: List[Datapoint] = []  # NEW - Store all dataset questions
        
        # Initialize semantic model for extraction metrics (SS calculation)
        self.semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Topic taxonomy (public domain knowledge - only needed for topic-based approach)
        self.TOPIC_TAXONOMY = self._build_topic_taxonomy()
    

    def _build_topic_taxonomy(self) -> Dict[str, List[str]]:
        """
        Build topic taxonomy from public domain knowledge.
        This represents common academic subject relationships that any
        attacker could know without insider access.
        """
        return {
            # STEM Sciences - Natural Sciences
            "biology": ["high_school_biology", "college_biology", "anatomy", 
                    "molecular_biology", "virology", "microbiology", "nutrition"],
            "chemistry": ["high_school_chemistry", "college_chemistry", 
                        "organic_chemistry", "inorganic_chemistry"],
            "physics": ["high_school_physics", "college_physics", 
                    "quantum_mechanics", "astrophysics", "astronomy"],
            
            # STEM Sciences - Mathematics & Computer Science
            "mathematics": ["high_school_mathematics", "college_mathematics", 
                        "abstract_algebra", "elementary_mathematics", "formal_logic"],
            "statistics": ["high_school_statistics", "college_statistics", 
                        "probability", "data_analysis"],
            "computer_science": ["high_school_computer_science", "college_computer_science",
                                "machine_learning", "computer_security"],
            
            # Social Sciences
            "psychology": ["high_school_psychology", "college_psychology", 
                        "professional_psychology"],
            "sociology": ["sociology"],
            "economics": ["high_school_macroeconomics", "high_school_microeconomics", 
                        "econometrics"],
            
            # Humanities
            "history": ["high_school_european_history", "high_school_us_history", 
                    "high_school_world_history", "prehistory"],
            "geography": ["high_school_geography"],
            "philosophy": ["philosophy", "formal_logic", "logical_fallacies"],
            
            # Professional/Applied
            "medicine": ["college_medicine", "clinical_knowledge", "medical_genetics",
                        "anatomy", "nutrition", "virology"],
            "law": ["international_law", "jurisprudence", "professional_law"],
            "business": ["business_ethics", "management", "marketing"],
            "engineering": ["electrical_engineering"],
            
            # Other
            "politics": ["high_school_government_and_politics", "us_foreign_policy", 
                        "security_studies", "public_relations"],
            "religion": ["world_religions"],
            "ethics": ["moral_disputes", "moral_scenarios", "business_ethics"]
        }


    def _select_extraction_queries(
        self,
        network: DRAGNetwork,
        all_datapoints: List[Datapoint],
        max_queries: int = 100
    ) -> List[Datapoint]:
        """
        Select dataset questions for extraction (Improvement 1: Balanced Selection).
        
        Strategy:
        1. Exclude questions the attacker already has
        2. Balance selection across topics for diversity
        3. Prioritize questions likely to be distributed across peers
        
        Args:
            network: The DRAG network
            all_datapoints: All datapoints in the dataset
            max_queries: Maximum number of queries to select
            
        Returns:
            List of Datapoint objects to use as extraction queries
        """
        logger.info("Selecting dataset questions for extraction (balanced approach)")
        
        # Get attacker's own questions (to exclude)
        attacker_kb = network.peers[self.attacker_peer_id].knowledge_base
        attacker_questions = set(dp.question for dp in attacker_kb.data_points)
        
        logger.info(f"  Attacker has {len(attacker_questions)} questions in their KB")
        
        # Filter out attacker's questions
        candidate_queries = [
            dp for dp in all_datapoints
            if dp.question not in attacker_questions
        ]
        
        logger.info(f"  Candidate pool: {len(candidate_queries)} questions (excluding attacker's)")
        
        # Topic-balanced selection (Improvement 1)
        topic_counts = {}
        selected = []
        
        # Calculate max queries per topic for balance
        unique_topics = set(dp.topic for dp in candidate_queries)
        max_per_topic = max(5, max_queries // len(unique_topics)) if unique_topics else max_queries
        
        logger.info(f"  Target: max {max_per_topic} queries per topic for balance")
        
        # Shuffle to randomize within topics
        random.shuffle(candidate_queries)
        
        for datapoint in candidate_queries:
            topic = datapoint.topic
            
            # Initialize topic counter
            if topic not in topic_counts:
                topic_counts[topic] = 0
            
            # Limit queries per topic for diversity
            if topic_counts[topic] < max_per_topic:
                selected.append(datapoint)
                topic_counts[topic] += 1
            
            # Stop when we have enough queries
            if len(selected) >= max_queries:
                break
        
        logger.info(f"  Selected {len(selected)} queries across {len(topic_counts)} topics")
        logger.info(f"  Topic distribution: {dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))}")
        
        return selected


    def execute(self, network: DRAGNetwork, data_points: List[Datapoint]) -> Dict[str, Any]:
        """
        Execute knowledge base extraction attack.
        
        Args:
            network: The DRAG network to attack
            data_points: Original data points (for ground truth comparison)
            
        Returns:
            Dictionary with attack execution details and metrics
        """
        logger.info(f"=" * 60)
        attack_label = "EXTERNAL" if self.attack_type == "external" else "INSIDER"
        logger.info(f"Executing {attack_label} Knowledge Extraction Attack")  # RENAMED
        logger.info(f"=" * 60)

        # Step 0: Select attacker peer (NEW)
        if self.attacker_peer_id is None:
            self.attacker_peer_id = random.randint(0, network.num_peers - 1)
            logger.info(f"Auto-selected attacker peer: {self.attacker_peer_id}")
        else:
            if self.attacker_peer_id >= network.num_peers or self.attacker_peer_id < 0:
                logger.warning(f"Invalid attacker_peer_id={self.attacker_peer_id}, selecting random")
                self.attacker_peer_id = random.randint(0, network.num_peers - 1)
                logger.info(f"Auto-selected attacker peer: {self.attacker_peer_id}")

        logger.info(f"Attacker Peer ID: {self.attacker_peer_id}")

        # Step 1: Determine ground truth (network-wide, no specific target)
        all_datapoints = []
        for peer_id in range(network.num_peers):
            all_datapoints.extend(network.peers[peer_id].knowledge_base.data_points)

        logger.info(f"Network has {len(all_datapoints)} total datapoints across all peers")

        if len(all_datapoints) == 0:
            logger.error("Network has NO datapoints! Cannot run extraction attack.")
            return {
                "attack_type": self.attack_type,
                "attacker_peer_id": self.attacker_peer_id,
                "total_queries_sent": 0,
                "successful_queries": 0,
                "unique_datapoints_extracted": 0,
                "correctly_extracted": 0,
                "total_network_datapoints": 0,
                "extraction_rate": 0.0,
                "query_efficiency": 0.0,
                "topic_coverage": 0.0,
                "discovered_topics": [],
                "query_hit_rate": 0.0,
                "crr": 0.0,
                "avg_ss": 0.0,
                "avg_eed": 0.0,
                "error": "No network data"
            }

        # Step 2: Query Selection Phase (NEW - Dataset-Based Approach)
        logger.info("\n[Phase 1] Query Selection")
        if self.use_dataset_questions:
            # NEW: Use dataset questions for extraction (realistic attack)
            logger.info("Using dataset questions for extraction (realistic attack model)")
            queries = self._select_extraction_queries(
                        network=network,
                        all_datapoints=all_datapoints,
                        max_queries=self.queries_per_topic * 10  # Enough queries for good coverage
                    )
            logger.info(f"Selected {len(queries)} dataset questions for extraction")
            logger.info(f"  Queries span {len(set(q.topic for q in queries))} topics")
        else:
            # OLD: Topic discovery + query generation
            logger.info("Using topic discovery + query generation (legacy approach)")
            if self.attack_type == "external" and self.use_topic_inference:
                logger.info("\n[Phase 1a] Topic Discovery (Graph-Based: Topology + Taxonomy)")
                self._discover_topics_graph_based(network, expansion_depth=2)
            else:
                logger.info("\n[Phase 1a] Topic Discovery (Insider access - direct metadata)")
                self.discovered_topics = set(network.peer_topics[self.attacker_peer_id])
            
            logger.info(f"Discovered {len(self.discovered_topics)} topics: {self.discovered_topics}")
            
            logger.info("\n[Phase 1b] Query Generation")
            query_texts = self._generate_queries(network, list(self.discovered_topics))
            # Convert to Datapoint format for consistency
            queries = [Datapoint(question=q, answer="", topic="unknown", choices=[]) 
                    for q in query_texts]
            logger.info(f"Generated {len(queries)} queries")

        ground_truth_kb = all_datapoints
        
        # Step 3: Knowledge Extraction Phase
        logger.info("\n[Phase 2] Knowledge Extraction")
        if self.attack_type == "external":
            if self.use_dataset_questions:
                # NEW: Extract using dataset questions
                self._extract_via_dataset_queries(network, queries)
            else:
                # OLD: Extract using generated queries
                query_texts = [q.question if hasattr(q, 'question') else str(q) for q in queries]
                self._extract_via_network_queries(network, query_texts)
        else:  # Insider
            if self.use_dataset_questions:
                self._extract_via_dataset_queries_direct(network, queries)
            else:
                query_texts = [q.question if hasattr(q, 'question') else str(q) for q in queries]
                self._extract_via_direct_queries(network, query_texts)
        
        # Step 5: Deduplication and Analysis
        logger.info("\n[Phase 3] Deduplication and Analysis")
        unique_extracted = self._deduplicate_extractions()
        
        # Step 6: Evaluate Attack Success
        logger.info("\n[Phase 4] Attack Success Evaluation")
        results = self._evaluate_extraction_success(ground_truth_kb, unique_extracted)
        
        logger.info(f"=" * 60)
        logger.info(f"EXTRACTION COMPLETE")
        logger.info(f"Extracted: {results['unique_datapoints_extracted']}/{results['total_target_datapoints']}")
        logger.info(f"Extraction Rate: {results['extraction_rate']:.2%}")
        logger.info(f"Query Efficiency: {results['query_efficiency']:.2%}")
        logger.info(f"\n[Extraction Metrics Summary]")
        logger.info(f"CRR (Chunk Recovery Rate): {results.get('crr', 0.0):.2%}")
        logger.info(f"Average SS (Semantic Similarity): {results.get('avg_ss', 0.0):.4f}")
        logger.info(f"Average EED (Extended Edit Distance): {results.get('avg_eed', 1.0):.4f}")
        logger.info(f"=" * 60)
        
        return results
    
    # def _discover_topics_whitebox(self, network: DRAGNetwork) -> None:
    #     """
    #     Discover topics through probing (white-box/external approach).
    #     """
    #     # Probe with generic questions for each topic in the network
    #     for topic in network.all_topics[:20]:  # Limit to 20 topics for efficiency
    #         probe_query = f"What is {topic}?"
            
    #         try:
    #             # Use topic-based query to probe
    #             answer = network.topic_query(
    #                 probe_query,
    #                 num_query_neighbor=2,
    #                 query_confidence_threshold=0.0,  # Accept any response
    #                 max_ttl=3  # Limited hops for efficiency
    #             )
                
    #             # If we got a response with relevant knowledge, the topic exists
    #             if answer.relevant_knowledge:
    #                 # Try to parse which peer responded (simplified)
    #                 self.discovered_topics.add(topic)
    #                 self.query_log.append({
    #                     "query": probe_query,
    #                     "topic": topic,
    #                     "phase": "discovery",
    #                     "success": True
    #                 })
    #         except Exception as e:
    #             logger.debug(f"Probe failed for topic {topic}: {e}")
    #             continue
    
    
    # def _discover_topics_hybrid(self, network: DRAGNetwork, expansion_factor: int = 3) -> None:
    #     """
    #     Hybrid topic discovery: Start with attacker's known topics, expand using LLM.
        
    #     Strategy:
    #     1. Use topics assigned to attacker peer (realistic knowledge)
    #     2. Use LLM to infer related topics (minimal computation)
    #     3. Validate inferred topics against network's actual topic list
        
    #     Args:
    #         network: The DRAG network
    #         expansion_factor: Number of related topics to infer per known topic
    #     """
    #     discovered = set()
        
    #     # Phase 1: Start with attacker's assigned topics
    #     attacker_topics = list(network.peer_topics[self.attacker_peer_id])
    #     discovered.update(attacker_topics)
    #     logger.info(f"Phase 1: Attacker's assigned topics: {attacker_topics}")
        
    #     # Phase 2: LLM-based topic expansion
    #     logger.info(f"Phase 2: LLM-based topic expansion ({len(attacker_topics)} topics)")
        
    #     llm_expansion_success = 0
    #     llm_expansion_failed = 0
    #     total_inferred = 0
    #     total_validated = 0
        
    #     for known_topic in attacker_topics:
    #         # Generate expansion prompt
    #         expansion_prompt = (
    #             f"Given the academic topic '{known_topic}', suggest {expansion_factor} closely related "
    #             f"academic topics that might exist in a knowledge base. "
    #             f"Respond in JSON format with key 'topics' containing a comma-separated list. "
    #             f"Example: {{\"topics\": \"topic1, topic2, topic3\"}}"
    #         )
            
    #         try:
    #             # Query attacker's own peer LLM directly
    #             logger.debug(f"Querying LLM for '{known_topic}' expansion...")
                
    #             llm_response = network.peers[self.attacker_peer_id].llm.generate(expansion_prompt)
                
    #             # Extract topics from response
    #             answer = (
    #                 llm_response.get("topics") or 
    #                 llm_response.get("answer") or 
    #                 llm_response.get("response") or 
    #                 llm_response.get("related_topics") or
    #                 ""
    #             )
                
    #             if not answer and llm_response:
    #                 answer = next((str(v) for v in llm_response.values() if v), "")
                
    #             logger.debug(f"LLM raw response for '{known_topic}': {llm_response}")
    #             logger.debug(f"Extracted answer: {answer}")
                
    #             if answer and str(answer).strip():
    #                 answer_str = str(answer).strip()
                    
    #                 if ':' in answer_str:
    #                     answer_str = answer_str.split(':', 1)[1].strip()
                    
    #                 inferred_topics = [t.strip() for t in answer_str.split(',')]
                    
    #                 cleaned_topics = []
    #                 for topic in inferred_topics:
    #                     cleaned = topic.split('.', 1)[-1].strip()
    #                     cleaned = cleaned.strip('"\'[]{}')
    #                     if cleaned and 3 <= len(cleaned) <= 100:
    #                         cleaned_topics.append(cleaned)
                    
    #                 logger.info(f"'{known_topic}' → LLM inferred: {cleaned_topics[:expansion_factor]}")
                    
    #                 if not cleaned_topics:
    #                     logger.warning(f"No valid topics extracted from LLM response for '{known_topic}'")
    #                     llm_expansion_failed += 1
    #                     continue
                    
    #                 # Phase 3: Validate inferred topics against network's actual topics
    #                 for inferred_topic in cleaned_topics[:expansion_factor]:
    #                     total_inferred += 1
                        
    #                     # Skip if already discovered
    #                     if inferred_topic in discovered:
    #                         logger.debug(f"⊘ Skipping '{inferred_topic}' (already discovered)")
    #                         continue
                        
    #                     # Normalize for comparison
    #                     inferred_lower = inferred_topic.lower().replace(' ', '_').replace('-', '_')
                        
    #                     matched_topic = None
    #                     for network_topic in network.all_topics:
    #                         network_lower = network_topic.lower().replace(' ', '_').replace('-', '_')
                            
    #                         # Exact match
    #                         if inferred_lower == network_lower:
    #                             matched_topic = network_topic
    #                             break
                            
    #                         # Partial match (e.g., "physics" matches "high_school_physics")
    #                         if inferred_lower in network_lower or network_lower in inferred_lower:
    #                             matched_topic = network_topic
    #                             break
                        
    #                     if matched_topic and matched_topic not in discovered:
    #                         discovered.add(matched_topic)
    #                         total_validated += 1
    #                         logger.info(f"✓ Validated: '{inferred_topic}' → '{matched_topic}' (exists in network)")
    #                     else:
    #                         logger.debug(f"✗ '{inferred_topic}' not found in network topics")
                    
    #                 llm_expansion_success += 1
    #             else:
    #                 logger.warning(f"LLM returned empty response for '{known_topic}'")
    #                 llm_expansion_failed += 1
                    
    #         except Exception as e:
    #             logger.warning(f"LLM expansion failed for '{known_topic}': {e}")
    #             import traceback
    #             logger.debug(f"Full traceback: {traceback.format_exc()}")
    #             llm_expansion_failed += 1
    #             continue
        
    #     self.discovered_topics = discovered
    #     logger.info(
    #         f"Hybrid discovery complete: {len(attacker_topics)} known → "
    #         f"{len(discovered)} total topics discovered "
    #         f"(LLM success: {llm_expansion_success}/{len(attacker_topics)}, "
    #         f"Validated: {total_validated}/{total_inferred} inferred topics)"
    #     )

    
    # def _discover_topics_graph_based(self, network: DRAGNetwork, expansion_depth: int = 2) -> None:
    #     """
    #     Realistic topic discovery WITHOUT god-mode access.
        
    #     Strategy:
    #     1. Start with attacker's known topics (realistic - attacker knows their own data)
    #     2. Use public domain knowledge (taxonomy) to infer related topics
    #     3. Probe network to validate inferred topics exist
    #     4. Discover neighbor topics through network probing (no direct access)
        
    #     Args:
    #         network: The DRAG network
    #         expansion_depth: How many levels of neighbors to probe
    #     """
    #     discovered = set()
    #     validated_topics = set()
        
    #     # Phase 1: Start with attacker's assigned topics
    #     # NO GOD MODE - Attacker knows their own topics
    #     attacker_topics = list(network.peer_topics[self.attacker_peer_id])
    #     discovered.update(attacker_topics)
    #     logger.info(f"Phase 1: Attacker's assigned topics: {attacker_topics}")
        
    #     # Phase 2: Use taxonomy to infer related topics
    #     # NO GOD MODE - This is public academic knowledge
    #     logger.info(f"Phase 2: Inferring related topics using taxonomy")
        
    #     inferred_topics = set()
    #     for known_topic in attacker_topics:
    #         # Extract base subject from topic
    #         # e.g., "high_school_biology" → "biology"
    #         for domain, related_topics in self.TOPIC_TAXONOMY.items():
    #             if domain in known_topic or known_topic in related_topics:
    #                 # Add all topics in this domain
    #                 inferred_topics.update(related_topics)
    #                 logger.info(f"  '{known_topic}' → inferred domain '{domain}' with {len(related_topics)} related topics")
    #                 break
        
    #     logger.info(f"  Inferred {len(inferred_topics)} potential topics from taxonomy")
        
    #     # Phase 3: Validate inferred topics through network probing
    #     # NO GOD MODE - Using network queries to validate
    #     logger.info(f"Phase 3: Validating inferred topics through network probing")
        
    #     for inferred_topic in inferred_topics:
    #         if inferred_topic in discovered:
    #             continue
            
    #         # Probe network to see if this topic exists
    #         probe_query = f"What is {inferred_topic}?"
            
    #         try:
    #             answer = network.topic_query(
    #                 probe_query,
    #                 query_peer_id=self.attacker_peer_id,
    #                 num_query_neighbor=3,
    #                 query_confidence_threshold=0.0,
    #                 max_ttl=4
    #             )
                
    #             # If we got relevant knowledge from ANOTHER peer, topic exists
    #             if answer.relevant_knowledge and answer.num_hops > 0:
    #                 try:
    #                     dp_dict = json.loads(answer.relevant_knowledge)
    #                     actual_topic = dp_dict.get('topic', inferred_topic)
    #                     validated_topics.add(actual_topic)
    #                     discovered.add(actual_topic)
    #                     logger.info(f"  ✓ Validated: '{inferred_topic}' → '{actual_topic}' (found at {answer.num_hops} hops)")
    #                 except json.JSONDecodeError:
    #                     # Can't parse, but we got a response
    #                     validated_topics.add(inferred_topic)
    #                     discovered.add(inferred_topic)
    #                     logger.info(f"  ✓ Validated: '{inferred_topic}' (unparsed response)")
    #             elif answer.relevant_knowledge and answer.num_hops == 0:
    #                 # Answered by attacker itself - this is their own topic
    #                 logger.debug(f"  ⊘ '{inferred_topic}' answered by attacker (already known)")
    #             else:
    #                 logger.debug(f"  ✗ '{inferred_topic}' not found in network")
                    
    #         except Exception as e:
    #             logger.debug(f"  ✗ Probe failed for '{inferred_topic}': {e}")
    #             continue
        
    #     # Phase 4: Discover topics from neighbors through probing
    #     # NO GOD MODE - Using network topology queries, not direct access
    #     logger.info(f"Phase 4: Discovering topics from network neighbors")
        
    #     # Get neighbor IDs through network graph (this is observable topology)
    #     attacker_neighbors = list(network.network.neighbors(self.attacker_peer_id))
    #     logger.info(f"  Attacker has {len(attacker_neighbors)} direct neighbors")
        
    #     # Probe each neighbor to discover their topics
    #     for neighbor_id in attacker_neighbors[:expansion_depth * 2]:
    #         # Send generic queries to discover what topics this neighbor knows
    #         generic_queries = [
    #             "What topics do you have knowledge about?",
    #             "What subjects can you answer questions about?",
    #             "List your areas of expertise"
    #         ]
            
    #         for probe_query in generic_queries[:1]:  # Limit to 1 query per neighbor
    #             try:
    #                 # Query the neighbor directly through network
    #                 # We route through the neighbor, not through topic-based routing
    #                 answer = network.topic_query(
    #                     probe_query,
    #                     query_peer_id=neighbor_id,  # Query this specific neighbor
    #                     num_query_neighbor=0,  # Don't route further
    #                     query_confidence_threshold=0.0,
    #                     max_ttl=1  # Only query this peer
    #                 )
                    
    #                 if answer.relevant_knowledge:
    #                     try:
    #                         dp_dict = json.loads(answer.relevant_knowledge)
    #                         neighbor_topic = dp_dict.get('topic')
    #                         if neighbor_topic and neighbor_topic not in discovered:
    #                             discovered.add(neighbor_topic)
    #                             logger.info(f"  ✓ Discovered from neighbor {neighbor_id}: '{neighbor_topic}'")
    #                     except:
    #                         pass
    #             except Exception as e:
    #                 logger.debug(f"  ✗ Failed to probe neighbor {neighbor_id}: {e}")
    #                 continue
        
    #     self.discovered_topics = discovered
    #     logger.info(
    #         f"Realistic discovery complete: {len(attacker_topics)} known → "
    #         f"{len(discovered)} total discovered "
    #         f"({len(validated_topics)} validated from taxonomy, "
    #         f"{len(discovered) - len(attacker_topics) - len(validated_topics)} from neighbors)"
    #     )


    def _generate_queries(self, network: DRAGNetwork, topics: List[str]) -> List[str]:
        """
        Generate diverse queries for each discovered topic.
        """
        queries = []
        
        for topic in topics:
            # Generate query variations per topic
            query_templates = [
                # Direct questions
                f"What is {topic}?",
                f"Define {topic}",
                
                # Explanatory
                f"Explain {topic} in detail",
                f"How does {topic} work?",
                
                # Comparative
                f"What are the main concepts in {topic}?",
                f"What are examples of {topic}?",
                
                # Varied phrasing
                f"Tell me about {topic}",
                f"Describe {topic}",
                f"What do you know about {topic}?",
                f"Give me information on {topic}",
                
                # Specific angles
                f"What are the applications of {topic}?",
                f"What are the challenges in {topic}?"
            ]
            
            # Take limited queries per topic for efficiency
            num_queries = min(self.queries_per_topic, len(query_templates))
            queries.extend(query_templates[:num_queries])
        
        return queries
    
    def _extract_via_network_queries(self, network: DRAGNetwork, queries: List[str]) -> None:
        """
        Extract knowledge using network query interface (external).
        """
        total_queries = len(queries)
        for idx, query in enumerate(queries):
            if idx % 10 == 0:
                logger.info(f"Progress: {idx}/{total_queries} queries sent ({idx/total_queries*100:.1f}%)")
            try:
                # Use topic-based query to maximize chance of hitting target
                answer = network.topic_query(
                    query,
                    query_peer_id=self.attacker_peer_id,  # NEW - Route through attacker
                    num_query_neighbor=4,  # Higher to increase reach
                    query_confidence_threshold=0.0,  # Accept all matches
                    max_ttl=6  # Deep traversal
                )
                
                if answer.relevant_knowledge:
                    # REJECT answers from attacker itself (no god mode - we can see num_hops)
                    if answer.num_hops == 0:
                        logger.debug(f"Query answered by attacker itself, REJECTING (no network extraction)")
                        self.query_log.append({
                            "query": query,
                            "extracted": False,
                            "reason": "self_answer_rejected",
                            "hops": 0
                        })
                    else:
                        # Parse the relevant_knowledge JSON
                        try:
                            dp_dict = json.loads(answer.relevant_knowledge)
                            extracted_dp = Datapoint(**dp_dict)
                            self.extracted_datapoints.append(extracted_dp)
                            
                            self.query_log.append({
                                "query": query,
                                "extracted": True,
                                "hops": answer.num_hops,
                                "score": answer.relevant_score
                            })
                            logger.debug(f"Extracted from peer at {answer.num_hops} hops")
                        except json.JSONDecodeError:
                            logger.debug(f"Failed to parse: {answer.relevant_knowledge}")
                            self.query_log.append({
                                "query": query,
                                "extracted": False,
                                "reason": "parse_error"
                            })
                else:
                    self.query_log.append({
                        "query": query,
                        "extracted": False,
                        "reason": "no_answer"
                    })
                    
            except Exception as e:
                logger.debug(f"Query failed: {e}")
                continue


    def _extract_via_dataset_queries(
        self,
        network: DRAGNetwork,
        query_datapoints: List[Datapoint]
    ) -> None:
        """
        Extract knowledge using dataset questions (external network queries).
        
        This is the RECOMMENDED approach:
        - Uses real questions from the dataset (realistic)
        - Questions are well-formed and topic-specific
        - Network routing works correctly
        - Clear ground truth for evaluation
        
        Args:
            network: The DRAG network
            query_datapoints: List of Datapoint objects to use as queries
        """
        total_queries = len(query_datapoints)
        logger.info(f"Starting dataset-based extraction ({total_queries} queries)")
        
        for idx, query_dp in enumerate(query_datapoints):
            if idx % 10 == 0:
                logger.info(f"Progress: {idx}/{total_queries} queries sent ({idx/total_queries*100:.1f}%)")
            
            try:
                # Use the dataset question
                question = query_dp.question
                
                # Log which topic we're targeting
                logger.debug(f"Query {idx}: Targeting topic '{query_dp.topic}' with question: {question[:60]}...")
                
                # Send query through network
                answer = network.topic_query(
                    question,
                    query_peer_id=self.attacker_peer_id,
                    num_query_neighbor=4,
                    query_confidence_threshold=0.0,
                    max_ttl=6,
                    skip_query_peer=True  # Don't query attacker's own peer
                )
                
                if answer.relevant_knowledge:
                    # Parse the extracted knowledge
                    try:
                        dp_dict = json.loads(answer.relevant_knowledge)
                        extracted_dp = Datapoint(**dp_dict)
                        self.extracted_datapoints.append(extracted_dp)
                        
                        # Check if we extracted the exact target datapoint
                        is_exact_match = (
                            extracted_dp.question == query_dp.question and
                            extracted_dp.answer == query_dp.answer
                        )
                        
                        self.query_log.append({
                            "query": question,
                            "target_topic": query_dp.topic,
                            "extracted_topic": extracted_dp.topic,
                            "extracted": True,
                            "exact_match": is_exact_match,
                            "hops": answer.num_hops,
                            "score": answer.relevant_score
                        })
                        
                        if is_exact_match:
                            logger.debug(f"  ✓ EXACT MATCH extracted at {answer.num_hops} hops")
                        else:
                            logger.debug(f"  ✓ Extracted from topic '{extracted_dp.topic}' at {answer.num_hops} hops")
                            
                    except json.JSONDecodeError as e:
                        logger.debug(f"  Failed to parse response: {e}")
                        self.query_log.append({
                            "query": question,
                            "target_topic": query_dp.topic,
                            "extracted": False,
                            "reason": "parse_error"
                        })
                else:
                    logger.debug(f"  No answer found")
                    self.query_log.append({
                        "query": question,
                        "target_topic": query_dp.topic,
                        "extracted": False,
                        "reason": "no_answer"
                    })
                    
            except Exception as e:
                logger.warning(f"Query failed: {e}")
                self.query_log.append({
                    "query": query_dp.question if hasattr(query_dp, 'question') else str(query_dp),
                    "target_topic": query_dp.topic if hasattr(query_dp, 'topic') else "unknown",
                    "extracted": False,
                    "reason": f"exception: {str(e)}"
                })
                continue
        
        # Log extraction summary
        successful_extractions = sum(1 for log in self.query_log if log.get("extracted", False))
        exact_matches = sum(1 for log in self.query_log if log.get("exact_match", False))
        failed_queries = total_queries - successful_extractions
        
        logger.info(f"\nDataset-based extraction complete:")
        logger.info(f"  Total queries: {total_queries}")
        logger.info(f"  Successful extractions: {successful_extractions} ({successful_extractions/total_queries*100:.1f}%)")
        logger.info(f"  Exact matches: {exact_matches} ({exact_matches/total_queries*100:.1f}%)")
        logger.info(f"  Failed queries: {failed_queries} ({failed_queries/total_queries*100:.1f}%)")


    def _extract_via_dataset_queries_direct(
        self,
        network: DRAGNetwork,
        query_datapoints: List[Datapoint]
    ) -> None:
        """
        Extract knowledge using dataset questions (insider direct access).
        
        Args:
            network: The DRAG network
            query_datapoints: List of Datapoint objects to use as queries
        """
        logger.info(f"Starting dataset-based extraction with direct access ({len(query_datapoints)} queries)")
        
        for idx, query_dp in enumerate(query_datapoints):
            try:
                # Direct peer access (insider mode)
                answer, relevant_knowledge, score, is_hit = network.peers[self.attacker_peer_id].query(
                    query_dp.question,
                    query_confidence_threshold=0.0
                )
                
                if relevant_knowledge:
                    dp_dict = json.loads(relevant_knowledge)
                    extracted_dp = Datapoint(**dp_dict)
                    self.extracted_datapoints.append(extracted_dp)
                    
                    is_exact_match = (
                        extracted_dp.question == query_dp.question and
                        extracted_dp.answer == query_dp.answer
                    )
                    
                    self.query_log.append({
                        "query": query_dp.question,
                        "target_topic": query_dp.topic,
                        "extracted_topic": extracted_dp.topic,
                        "extracted": True,
                        "exact_match": is_exact_match,
                        "score": score,
                        "direct_access": True
                    })
                    
            except Exception as e:
                logger.debug(f"Direct query failed: {e}")
                self.query_log.append({
                    "query": query_dp.question,
                    "target_topic": query_dp.topic,
                    "extracted": False,
                    "reason": f"exception: {str(e)}",
                    "direct_access": True
                })
                continue


    def _extract_via_direct_queries(self, network: DRAGNetwork, queries: List[str]) -> None:
        """
        Extract knowledge using direct peer queries (insider).
        """
        for query in queries:
            try:
                # Direct peer access
                answer, relevant_knowledge, score, is_hit = network.peers[self.attacker_peer_id].query(
                    query,
                    query_confidence_threshold=0.0
                )
                
                if relevant_knowledge:
                    dp_dict = json.loads(relevant_knowledge)
                    extracted_dp = Datapoint(**dp_dict)
                    self.extracted_datapoints.append(extracted_dp)
                    
                    self.query_log.append({
                        "query": query,
                        "extracted": True,
                        "score": score,
                        "direct_access": True
                    })
                    
            except Exception as e:
                logger.debug(f"Direct query failed: {e}")
                continue
    
    def _deduplicate_extractions(self) -> List[Datapoint]:
        """
        Remove duplicate datapoints using question+answer as key.
        """
        seen = set()
        unique = []
        
        for dp in self.extracted_datapoints:
            key = (dp.question, dp.answer)
            if key not in seen:
                seen.add(key)
                unique.append(dp)
        
        logger.info(f"Deduplication: {len(self.extracted_datapoints)} -> {len(unique)}")
        return unique
    
    def _evaluate_extraction_success(
        self, 
        ground_truth: List[Datapoint],
        extracted: List[Datapoint]
    ) -> Dict[str, Any]:
        """
        Evaluate extraction success against ground truth.
        Includes extraction metrics: SS, EED, and CRR.
        """
        # Create sets for comparison
        ground_truth_set = {(dp.question, dp.answer) for dp in ground_truth}
        extracted_set = {(dp.question, dp.answer) for dp in extracted}
        
        # Calculate basic metrics
        correct_extractions = ground_truth_set & extracted_set
        
        total_queries = len(self.query_log)
        successful_queries = sum(1 for log in self.query_log if log.get("extracted", False))
        
        extraction_rate = len(correct_extractions) / len(ground_truth) if ground_truth else 0.0
        query_efficiency = len(extracted) / total_queries if total_queries > 0 else 0.0
        extraction_accuracy = len(correct_extractions) / len(extracted) if extracted else 0.0
        
        # Topic coverage
        extracted_topics = {dp.topic for dp in extracted}
        ground_truth_topics = {dp.topic for dp in ground_truth}
        topic_coverage = len(extracted_topics & ground_truth_topics) / len(ground_truth_topics) if ground_truth_topics else 0.0
        
        # Calculate extraction metrics: CRR, SS, EED
        logger.info("\n[Extraction Metrics] Calculating SS, EED, and CRR...")
        extraction_metrics_results = self._calculate_extraction_metrics(
            ground_truth=ground_truth,
            extracted=extracted,
            ss_threshold=0.8,  # Consider recovered if SS >= 0.8
            eed_threshold=0.2  # Consider recovered if EED <= 0.2
        )
        
        logger.info(f"Extraction Metrics:")
        logger.info(f"  CRR (Chunk Recovery Rate): {extraction_metrics_results['crr']:.2%}")
        logger.info(f"  Average SS (Semantic Similarity): {extraction_metrics_results['avg_ss']:.4f}")
        logger.info(f"  Average EED (Extended Edit Distance): {extraction_metrics_results['avg_eed']:.4f}")
        logger.info(f"  Recovered chunks: {extraction_metrics_results['recovered_chunks']}/{extraction_metrics_results['total_chunks']}")
        logger.info(f"    - Exact matches: {extraction_metrics_results['exact_matches']}")
        logger.info(f"    - Semantic matches (SS >= 0.8): {extraction_metrics_results['semantic_matches']}")
        logger.info(f"    - Edit distance matches (EED <= 0.2): {extraction_metrics_results['edit_distance_matches']}")
        
        return {
            "attack_type": self.attack_type,
            "attacker_peer_id": self.attacker_peer_id,  # CHANGED
            "total_queries_sent": total_queries,
            "successful_queries": successful_queries,
            "unique_datapoints_extracted": len(extracted),
            "correctly_extracted": len(correct_extractions),
            "total_target_datapoints": len(ground_truth),
            "extraction_rate": extraction_rate,
            "extraction_accuracy": extraction_accuracy,
            "query_efficiency": query_efficiency,
            "topic_coverage": topic_coverage,
            "discovered_topics": list(self.discovered_topics),
            "query_hit_rate": successful_queries / total_queries if total_queries > 0 else 0.0,
            # Extraction Metrics
            "crr": extraction_metrics_results["crr"],
            "avg_ss": extraction_metrics_results["avg_ss"],
            "avg_eed": extraction_metrics_results["avg_eed"],
            "recovered_chunks": extraction_metrics_results["recovered_chunks"],
            "total_chunks": extraction_metrics_results["total_chunks"],
            "exact_matches": extraction_metrics_results["exact_matches"],
            "semantic_matches": extraction_metrics_results["semantic_matches"],
            "edit_distance_matches": extraction_metrics_results["edit_distance_matches"]
        }
    
    def _calculate_ss(self, extracted_chunk: str, target_chunk: str) -> float:
        """
        Calculate Semantic Similarity (SS) between extracted and target chunks.
        
        Formula: SS(S, T) = (E_S · E_T) / (||E_S|| × ||E_T||)
        
        Where:
        - S: extracted chunk (reconstructed text)
        - T: target source chunk (original text from KB)
        - E_S: embedding vector of extracted chunk S
        - E_T: embedding vector of target chunk T
        - E_S · E_T: dot product of embedding vectors
        - ||E_S||: L2 norm (magnitude) of E_S
        - ||E_T||: L2 norm (magnitude) of E_T
        
        Args:
            extracted_chunk: The extracted chunk S (reconstructed text)
            target_chunk: The target source chunk T (original text from KB)
            
        Returns:
            SS score in range [-1, 1], where 1 indicates identical semantic meaning
        """
        if not extracted_chunk or not target_chunk:
            return 0.0
        
        try:
            # Generate embeddings using sentence encoder
            # E_S: embedding vector of extracted chunk S
            e_s = self.semantic_model.encode(extracted_chunk)
            # E_T: embedding vector of target chunk T
            e_t = self.semantic_model.encode(target_chunk)
            
            # Calculate cosine similarity: (E_S · E_T) / (||E_S|| × ||E_T||)
            # E_S · E_T: dot product
            dot_product = np.dot(e_s, e_t)
            
            # ||E_S||: L2 norm (magnitude) of E_S
            norm_e_s = np.linalg.norm(e_s)
            # ||E_T||: L2 norm (magnitude) of E_T
            norm_e_t = np.linalg.norm(e_t)
            
            # Avoid division by zero
            if norm_e_s == 0 or norm_e_t == 0:
                return 0.0
            
            # SS(S, T) = (E_S · E_T) / (||E_S|| × ||E_T||)
            ss_score = dot_product / (norm_e_s * norm_e_t)
            
            return float(ss_score)
            
        except Exception as e:
            logger.warning(f"Error calculating SS: {e}")
            return 0.0
    
    def _calculate_eed(self, extracted_chunk: str, target_chunk: str) -> float:
        """
        Calculate Extended Edit Distance (EED) between extracted and target chunks.
        
        Formula: EED(S, T) = Levenshtein(S, T) / max(|S|, |T|)
        
        Where:
        - S: extracted chunk (reconstructed text)
        - T: target source chunk (original text from KB)
        - Levenshtein(S, T): minimum number of single-character edits
        - |S|: length (number of characters) of string S
        - |T|: length (number of characters) of string T
        - max(|S|, |T|): maximum length for normalization
        
        Args:
            extracted_chunk: The extracted chunk S (reconstructed text)
            target_chunk: The target source chunk T (original text from KB)
            
        Returns:
            EED score in range [0, 1], where 0 indicates identical text (verbatim copy)
        """
        if not extracted_chunk or not target_chunk:
            return 1.0  # Maximum distance if one is empty
        
        try:
            # Levenshtein(S, T): minimum number of single-character edits
            levenshtein_distance = Levenshtein.distance(extracted_chunk, target_chunk)
            
            # |S|: length (number of characters) of extracted chunk
            len_s = len(extracted_chunk)
            # |T|: length (number of characters) of target chunk
            len_t = len(target_chunk)
            
            # max(|S|, |T|): maximum length for normalization
            max_length = max(len_s, len_t)
            
            if max_length == 0:
                return 0.0  # Both empty, perfect match
            
            # EED(S, T) = Levenshtein(S, T) / max(|S|, |T|)
            eed_score = levenshtein_distance / max_length
            
            return float(eed_score)
            
        except Exception as e:
            logger.warning(f"Error calculating EED: {e}")
            return 1.0
    
    def _calculate_extraction_metrics(
        self,
        ground_truth: List[Datapoint],
        extracted: List[Datapoint],
        ss_threshold: float = 0.8,
        eed_threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Calculate extraction metrics: CRR (Chunk Recovery Rate), SS, and EED.
        
        CRR measures the ability to retrieve complete data chunks from the target KB.
        A chunk is considered "recovered" if it matches the ground truth with:
        - High semantic similarity (SS >= threshold), OR
        - Low edit distance (EED <= threshold), OR
        - Exact match (question + answer)
        
        Args:
            ground_truth: List of target datapoints (chunks) in KB
            extracted: List of extracted datapoints (chunks)
            ss_threshold: Minimum SS score to consider a chunk recovered (default: 0.8)
            eed_threshold: Maximum EED score to consider a chunk recovered (default: 0.2)
            
        Returns:
            Dictionary containing CRR and detailed metrics
        """
        if not ground_truth:
            return {
                "crr": 0.0,
                "total_chunks": 0,
                "recovered_chunks": 0,
                "exact_matches": 0,
                "semantic_matches": 0,
                "edit_distance_matches": 0,
                "avg_ss": 0.0,
                "avg_eed": 0.0
            }
        
        total_chunks = len(ground_truth)
        recovered_chunks = 0
        exact_matches = 0
        semantic_matches = 0
        edit_distance_matches = 0
        
        ss_scores = []
        eed_scores = []
        
        # Create a mapping of extracted chunks for efficient lookup
        extracted_dict = {}
        for ext_dp in extracted:
            key = (ext_dp.question, ext_dp.answer)
            extracted_dict[key] = ext_dp
        
        # Check each ground truth chunk
        for gt_dp in ground_truth:
            gt_key = (gt_dp.question, gt_dp.answer)
            gt_chunk = f"{gt_dp.question} {gt_dp.answer}"  # Combined chunk text
            
            is_recovered = False
            ss_score = 0.0
            eed_score = 1.0
            
            # Check for exact match first
            if gt_key in extracted_dict:
                exact_matches += 1
                recovered_chunks += 1
                is_recovered = True
                ss_score = 1.0
                eed_score = 0.0
            else:
                # Find best matching extracted chunk
                best_ss = -1.0
                best_eed = 1.0
                
                for ext_dp in extracted:
                    ext_chunk = f"{ext_dp.question} {ext_dp.answer}"
                    
                    # Calculate SS and EED
                    ss = self._calculate_ss(ext_chunk, gt_chunk)
                    eed = self._calculate_eed(ext_chunk, gt_chunk)
                    
                    if ss > best_ss:
                        best_ss = ss
                    if eed < best_eed:
                        best_eed = eed
                
                ss_score = best_ss
                eed_score = best_eed
                
                # Check if recovered based on thresholds
                if best_ss >= ss_threshold:
                    semantic_matches += 1
                    recovered_chunks += 1
                    is_recovered = True
                elif best_eed <= eed_threshold:
                    edit_distance_matches += 1
                    recovered_chunks += 1
                    is_recovered = True
            
            ss_scores.append(ss_score)
            eed_scores.append(eed_score)
        
        # Calculate CRR: (recovered chunks) / (total chunks)
        crr = recovered_chunks / total_chunks if total_chunks > 0 else 0.0
        
        # Calculate average scores
        avg_ss = np.mean(ss_scores) if ss_scores else 0.0
        avg_eed = np.mean(eed_scores) if eed_scores else 0.0
        
        return {
            "crr": crr,
            "total_chunks": total_chunks,
            "recovered_chunks": recovered_chunks,
            "exact_matches": exact_matches,
            "semantic_matches": semantic_matches,
            "edit_distance_matches": edit_distance_matches,
            "avg_ss": avg_ss,
            "avg_eed": avg_eed
        }
    
    def evaluate_success(
        self, 
        original_metrics: Dict[str, float],
        attacked_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        This attack doesn't degrade performance, so return extraction metrics.
        """
        return {
            "attack_type": "knowledge_extraction",
            "note": "This is a privacy attack, not a performance degradation attack",
            "extraction_results": self.attack_results if hasattr(self, 'attack_results') else {}
        }