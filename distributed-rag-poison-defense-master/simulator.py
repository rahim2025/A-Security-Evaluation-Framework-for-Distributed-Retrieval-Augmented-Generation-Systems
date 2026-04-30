import json
import random
import sys
import os
import traceback
from typing import List
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from jsonargparse import Namespace
from loguru import logger
import numpy as np
from tqdm import tqdm

from modules.exp_logger import ExpLogger
from modules.data_types import Datapoint, Testcase
from modules.rag_network import DRAGNetwork, CRAGNetwork, NoRAGNetwork
from modules.evaluator import QAEvaluator
from modules.options import parse_args

# new
from modules.attacks import (
    DataPoisoningAttack,
    KnowledgeBaseExtractionAttack,
    MembershipInferenceAttack,
)


def get_nested_value(data_dict: dict, dot_key_path: str):
    """
    Retrieves a nested value from a dictionary using a dot-separated key path.

    Args:
        data_dict: The dictionary to retrieve the value from.
        dot_key_path: A string representing the nested keys separated by dots (e.g., "key1.key2.key3").

    Returns:
        The value at the specified path in the dictionary.
    """
    keys = dot_key_path.split(".")
    value = data_dict
    for key in keys:
        value = value[key]
    return value


def _get_node_attack_config(cfg):
    security_cfg = getattr(cfg, 'security', None)
    return getattr(security_cfg, 'node_attack', None) if security_cfg else None


def _get_node_attack_value(cfg, key, default=None):
    node_attack_cfg = _get_node_attack_config(cfg)
    sentinel = object()
    if node_attack_cfg is not None:
        value = getattr(node_attack_cfg, key, sentinel)
        if value is not sentinel:
            return value
    legacy_key_map = {
        'enabled': 'enable_node_attack',
        'type': 'node_attack_type',
        'ratio': 'node_attack_ratio',
        'iterations': 'node_attack_iterations',
        'strategy': 'attack_strategy',
        'target_peers': 'target_peers',
    }
    return getattr(cfg.rag, legacy_key_map[key], default)


# ==============================================================================
# NODE AVAILABILITY ATTACK FUNCTIONS
# ==============================================================================

def run_node_availability_attack(cfg: Namespace, rag_network, dataset_type: str):
    """
    Run node availability attack to test system resilience.

    Args:
        cfg: Configuration namespace
        rag_network: The RAG network to attack
        dataset_type: Type of dataset being used

    Returns:
        Dictionary containing attack results
    """
    print("\n" + "="*70)
    print("NODE AVAILABILITY ATTACK - TESTING SYSTEM RESILIENCE")
    print("="*70)
    sys.stdout.flush()

    enable_attack = getattr(cfg.rag, 'enable_node_attack', False)
    attack_type = getattr(cfg.rag, 'node_attack_type', 'none')

    print(f"DEBUG: enable_node_attack = {enable_attack}")
    print(f"DEBUG: node_attack_type = {attack_type}")
    sys.stdout.flush()

    if not enable_attack:
        print("Node availability attack is DISABLED. Skipping...")
        sys.stdout.flush()
        return {}

    if attack_type == 'none':
        print("No node attack type specified. Skipping...")
        sys.stdout.flush()
        return {}

    # Import node attack modules
    drag_root = os.path.dirname(os.path.abspath(__file__))
    attacks_dir = os.path.join(drag_root, 'modules', 'attacks')
    if attacks_dir not in sys.path:
        sys.path.insert(0, attacks_dir)

    try:
        from node_attack import NodeAttack, evaluate_system_availability
        print("[✓] Successfully imported node attack modules")
        sys.stdout.flush()
    except ImportError as e:
        print(f"[✗] Could not import node attack modules: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return {}

    attack_ratio = getattr(cfg.rag, 'node_attack_ratio', 0.3)
    attack_iterations = getattr(cfg.rag, 'node_attack_iterations', 5)
    attack_strategy = getattr(cfg.rag, 'attack_strategy', 'random')
    target_peers = getattr(cfg.rag, 'target_peers', [])

    print(f"\nAttack Configuration:")
    print(f"  Attack Type: {attack_type}")
    print(f"  Attack Ratio: {attack_ratio} ({attack_ratio*100:.0f}% of nodes)")
    print(f"  Attack Strategy: {attack_strategy}")
    print(f"  Iterations: {attack_iterations}")
    print(f"  Dataset: {dataset_type}")
    if target_peers:
        print(f"  Target Peers: {target_peers}")
    sys.stdout.flush()

    if not hasattr(rag_network, 'peers'):
        print("ERROR: RAG network has no 'peers' attribute")
        sys.stdout.flush()
        return {'status': 'failed', 'error': 'No peers found in network'}

    peers = rag_network.peers
    if isinstance(peers, dict):
        nodes = list(peers.values())
        num_peers = len(peers)
    elif isinstance(peers, list):
        nodes = peers
        num_peers = len(peers)
    else:
        print(f"ERROR: Unknown peers format: {type(peers)}")
        sys.stdout.flush()
        return {'status': 'failed', 'error': 'Invalid peers format'}

    print(f"  Total Peers: {num_peers}\n")
    sys.stdout.flush()

    node_attack = NodeAttack(
        attack_type=attack_type,
        attack_ratio=attack_ratio,
        attack_iterations=attack_iterations,
        seed=cfg.rag.random_seed
    )

    results = {
        'attack_type': attack_type,
        'attack_ratio': attack_ratio,
        'attack_strategy': attack_strategy,
        'total_peers': num_peers,
        'iterations': [],
        'initial_availability': 100.0,
        'final_availability': 0.0
    }

    node_data = {
        'peer_ids': list(range(num_peers)),
        'initial_count': num_peers,
        'dataset_type': dataset_type,
        'target_peers': target_peers if target_peers else None
    }

    print("Starting attack simulation...")
    print("="*70)
    sys.stdout.flush()

    start_time = datetime.now()

    for iteration in range(attack_iterations):
        print(f"\n--- Iteration {iteration + 1}/{attack_iterations} ---")
        sys.stdout.flush()

        attack_result = node_attack.execute_attack(nodes, node_data, attack_strategy)
        availability_metrics = evaluate_system_availability(nodes, node_data)

        iteration_result = {
            'iteration': iteration + 1,
            'attack_result': attack_result,
            'availability_metrics': availability_metrics
        }
        results['iterations'].append(iteration_result)

        avail_pct = availability_metrics['availability_percentage']
        active = availability_metrics['active_nodes']
        print(f"  Availability: {avail_pct:.2f}% ({active}/{num_peers} nodes active)")

        if attack_type == 'byzantine':
            print(f"  Byzantine Nodes: {availability_metrics.get('byzantine_nodes', 0)}")
        elif attack_type == 'ddos':
            print(f"  DDoS Targets: {availability_metrics.get('ddos_targets', 0)}")
        elif attack_type == 'sybil':
            print(f"  Sybil Nodes: {availability_metrics.get('sybil_nodes', 0)}")

        sys.stdout.flush()

        confidence_threshold = getattr(cfg.rag, 'confidence_threshold', 0.5)
        if avail_pct < confidence_threshold * 100:
            print(f"  ⚠️  System availability below threshold ({confidence_threshold*100:.0f}%)")
            sys.stdout.flush()

    elapsed_time = (datetime.now() - start_time).total_seconds()

    final_metrics = evaluate_system_availability(nodes, node_data)
    results['final_availability'] = final_metrics['availability_percentage']
    results['execution_time'] = elapsed_time
    results['attack_summary'] = node_attack.get_attack_summary()

    print("\n" + "="*70)
    print("NODE AVAILABILITY ATTACK SUMMARY")
    print("="*70)
    print(f"Attack Type: {attack_type}")
    print(f"Total Iterations: {attack_iterations}")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"Initial Availability: {results['initial_availability']:.2f}%")
    print(f"Final Availability: {results['final_availability']:.2f}%")
    print(f"Degradation: {results['initial_availability'] - results['final_availability']:.2f}%")
    print("="*70)
    sys.stdout.flush()

    drag_root = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(drag_root, 'security_evaluation', 'results')
    os.makedirs(results_dir, exist_ok=True)

    result_file = os.path.join(
        results_dir,
        f"node_attack_{attack_type}_{dataset_type}_{int(datetime.now().timestamp())}.json"
    )

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[+] Node attack results saved to: {result_file}\n")
    sys.stdout.flush()

    return results


def apply_node_attack_damage(rag_network, node_attack_results: dict):
    print("=" * 70)
    print("APPLYING NODE ATTACK DAMAGE TO NETWORK")
    print("=" * 70)
    sys.stdout.flush()

    if not node_attack_results or 'attack_type' not in node_attack_results:
        print("WARNING: No node attack results found, skipping damage application")
        sys.stdout.flush()
        return

    attack_type = node_attack_results['attack_type']
    print(f"Attack type: {attack_type}")

    # Collect all disabled node indices from all iterations
    disabled_indices = set()
    for iteration_result in node_attack_results.get('iterations', []):
        attack_result = iteration_result.get('attack_result', {})
        for idx in attack_result.get('disabled_nodes', []):
            disabled_indices.add(idx)
        for idx in attack_result.get('removed_indices', []):
            disabled_indices.add(idx)

    if not hasattr(rag_network, 'peers'):
        print("WARNING: Network has no peers attribute")
        sys.stdout.flush()
        return

    # Physically set peers to None
    peers = rag_network.peers
    actually_disabled = 0
    for idx in disabled_indices:
        if isinstance(peers, list) and idx < len(peers):
            if peers[idx] is not None:
                peers[idx] = None
                actually_disabled += 1
        elif isinstance(peers, dict) and idx in peers:
            if peers[idx] is not None:
                peers[idx] = None
                actually_disabled += 1

    final_avail = node_attack_results.get('final_availability', 100.0)
    initial_avail = node_attack_results.get('initial_availability', 100.0)
    degradation = initial_avail - final_avail

    print(f"Disabled node indices: {sorted(disabled_indices)}")
    print(f"Actually disabled: {actually_disabled} peers set to None")
    print(f"Network availability degraded by {degradation:.2f}%")
    print("=" * 70)
    sys.stdout.flush()


def run_simulation(cfg: Namespace):
    # Init csv logger
    exp_logger = ExpLogger()
    logger.info(f"Experiment Log Directory: {exp_logger.experiment_dir}")
    config_logger = exp_logger.get_yaml_logger("config")
    metrics_logger = exp_logger.get_csv_logger("metrics")
    test_cases_logger = exp_logger.get_csv_logger("test_cases")

    # Save all config
    config_logger.log(cfg.as_dict())
    config_logger.save()

    # Load Huggingface dataset
    dataset = load_dataset(**cfg.data.load.as_dict())
    data_points: List[Datapoint] = []
    all_topics = set()

    task_type = cfg.data.task_type

    if cfg.rag.test_mode:
        dataset = dataset.select(range(20))
    else:
        if cfg.data.num_samples is not None:
            dataset = dataset.shuffle(seed=cfg.rag.random_seed).take(min(cfg.data.num_samples, len(dataset)))
        else:
            dataset = dataset.shuffle(seed=cfg.rag.random_seed)

    # Prepare data points
    for item in dataset:
        topic = get_nested_value(item, cfg.data.topic_path)
        question = get_nested_value(item, cfg.data.question_path)
        answer = get_nested_value(item, cfg.data.answer_path)
        if task_type == "mcqa":
            choices = get_nested_value(item, cfg.data.choices_path)
            connection_term = " Select the best answer from the following candidates, replying with 1, 2, 3, or 4: "
            question = str(question) + connection_term + str(choices)

        data_point = Datapoint(topic=str(topic), question=str(question), answer=str(answer))
        all_topics.add(str(topic))
        data_points.append(data_point)

    if cfg.rag.network_type == "DRAG":
        filtered_data_points = data_points
    elif cfg.rag.network_type == "CRAG":
        num_topics_to_keep = int(len(all_topics) * (1.0 - cfg.rag.filter_out_topic_ratio))
        filtered_topics = random.sample(list(all_topics), k=num_topics_to_keep)
        filtered_data_points = [
            dp for dp in data_points if dp.topic in filtered_topics
        ]
        num_datapoints_to_keep = int(len(filtered_data_points) * (1.0 - cfg.rag.filter_out_qa_ratio))
        filtered_data_points = random.sample(filtered_data_points, k=num_datapoints_to_keep)
    elif cfg.rag.network_type == "NoRAG":
        filtered_data_points = []
    else:
        raise ValueError(f"Unknown network type: {cfg.rag.network_type}")

    # Initialize DRAG parameters
    query_confidence_threshold = cfg.rag.query_confidence_threshold
    num_query_neighbor = min(cfg.rag.num_query_neighbor, cfg.rag.num_peers - 1)
    query_ttl = cfg.rag.query_ttl

    # Initialize RAG network with peers and knowledges
    if cfg.rag.network_type == "DRAG":
        rag_net = DRAGNetwork(cfg.rag.num_peers, cfg.rag.num_peer_attachments, cfg.llm.base_url, cfg.llm.name,
                              cfg.llm.num_ctx, cfg.rag.random_seed)
        rag_net.init_knowledge(filtered_data_points, replication_factor=cfg.rag.replication_factor)
    elif cfg.rag.network_type == "CRAG":
        rag_net = CRAGNetwork(cfg.llm.base_url, cfg.llm.name, cfg.llm.num_ctx, cfg.rag.random_seed)
        rag_net.init_knowledge(filtered_data_points)
    elif cfg.rag.network_type == "NoRAG":
        rag_net = NoRAGNetwork(cfg.llm.base_url, cfg.llm.name, cfg.llm.num_ctx, cfg.rag.random_seed)
        rag_net.init_knowledge(filtered_data_points)
    else:
        raise ValueError(f"Unknown network type: {cfg.rag.network_type}")

    # === ATTACK SIMULATION ===
    attack_results = None
    mia_results = None
    if cfg.security.enable_attack:
        logger.info("=" * 50)
        logger.info("ATTACK SIMULATION ENABLED")
        logger.info("=" * 50)

        attack = DataPoisoningAttack(
            poisoning_ratio=cfg.security.poisoning_ratio,
            attack_strategy=cfg.security.attack_strategy,
            poison_type=cfg.security.poison_type,
            target_peer_ids=cfg.security.get('target_peer_ids', []),
            amplification_factor=cfg.security.get('amplification_factor', 3),
            question_variants=cfg.security.get('question_variants', 2)
        )

        attack_results = attack.execute(rag_net, filtered_data_points)

        attack_logger = exp_logger.get_yaml_logger("attack_config")
        attack_logger.log(attack_results)
        attack_logger.save()

        logger.info(f"Attack executed: {attack_results['num_poisoned_peers']} peers poisoned")

    if cfg.security.get('enable_membership_inference', False):
        logger.info("=" * 50)
        logger.info("MEMBERSHIP INFERENCE ATTACK ENABLED")
        logger.info("=" * 50)
        if cfg.rag.network_type != "DRAG":
            logger.warning("Membership inference attack is only supported for DRAG networks. Skipping.")
        else:
            mia_attack = MembershipInferenceAttack(
                inference_method=cfg.security.get('mia_inference_method', 'confidence_based'),
                test_size=cfg.security.get('mia_test_size', 0.3),
                threshold_percentile=cfg.security.get('mia_threshold_percentile', 50),
                random_seed=cfg.security.get('mia_random_seed', 42)
            )
            mia_results = mia_attack.execute(rag_net, filtered_data_points)
            mia_logger = exp_logger.get_yaml_logger("mia_config")
            mia_logger.log(mia_results)
            mia_logger.save()
            logger.info(
                f"MIA executed: Accuracy={mia_results['attack_accuracy']:.2%}, "
                f"Privacy Risk={mia_results['privacy_risk']}"
            )

    # Execute knowledge base extraction attack if enabled
    if cfg.security.enable_extraction:
        logger.info("\n" + "=" * 80)
        logger.info("KNOWLEDGE BASE EXTRACTION ATTACK")
        logger.info("=" * 80)

        extraction_attack = KnowledgeBaseExtractionAttack(
            attacker_peer_id=cfg.security.extraction_attacker_peer,
            queries_per_topic=cfg.security.extraction_queries_per_topic,
            attack_type=cfg.security.extraction_attack_type,
            use_topic_inference=cfg.security.extraction_use_topic_inference,
            use_dataset_questions=cfg.security.extraction_use_dataset_questions
        )

        extraction_results = extraction_attack.execute(rag_net, filtered_data_points)

        logger.info("\n" + "-" * 80)
        logger.info("EXTRACTION ATTACK SUMMARY:")
        logger.info(f"  Attacker Peer: {extraction_results['attacker_peer_id']}")
        logger.info(f"  Queries Sent: {extraction_results['total_queries_sent']}")
        logger.info(f"  Successful Queries: {extraction_results['successful_queries']}")
        logger.info(f"  Unique Extracted: {extraction_results['unique_datapoints_extracted']}")
        logger.info(f"  Total Network KB Size: {extraction_results.get('total_network_datapoints', extraction_results.get('total_target_datapoints', 0))}")
        logger.info(f"  Extraction Rate: {extraction_results['extraction_rate']:.2%}")
        logger.info(f"  Query Efficiency: {extraction_results['query_efficiency']:.2%}")
        logger.info(f"  Topic Coverage: {extraction_results['topic_coverage']:.2%}")
        logger.info("-" * 80 + "\n")

        extraction_metrics_logger = exp_logger.get_csv_logger("extraction_metrics")
        extraction_metrics_logger.log({
            "extraction_rate": extraction_results['extraction_rate'],
            "extraction_queries_sent": extraction_results['total_queries_sent'],
            "extraction_unique_extracted": extraction_results['unique_datapoints_extracted'],
            "extraction_query_efficiency": extraction_results['query_efficiency'],
            "extraction_topic_coverage": extraction_results['topic_coverage'],
            "crr": extraction_results.get('crr', 0.0),
            "avg_ss": extraction_results.get('avg_ss', 0.0),
            "avg_eed": extraction_results.get('avg_eed', 1.0),
            "recovered_chunks": extraction_results.get('recovered_chunks', 0),
            "total_chunks": extraction_results.get('total_chunks', 0),
            "exact_matches": extraction_results.get('exact_matches', 0),
            "semantic_matches": extraction_results.get('semantic_matches', 0),
            "edit_distance_matches": extraction_results.get('edit_distance_matches', 0)
        })
        extraction_metrics_logger.save()

        extraction_output_path = f"{exp_logger.experiment_dir}/extraction_results.json"
        with open(extraction_output_path, 'w') as f:
            json.dump(extraction_results, f, indent=2)
        logger.info(f"Detailed extraction results saved to: {extraction_output_path}")

        if cfg.security.enable_extraction:
            logger.info("=" * 80)
            logger.info("EXTRACTION ATTACK COMPLETE - Skipping normal evaluation")
            logger.info("Use a separate config without extraction for performance evaluation")
            logger.info("=" * 80)
            return  # Exit early

    # =======================================================================
    # NODE AVAILABILITY ATTACK (PHASE 1: baseline → attack → post-attack)
    # =======================================================================
    eval_results = {}  # BUG FIX 1: always defined — prevents UnboundLocalError at bottom

    enable_node_attack = getattr(cfg.rag, 'enable_node_attack', False)
    if enable_node_attack:
        # Determine dataset type from config file path
        config_file_path = getattr(cfg, 'config', None)
        if config_file_path:
            config_file_str = str(config_file_path).lower()
            if 'medical' in config_file_str:
                dataset_type = 'medical'
            elif 'mmlu' in config_file_str:
                dataset_type = 'mmlu'
            elif 'news' in config_file_str:
                dataset_type = 'news'
            else:
                dataset_type = 'unknown'
        else:
            dataset_type = 'unknown'

        logger.info("=" * 50)
        logger.info("NODE AVAILABILITY ATTACK ENABLED")
        logger.info("=" * 50)

        # --- PHASE 1: Baseline evaluation (before node attack) ---
        logger.info("PHASE 1: BASELINE EVALUATION (BEFORE NODE ATTACK)")
        qa_evaluator_baseline = QAEvaluator()
        baseline_successful = 0
        baseline_failed = 0

        for idx, data_point in enumerate(tqdm(data_points, desc=f"Baseline evaluation on {len(data_points)} test case(s)")):
            try:
                if cfg.rag.network_type == "DRAG":
                    if cfg.rag.search_algorithm == "TARW":
                        rag_answer = rag_net.topic_query(
                            data_point.question,
                            num_query_neighbor=num_query_neighbor,
                            query_confidence_threshold=query_confidence_threshold,
                            max_ttl=query_ttl
                        )
                    elif cfg.rag.search_algorithm == "RW":
                        rag_answer = rag_net.random_walk_query(
                            data_point.question,
                            query_confidence_threshold=query_confidence_threshold,
                            max_ttl=query_ttl
                        )
                    elif cfg.rag.search_algorithm == "FL":
                        rag_answer = rag_net.flooding_query(
                            data_point.question,
                            query_confidence_threshold=query_confidence_threshold,
                            max_ttl=query_ttl
                        )
                    else:
                        raise ValueError(f"Unknown search algorithm: {cfg.rag.search_algorithm}")
                elif cfg.rag.network_type == "CRAG":
                    rag_answer = rag_net.query(
                        data_point.question,
                        query_confidence_threshold=query_confidence_threshold
                    )
                elif cfg.rag.network_type == "NoRAG":
                    rag_answer = rag_net.query(data_point.question)
                else:
                    raise ValueError(f"Unknown network type: {cfg.rag.network_type}")

                test_case = Testcase(
                    question=data_point.question,
                    expected_output=data_point.answer,
                    actual_output=rag_answer.answer,
                    relevant_knowledge=rag_answer.relevant_knowledge,
                    relevant_score=rag_answer.relevant_score,
                    num_hops=rag_answer.num_hops,
                    num_messages=rag_answer.num_messages,
                    is_query_hit=rag_answer.is_query_hit
                )
                baseline_successful += 1

            except AttributeError as e:
                if "'NoneType' object has no attribute" in str(e):
                    logger.warning(f"Baseline query {idx} hit disabled node: {str(e)}")
                    test_case = Testcase(
                        question=data_point.question,
                        expected_output=data_point.answer,
                        actual_output="QUERY_FAILED_NODE_UNAVAILABLE",
                        relevant_knowledge="",
                        relevant_score=0.0,
                        num_hops=0,
                        num_messages=0,
                        is_query_hit=False
                    )
                    baseline_failed += 1
                else:
                    raise

            test_case_dict = test_case.model_dump()
            test_case_dict['evaluation_phase'] = 'baseline'
            test_cases_logger.log(test_case_dict)
            qa_evaluator_baseline.add(test_case)

            if idx % cfg.rag.log_every_n_steps == 0:
                test_cases_logger.save()

        test_cases_logger.save()
        baseline_results = qa_evaluator_baseline.get_results()
        baseline_results['evaluation_phase'] = 'baseline'
        baseline_results['successful_queries'] = baseline_successful
        baseline_results['failed_queries'] = baseline_failed
        baseline_results['query_failure_rate'] = baseline_failed / len(data_points) if data_points else 0
        metrics_logger.log(baseline_results)
        metrics_logger.save()
        logger.info(f"Baseline Results: {json.dumps(baseline_results)}")

        # --- PHASE 2: Execute node availability attack ---
        node_attack_results = run_node_availability_attack(cfg, rag_net, dataset_type)

        # --- PHASE 3: Apply node attack damage ---
        if node_attack_results:
            apply_node_attack_damage(rag_net, node_attack_results)

        # --- PHASE 4: Post-attack evaluation ---
        logger.info("PHASE 4: POST-ATTACK EVALUATION")
        qa_evaluator_post = QAEvaluator()
        failed_queries = 0
        successful_queries = 0  # BUG FIX 2: track successful queries — fixes N/A in summary

        for idx, data_point in enumerate(tqdm(data_points, desc=f"Post-attack evaluation on {len(data_points)} test case(s)")):
            try:
                if cfg.rag.network_type == "DRAG":
                    if cfg.rag.search_algorithm == "TARW":
                        rag_answer = rag_net.topic_query(
                            data_point.question,
                            num_query_neighbor=num_query_neighbor,
                            query_confidence_threshold=query_confidence_threshold,
                            max_ttl=query_ttl
                        )
                    elif cfg.rag.search_algorithm == "RW":
                        rag_answer = rag_net.random_walk_query(
                            data_point.question,
                            query_confidence_threshold=query_confidence_threshold,
                            max_ttl=query_ttl
                        )
                    elif cfg.rag.search_algorithm == "FL":
                        rag_answer = rag_net.flooding_query(
                            data_point.question,
                            query_confidence_threshold=query_confidence_threshold,
                            max_ttl=query_ttl
                        )
                    else:
                        raise ValueError(f"Unknown search algorithm: {cfg.rag.search_algorithm}")
                elif cfg.rag.network_type == "CRAG":
                    rag_answer = rag_net.query(
                        data_point.question,
                        query_confidence_threshold=query_confidence_threshold
                    )
                elif cfg.rag.network_type == "NoRAG":
                    rag_answer = rag_net.query(data_point.question)
                else:
                    raise ValueError(f"Unknown network type: {cfg.rag.network_type}")

                test_case = Testcase(
                    question=data_point.question,
                    expected_output=data_point.answer,
                    actual_output=rag_answer.answer,
                    relevant_knowledge=rag_answer.relevant_knowledge,
                    relevant_score=rag_answer.relevant_score,
                    num_hops=rag_answer.num_hops,
                    num_messages=rag_answer.num_messages,
                    is_query_hit=rag_answer.is_query_hit
                )

                # Check if this is a Byzantine response
                if rag_answer.answer.startswith("INCORRECT_BYZANTINE_RESPONSE_"):
                    logger.warning(f"Post-attack query {idx} received Byzantine response: {rag_answer.answer}")
                    test_case.actual_output = "BYZANTINE_INCORRECT_ANSWER"
                    test_case.relevant_score = 0.0
                    test_case.is_query_hit = False

                successful_queries += 1  # BUG FIX 2: count successful queries

            except AttributeError as e:
                if "'NoneType' object has no attribute" in str(e):
                    logger.warning(f"Post-attack query {idx} hit disabled node: {str(e)}")
                    test_case = Testcase(
                        question=data_point.question,
                        expected_output=data_point.answer,
                        actual_output="QUERY_FAILED_NODE_UNAVAILABLE",
                        relevant_knowledge="",
                        relevant_score=0.0,
                        num_hops=0,
                        num_messages=0,
                        is_query_hit=False
                    )
                    failed_queries += 1
                else:
                    raise

            test_case_dict = test_case.model_dump()
            test_case_dict['evaluation_phase'] = 'post_attack'
            test_cases_logger.log(test_case_dict)
            qa_evaluator_post.add(test_case)

            if idx % cfg.rag.log_every_n_steps == 0:
                test_cases_logger.save()

        test_cases_logger.save()
        post_attack_results = qa_evaluator_post.get_results()
        post_attack_results['evaluation_phase'] = 'post_attack'
        post_attack_results['failed_queries'] = failed_queries
        post_attack_results['successful_queries'] = successful_queries  # BUG FIX 2: store count
        post_attack_results['query_failure_rate'] = failed_queries / len(data_points) if len(data_points) > 0 else 0
        metrics_logger.log(post_attack_results)
        metrics_logger.save()
        logger.info(f"Post-Attack Results: {json.dumps(post_attack_results)}")

        # --- PHASE 5: Comparison and final report ---
        logger.info("=" * 50)
        logger.info("FINAL COMPARISON: BASELINE vs POST-ATTACK")
        logger.info("=" * 50)

        comparison = {}
        for metric in ['exact_match', 'precision', 'recall', 'f1', 'avg_query_hit', 'query_failure_rate']:
            baseline_val = baseline_results.get(metric, 0)
            post_attack_val = post_attack_results.get(metric, 0)

            if metric == 'query_failure_rate':
                degradation = post_attack_val - baseline_val
            else:
                degradation = baseline_val - post_attack_val
            degradation_pct = (degradation / (baseline_val + 0.0001)) * 100

            comparison[metric] = {
                'baseline': baseline_val,
                'post_attack': post_attack_val,
                'degradation': degradation,
                'degradation_pct': degradation_pct
            }

        # ================================================================
        # PRINT CLEAN TERMINAL SUMMARY
        # ================================================================
        attack_type  = getattr(cfg.rag, 'node_attack_type', 'none')
        attack_ratio = getattr(cfg.rag, 'node_attack_ratio', 0.0)
        total_queries = len(data_points)

        b_success   = baseline_results.get('successful_queries', total_queries)
        b_failed    = baseline_results.get('failed_queries', 0)
        b_fail_rate = baseline_results.get('query_failure_rate', 0)

        p_success   = post_attack_results.get('successful_queries', 0)
        p_failed    = post_attack_results.get('failed_queries', 0)
        p_fail_rate = post_attack_results.get('query_failure_rate', 0)

        print("\n" + "=" * 70)
        print(f"  NODE ATTACK RESULTS SUMMARY")
        print(f"  Attack Type  : {attack_type.upper()}")
        print(f"  Attack Ratio : {attack_ratio} ({attack_ratio*100:.0f}% of nodes removed)")
        print("=" * 70)

        print(f"\n  QUERY AVAILABILITY")
        print(f"  {'─'*50}")
        print(f"  Baseline  → Successful: {b_success}/{total_queries}  |  Failed: {b_failed}/{total_queries}  |  Failure Rate: {b_fail_rate:.0%}")
        print(f"  Attacked  → Successful: {p_success}/{total_queries}  |  Failed: {p_failed}/{total_queries}  |  Failure Rate: {p_fail_rate:.0%}")

        print(f"\n  {'METRIC':<25} {'BASELINE':>10} {'POST-ATTACK':>12} {'DEGRADATION':>14} {'IMPACT %':>10}")
        print(f"  {'─'*25} {'─'*10} {'─'*12} {'─'*14} {'─'*10}")

        for metric in ['exact_match', 'precision', 'recall', 'f1', 'avg_query_hit']:
            c = comparison[metric]
            bar_len = int(abs(c['degradation_pct']) / 5)
            bar = '█' * min(bar_len, 15)
            print(f"  {metric.upper():<25} {c['baseline']:>10.4f} {c['post_attack']:>12.4f} {c['degradation']:>+14.4f} {c['degradation_pct']:>9.2f}%  {bar}")

        print(f"\n  {'─'*70}")
        qfr = comparison['query_failure_rate']
        print(f"  {'QUERY_FAILURE_RATE':<25} {qfr['baseline']:>10.4f} {qfr['post_attack']:>12.4f} {qfr['degradation']:>+14.4f}")
        print("=" * 70)
        print(f"  ✅ Node attack simulation complete.")
        print("=" * 70 + "\n")
        sys.stdout.flush()

        # Save comparison JSON
        comparison_file = os.path.join(exp_logger.experiment_dir, "node_attack_impact_comparison.json")
        with open(comparison_file, 'w') as f:
            json.dump({
                'baseline_results': baseline_results,
                'node_attack_results': node_attack_results,
                'post_attack_results': post_attack_results,
                'comparison': comparison,
                'attack_type': attack_type
            }, f, indent=2)
        logger.info(f"Comparison report saved to: {comparison_file}")

        eval_results = post_attack_results  # BUG FIX 3: set for downstream defense/attack logging

    else:
        # =======================================================================
        # NORMAL EVALUATION (no node attack)
        # =======================================================================
        qa_evaluator = QAEvaluator()

        for idx, data_point in enumerate(tqdm(data_points, desc=f"Inferencing on {len(data_points)} test case(s)")):
            if cfg.rag.network_type == "DRAG":
                if cfg.rag.search_algorithm == "TARW":
                    rag_answer = rag_net.topic_query(
                        data_point.question,
                        num_query_neighbor=num_query_neighbor,
                        query_confidence_threshold=query_confidence_threshold,
                        max_ttl=query_ttl
                    )
                elif cfg.rag.search_algorithm == "RW":
                    rag_answer = rag_net.random_walk_query(
                        data_point.question,
                        query_confidence_threshold=query_confidence_threshold,
                        max_ttl=query_ttl
                    )
                elif cfg.rag.search_algorithm == "FL":
                    rag_answer = rag_net.flooding_query(
                        data_point.question,
                        query_confidence_threshold=query_confidence_threshold,
                        max_ttl=query_ttl
                    )
                else:
                    raise ValueError(f"Unknown search algorithm: {cfg.rag.search_algorithm}")
            elif cfg.rag.network_type == "CRAG":
                rag_answer = rag_net.query(
                    data_point.question,
                    query_confidence_threshold=query_confidence_threshold
                )
            elif cfg.rag.network_type == "NoRAG":
                rag_answer = rag_net.query(data_point.question)
            else:
                raise ValueError(f"Unknown network type: {cfg.rag.network_type}")

            test_case = Testcase(
                question=data_point.question,
                expected_output=data_point.answer,
                actual_output=rag_answer.answer,
                relevant_knowledge=rag_answer.relevant_knowledge,
                relevant_score=rag_answer.relevant_score,
                num_hops=rag_answer.num_hops,
                num_messages=rag_answer.num_messages,
                is_query_hit=rag_answer.is_query_hit
            )
            test_cases_logger.log(test_case.model_dump())
            qa_evaluator.add(test_case)

            if idx % cfg.rag.log_every_n_steps == 0:
                test_cases_logger.save()
                eval_results = qa_evaluator.get_results()
                metrics_logger.log(eval_results)
                metrics_logger.save()

        test_cases_logger.save()
        eval_results = qa_evaluator.get_results()
        metrics_logger.log(eval_results)
        metrics_logger.save()

    # Log defense statistics if enabled
    if cfg.rag.network_type == "DRAG" and rag_net.defense_enabled and rag_net.defense_mechanism:
        defense_stats = rag_net.defense_mechanism.get_stats()
        logger.info("=" * 50)
        logger.info("DEFENSE STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Defense: {defense_stats['name']}")
        logger.info(f"Total Validations: {defense_stats['total_validations']}")
        logger.info(f"Blocked Answers: {defense_stats['blocked_answers']}")
        logger.info(f"Passed Answers: {defense_stats['passed_answers']}")
        logger.info(f"Block Rate: {defense_stats['block_rate']:.2%}")
        logger.info(f"Avg Confidence: {defense_stats['avg_confidence']:.3f}")

        eval_results['defense_stats'] = defense_stats

        defense_logger = exp_logger.get_yaml_logger("defense_stats")
        defense_logger.log(defense_stats)
        defense_logger.save()

    # Evaluate attack success if data poisoning attack was performed
    if attack_results is not None:
        logger.info("=" * 50)
        logger.info("EVALUATING ATTACK SUCCESS")
        logger.info("=" * 50)

        attack_eval = {
            "attack_config": attack_results,
            "attacked_metrics": eval_results,
            "performance_impact": {
                "f1_score": eval_results.get("f1", 0.0),
                "exact_match": eval_results.get("exact_match", 0.0),
                "semantic_similarity": eval_results.get("semantic_similarity", 0.0)
            }
        }

        attack_eval_logger = exp_logger.get_yaml_logger("attack_evaluation")
        attack_eval_logger.log(attack_eval)
        attack_eval_logger.save()

        logger.info(f"\nAttack Evaluation:\n{json.dumps(attack_eval, indent=2)}\n")

    if mia_results is not None:
        logger.info("=" * 50)
        logger.info("EVALUATING MEMBERSHIP INFERENCE ATTACK SUCCESS")
        logger.info("=" * 50)
        mia_eval = {
            "attack_results": mia_results,
            "interpretation": {
                "attack_success": mia_results["attack_accuracy"] > 0.5,
                "privacy_leak_detected": mia_results["attack_accuracy"] > 0.6,
                "attack_accuracy": f"{mia_results['attack_accuracy']:.2%}",
                "privacy_risk": mia_results["privacy_risk"],
            },
        }
        mia_eval_logger = exp_logger.get_yaml_logger("mia_evaluation")
        mia_eval_logger.log(mia_eval)
        mia_eval_logger.save()
        logger.info(f"\nMIA Evaluation:\n{json.dumps(mia_eval, indent=2)}\n")

    logger.info(f"\nFinal Evaluation Results:\n{json.dumps(eval_results)}\n")


def main():
    # parse arguments
    cfg = parse_args()

    # Initialize random seeds
    random.seed(cfg.rag.random_seed)
    np.random.seed(cfg.rag.random_seed)

    # Changing the level of the logger
    logger.remove()  # Remove default handler.
    logger.add(sys.stderr, level=cfg.log_level)

    # run evaluation
    run_simulation(cfg)


if __name__ == "__main__":
    main()