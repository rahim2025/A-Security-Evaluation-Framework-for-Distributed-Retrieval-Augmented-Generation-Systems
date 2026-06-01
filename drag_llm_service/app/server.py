"""API server for the LLM service that orchestrates retrieval, reranking, and sentence importance analysis.
"""
import json
import sys
import os
import yaml
import re
import random
from typing import Dict, List, Optional
import requests
import numpy as np
import logging
from pathlib import Path

from flask import Flask, request, jsonify, Response
from queue import Queue
from eth_account import Account

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add drag_python_client to path (both for local and Docker contexts)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'drag_python_client')))
sys.path.insert(0, '/app/drag_python_client')

from src.retriever.reranker import Reranker
from src.utils.helper_functions import _normalize_for_match, _norm_join
from src.mc_shap.base import ModelBase, LocalModel, OpenAIModel
from src.mc_shap.mc_shap import TokenSHAP, StringSplitter
from src.mc_shap.base import TfidfTextVectorizer
from src.rora.rora import RORAModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import drag contract library
try:
    from drag_python_client import DragScoresClient
except ImportError as e:
    logger.warning(f"drag_python_client not available: {str(e)}. Blockchain features will be disabled.")
    DragScoresClient = None

app = Flask(__name__)

# Global components
model = None
mc_shap = None
rora = None
reranker = None
sentence_importance_config = None
config = None
data_source_configs = None
blockchain_client = None
retrieval_config = None
splitter = None
vectorizer = None

def _replace_localhost_in_config(config: Dict, replacement: str) -> Dict:
    """Recursively replace localhost with replacement in URL strings."""
    if isinstance(config, dict):
        return {k: _replace_localhost_in_config(v, replacement) for k, v in config.items()}
    elif isinstance(config, list):
        return [_replace_localhost_in_config(item, replacement) for item in config]
    elif isinstance(config, str) and ('http://localhost' in config or 'https://localhost' in config):
        return config.replace('localhost', replacement)
    else:
        return config


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Replace localhost if REPLACE_LOCALHOST_WITH environment variable is set
    replacement = os.getenv('REPLACE_LOCALHOST_WITH')
    if replacement:
        config = _replace_localhost_in_config(config, replacement)
    
    return config


def initialize_model(model_config: Dict) -> None:
    """Initialize the model based on configuration."""
    global model
    
    logger.info(f"Initializing model of type: {model_config['type']}")
    
    if model_config['type'] == "local":
        model = LocalModel(
            model_name=model_config['local_model_name'],
            system_setting=model_config['system_prompt'],
            temperature=model_config['temperature'],
            seed=model_config['seed']
        )
    elif model_config['type'] == "openai":
        api_key = model_config.get('openai_api_key') or os.getenv('OPENAI_API_KEY', '')
        model = OpenAIModel(
            model_name=model_config['openai_model_name'],
            api_key=api_key,
            base_url=model_config.get('openai_base_url'),
            system_prompt=model_config['system_prompt']
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")

def initialize_sentence_importance(sentence_importance_config: Dict) -> None:
    """Initialize sentence importance evaluation method based on config.
    
    Args:
        sentence_importance_config: Dictionary containing:
            - sentence_importance_method: "mc_shap" or "rora"
            - mc_shap: {split_pattern, max_tokens, aggregate_type}
            - rora: {model_dir, rationale_format, device, aggregate_type}
    """
    global mc_shap, rora, splitter, vectorizer
    
    # Read the method name from config
    method = sentence_importance_config.get('sentence_importance_method', 'mc_shap')
    
    if method == 'mc_shap':
        # Read mc_shap parameters from config
        mc_shap_config = sentence_importance_config.get('mc_shap', {})
        split_pattern = mc_shap_config.get('split_pattern', r'\.\s+|\n')
        max_tokens = mc_shap_config.get('max_tokens', 900)
        aggregate_type = mc_shap_config.get('aggregate_type', 'mean')
        
        # Initialize splitter and mc_shap
        splitter = StringSplitter(split_pattern=split_pattern)
        vectorizer = TfidfTextVectorizer()
        # mc_shap = TokenSHAP(model, splitter, vectorizer, debug=False)
        # logger.info(f"MC-Shapley initialized successfully (split_pattern={split_pattern}, max_tokens={max_tokens}, aggregate_type={aggregate_type})")
        
        # Store aggregate_type in retrieval_config for later use
        retrieval_config['aggregate_type'] = aggregate_type
        retrieval_config['max_tokens'] = max_tokens
        
    elif method == 'rora':
        # Read rora parameters from config
        rora_config = sentence_importance_config.get('rora', {})
        model_dir = rora_config.get('model_dir')
        rationale_format = rora_config.get('rationale_format', 'g')
        device = rora_config.get('device', 'cuda:0')
        aggregate_type = rora_config.get('aggregate_type', 'max')
        
        if not model_dir:
            raise ValueError("RORA model_dir must be specified in config")
        
        # Initialize rora
        rora = RORAModel(
            model_dir=model_dir,
            rationale_format=rationale_format,
            device=device
        )
        rora.load_model()
        logger.info(f"RORA initialized successfully (model_dir={model_dir}, rationale_format={rationale_format}, device={device}, aggregate_type={aggregate_type})")
        
        # Store aggregate_type in retrieval_config for later use
        retrieval_config['aggregate_type'] = aggregate_type
        
    else:
        raise ValueError(f"Unknown sentence importance method: {method}. Must be 'mc_shap' or 'rora'")

def query_data_sources(query: str, n_contexts: int, selected_sources: Optional[Dict[str, tuple]] = None) -> List[Dict]:
    """Query all configured data sources and return candidates.
    
    Args:
        query: Search query string
        n_contexts: Number of contexts to retrieve per source
        selected_sources: Optional dict mapping source_id to (usefulness, reliability) tuple
                         If provided, will be sent to data sources for validation and signing
    
    Returns:
        List of candidate dictionaries with id, text, score, and meta
    """
    candidates: List[Dict] = []
    
    for src_config in data_source_configs:
        try:
            url = f"{src_config['url']}/query"
            payload = {"query": query, "k": n_contexts}
            
            # Include selected_sources if provided
            if selected_sources:
                payload["selected_sources"] = selected_sources
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                response_data = response.json()
                results = response_data.get('results', [])
                
                for result in results:
                    candidates.append({
                        "id": f"{src_config['name']}::{result['id']}",
                        "text": result['text'],
                        "score": result['score'],
                        "meta": {**(result.get('meta', {})), "source": src_config['name']}
                    })
                
                # Log signature if present (for debugging)
                if 'signature' in response_data:
                    logger.debug(f"Received signature from {src_config['name']}: {response_data['signature'][:20]}...")
            else:
                logger.warning(f"Failed to query {src_config['name']}: {response.status_code}")
                if response.text:
                    logger.warning(f"Error response: {response.text[:200]}")
        except Exception as e:
            logger.error(f"Error querying {src_config['name']}: {str(e)}")
    
    return candidates


def get_scores_from_blockchain(source_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Retrieve usefulness and reliability scores from blockchain contract."""
    scores = {}
    try:
        if blockchain_client:
            # Get scores in batch
            returned_ids, reliability_scores, usefulness_scores = blockchain_client.get_scores_batch(source_names)
            
            # Build scores dictionary
            for i, source_id in enumerate(returned_ids):
                scores[source_id] = {
                    'reliability': float(reliability_scores[i]),
                    'usefulness': float(usefulness_scores[i])
                }
        else:
            # Fallback: return default scores if blockchain client not available
            logger.warning("Blockchain client not available, using default scores")
            for source_id in source_names:
                scores[source_id] = {
                    'reliability': 0.0,
                    'usefulness': 0.0
                }
    except Exception as e:
        logger.error(f"Error retrieving scores from blockchain: {str(e)}")
        # Fallback: return default scores on error
        for source_id in source_names:
            scores[source_id] = {
                'reliability': 0.0,
                'usefulness': 0.0
            }
    
    return scores


def sample_data_sources(sample_with_usefulness: bool, n_retrievers: int) -> List[str]:
    """Sample data sources based on usefulness scores from blockchain."""
    source_names = [src['name'] for src in data_source_configs]
    
    if sample_with_usefulness and n_retrievers < len(source_names):
        # Get usefulness scores from blockchain
        scores = get_scores_from_blockchain(source_names)
        # Use usefulness scores as weights (convert to positive values for sampling)
        weights = [max(0.0, scores.get(name, {}).get('usefulness', 0.0)) for name in source_names]
        
        # If all weights are zero, use uniform sampling
        if sum(weights) == 0:
            weights = [1.0] * len(source_names)
        
        sampled = random.choices(source_names, weights=weights, k=n_retrievers)
    else:
        # Use all sources
        sampled = source_names
    
    return sampled


def analyze_with_mc_shapley(prompt: str, max_tokens: int = 900, progress_queue: Optional[Queue] = None) -> Optional[Dict]:
    """Perform MC-Shapley analysis on the prompt.
    
    Args:
        prompt: Text prompt to analyze
        max_tokens: Maximum tokens to analyze
        progress_queue: Optional Queue for progress updates (for SSE streaming)
    """
    # Check prompt length
    if len(splitter.split(prompt)) > max_tokens:
        logger.warning(f"Prompt too long ({len(splitter.split(prompt))} tokens), skipping SHAP analysis")
        return None
    
    try:
        # If progress_queue is provided, create a new TokenSHAP instance with it
        if progress_queue is not None:
            mc_shap = TokenSHAP(model, splitter, vectorizer, debug=False, progress_queue=progress_queue)
            df = mc_shap.analyze(prompt, sampling_ratio=0.0, print_highlight_text=False)
            return {
                'response': mc_shap.baseline_text,
                'importance': mc_shap.shapley_values
            }
        else:
            mc_shap = TokenSHAP(model, splitter, vectorizer, debug=False)
            df = mc_shap.analyze(prompt, sampling_ratio=0.0, print_highlight_text=False)
            return {
                'response': mc_shap.baseline_text,
                'importance': mc_shap.shapley_values
            }
    except Exception as e:
        logger.error(f"Error in MC-Shapley analysis: {str(e)}")
        return None

def analyze_with_rora(question: str, answer_list: List[str], context_blocks: List[str]) -> Dict:
    """Analyze the sentence importance using RORA.

    Args:
        question: The question to answer
        answer: The ground-truth answer to the question
        context_blocks: the retrieved documents that are used to answer the question
    """
    try:
        #TODO: add progress_queue support
        context = "".join(context_blocks)
        prompt = context + "Question: " + question + "\n\nAnswer: "
        response = model(prompt)
        model.restart()

        # Format answers to be unambiguous even if individual answers contain commas
        safe_answers = [f'[{i}] {json.dumps(a)}' for i, a in enumerate(answer_list, start=1)]
        answer = "; ".join(safe_answers)
        binary_question = (
            f"{question}. Is at least one of the following answers correct: {answer}?"
        )   

        importance: Dict[str, float] = {}
        for context  in context_blocks:
            sentences = [s.strip() for s in re.split(r"(?<=[\.!?])\s+", context) if s.strip()]
            for sentence in sentences:
                rationale = sentence if sentence.endswith(('.', '!', '?')) else sentence + '.'
                score = float(rora.evaluate(rationale, binary_question, True))
                importance[sentence] = score

        return {
            'response': response,
            'importance': importance
        }

    except Exception as e:
        logger.error(f"Error in RORA analysis: {str(e)}")
        return None

def _format_sse_event(event_type: str, data: Dict) -> str:
    """Format a Server-Sent Event message.
    
    Args:
        event_type: The event type (e.g., 'progress', 'result')
        data: Dictionary containing event data (will be JSON serialized)
    
    Returns:
        Formatted SSE string following specification
    """
    json_data = json.dumps(data)
    return f"event: {event_type}\ndata: {json_data}\n\n"


def compute_importance_scores(sentences_importance: Dict, ranks: List[str], aggregate_type: str) -> List[List]:
    """Compute average importance score per context block."""
    importance_score = [[rank, 0] for rank in ranks]
    
    if aggregate_type == 'mean':
        i = -1
        count = 0
        temp_importance = 0.0
        for sentence, importance in sentences_importance.items():
            if '[Context]' in sentence:
                if i == -1:
                    temp_importance = importance
                    count = 1
                    i += 1
                else:
                    if i < len(importance_score) and count > 0:
                        importance_score[i][1] = temp_importance / count
                    temp_importance = importance
                    count = 1
                    i += 1
            else:
                temp_importance += importance
                count += 1
        importance_score[i][1] = temp_importance / count
    elif aggregate_type == 'max':
        i = -1
        temp_max = float('-inf')
        for sentence, importance in sentences_importance.items():
            if '[Context]' in sentence:
                if i == -1:
                    temp_max = importance
                    i += 1
                else:
                    if i < len(importance_score):
                        importance_score[i][1] = temp_max
                    temp_max = importance
                    i += 1
            else:
                if importance > temp_max:
                    temp_max = importance
        importance_score[i][1] = temp_max
    else:
        raise ValueError(f"Invalid aggregate type: {aggregate_type}")
    
    return importance_score


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route('/health/data_sources', methods=['GET'])
def health_check_data_sources():
    """Health check endpoint that verifies connectivity to all configured data sources.
    
    Returns:
        {
            "status": "healthy" | "degraded" | "unhealthy",
            "data_sources": {
                "source_name": {
                    "status": "healthy" | "unreachable" | "error",
                    "url": "...",
                    "message": "..."
                }
            }
        }
    """
    data_source_statuses = {}
    overall_status = "healthy"
    
    for src_config in data_source_configs:
        source_name = src_config['name']
        source_url = src_config['url']
        
        try:
            # Try to reach the health endpoint
            health_url = f"{source_url}/health"
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                health_data = response.json()
                data_source_statuses[source_name] = {
                    "status": "healthy",
                    "url": source_url,
                    "message": health_data.get("status", "ok"),
                    "dataset": health_data.get("dataset", "unknown")
                }
            else:
                data_source_statuses[source_name] = {
                    "status": "error",
                    "url": source_url,
                    "message": f"HTTP {response.status_code}"
                }
                overall_status = "degraded"
        except requests.exceptions.Timeout:
            data_source_statuses[source_name] = {
                "status": "unreachable",
                "url": source_url,
                "message": "Connection timeout"
            }
            overall_status = "degraded"
        except requests.exceptions.ConnectionError:
            data_source_statuses[source_name] = {
                "status": "unreachable",
                "url": source_url,
                "message": "Connection refused"
            }
            overall_status = "degraded"
        except Exception as e:
            data_source_statuses[source_name] = {
                "status": "error",
                "url": source_url,
                "message": str(e)
            }
            overall_status = "degraded"
    
    # If all sources are unreachable, mark as unhealthy
    if all(status["status"] == "unreachable" for status in data_source_statuses.values()):
        overall_status = "unhealthy"
    
    return jsonify({
        "status": overall_status,
        "data_sources": data_source_statuses
    }), 200 if overall_status == "healthy" else 503


@app.route('/score_events', methods=['GET'])
def get_score_events():
    """Get score update events from the blockchain.
    
    Query parameters:
        source_id: Optional filter by source ID
        source_address: Optional filter by source address
        from_block: Optional starting block number (inclusive)
        to_block: Optional ending block number (inclusive)
    
    Returns:
        {
            "events": [
                {
                    "sourceAddress": "...",
                    "sourceID": "...",
                    "sourceName": "...",
                    "reliabilityScore": int,
                    "usefulnessScore": int,
                    "timestamp": int,
                    "info": str,
                    "blockNumber": int,
                    "transactionHash": str,
                    "logIndex": int
                },
                ...
            ]
        }
    """
    try:
        if not blockchain_client:
            return jsonify({"error": "Blockchain client not available"}), 503
        
        # Get query parameters
        source_id = request.args.get('source_id', None)
        source_address = request.args.get('source_address', None)
        from_block = request.args.get('from_block', None, type=int)
        to_block = request.args.get('to_block', None, type=int)
        
        # Call blockchain client to get events
        events = blockchain_client.get_score_record_updated_events(
            source_id=source_id,
            source_address=source_address,
            from_block=from_block,
            to_block=to_block
        )
        
        return jsonify({"events": events}), 200
        
    except Exception as e:
        logger.error(f"Error retrieving score events: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query():
    """Query endpoint that returns just the response.
    
    Request body:
        {
            "query": "search query string"
        }
    
    Returns:
        {
            "response": "answer text"
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "Missing required field 'query'"}), 400
        
        query_text = data['query']
        n_contexts = retrieval_config['n_contexts']
        top_k = retrieval_config['top_k']
        
        # Get scores from blockchain to pass to data sources
        source_names = [src['name'] for src in data_source_configs]
        scores = get_scores_from_blockchain(source_names)
        
        # Format selected_sources as dict mapping source_id to (usefulness, reliability) tuples
        selected_sources = {}
        for name in source_names:
            score_data = scores.get(name, {})
            usefulness = int(score_data.get('usefulness', 0.0))
            reliability = int(score_data.get('reliability', 0.0))
            selected_sources[name] = (usefulness, reliability)
        
        # Query all data sources with blockchain scores
        candidates = query_data_sources(query_text, n_contexts, selected_sources=selected_sources)
        
        if not candidates:
            return jsonify({"error": "No candidates retrieved from data sources"}), 500
        
        # Rerank
        if retrieval_config['rerank_with_reliability']:
            # Get reliability scores from blockchain
            source_names = [src['name'] for src in data_source_configs]
            scores = get_scores_from_blockchain(source_names)
            reliability_scores = {name: scores.get(name, {}).get('reliability', 0.0) for name in source_names}
            
            selected = reranker.rerank_with_reliability(
                query_text,
                candidates,
                top_k=top_k,
                reliability_scores=reliability_scores,
                reliability_weight=retrieval_config['reliability_weight'],
                reliability_meta_key="source",
            )
        else:
            selected = reranker.rerank(query_text, candidates, top_k=top_k)
        
        selected = selected[::-1]  # Reverse to make last context most reliable
        
        # Build prompt
        context_blocks: List[str] = []
        for c in selected:
            text = str(c["text"]).strip()
            if text and not text.endswith('.'):
                text += '.'
            if text:
                context_blocks.append("[Context] " + text + "\n\n")
        
        context = "".join(context_blocks)
        prompt = context + "Question: " + query_text + "\n\nAnswer: "
        
        # Generate response using the model
        response_text = model.generate(prompt)
        
        return jsonify({"response": response_text}), 200
        
    except Exception as e:
        logger.error(f"Error during query: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _query_analyze(query_text: str, ground_truth: List[str], update_scores: bool, stream_mode: bool = False, progress_queue: Optional[Queue] = None):
    """Unified query analysis implementation that supports both streaming and non-streaming modes.
    
    Args:
        query_text: Search query string
        ground_truth: List of expected answers
        update_scores: Whether to update blockchain scores
        stream_mode: If True, yields SSE events; if False, returns a dictionary
        progress_queue: Optional Queue for progress updates (only used in streaming mode)
    
    Yields (if stream_mode=True):
        SSE events as the processing progresses
    
    Returns (if stream_mode=False):
        Dictionary with response, correctness, sampled_sources, importance_score, updated_scores
    """
    n_retrievers = retrieval_config['n_retrievers']
    n_contexts = retrieval_config['n_contexts']
    top_k = retrieval_config['top_k']
    
    # Sample data sources
    sampled_sources = sample_data_sources(
        retrieval_config['sample_with_usefulness'],
        n_retrievers
    )
    if stream_mode:
        yield _format_sse_event('sampling_sources', {'sources': sampled_sources})
    
    # Get scores from blockchain to pass to data sources
    source_names = [src['name'] for src in data_source_configs]
    scores = get_scores_from_blockchain(source_names)
    
    # Format selected_sources as dict mapping source_id to (usefulness, reliability) tuples
    # Only include sampled sources
    selected_sources = {}
    for name in sampled_sources:
        score_data = scores.get(name, {})
        usefulness = int(score_data.get('usefulness', 0.0))
        reliability = int(score_data.get('reliability', 0.0))
        selected_sources[name] = (usefulness, reliability)
    
    # Query sampled data sources with blockchain scores
    candidates = []
    source_signatures = {}  # Store signatures from data sources for score updates
    for src_config in data_source_configs:
        if src_config['name'] in sampled_sources:
            try:
                if stream_mode:
                    yield _format_sse_event('querying_sources', {'source': src_config['name'], 'status': 'querying'})
                url = f"{src_config['url']}/query"
                payload = {"query": query_text, "k": n_contexts}
                
                # Include selected_sources for validation and signing
                if selected_sources:
                    payload["selected_sources"] = selected_sources
                
                response = requests.post(url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    response_data = response.json()
                    results = response_data.get('results', [])
                    
                    for result in results:
                        candidates.append({
                            "id": f"{src_config['name']}::{result['id']}",
                            "text": result['text'],
                            "score": result['score'],
                            "meta": {**(result.get('meta', {})), "source": src_config['name']}
                        })
                    
                    # Store signature if present (needed for score updates)
                    if 'signature' in response_data and update_scores:
                        signature_hex = response_data['signature']
                        # Convert hex string to bytes
                        if signature_hex.startswith('0x'):
                            source_signatures[src_config['name']] = bytes.fromhex(signature_hex[2:])
                        else:
                            source_signatures[src_config['name']] = bytes.fromhex(signature_hex)
                        logger.debug(f"Received signature from {src_config['name']}: {signature_hex[:20]}...")
                    
                    if stream_mode:
                        yield _format_sse_event('querying_sources', {'source': src_config['name'], 'status': 'success', 'results_count': len(results)})
                else:
                    logger.warning(f"Failed to query {src_config['name']}: {response.status_code}")
                    if stream_mode:
                        yield _format_sse_event('querying_sources', {'source': src_config['name'], 'status': 'error', 'error': f"HTTP {response.status_code}"})
                    if response.text:
                        logger.warning(f"Error response: {response.text[:200]}")
            except Exception as e:
                logger.error(f"Error querying {src_config['name']}: {str(e)}")
                if stream_mode:
                    yield _format_sse_event('querying_sources', {'source': src_config['name'], 'status': 'error', 'error': str(e)})
    
    if not candidates:
        error_msg = 'No candidates retrieved from data sources'
        if stream_mode:
            yield _format_sse_event('error', {'error': error_msg})
            return
        else:
            raise ValueError(error_msg)
    
    # Rerank
    if stream_mode:
        yield _format_sse_event('reranking', {'status': 'starting'})
    if retrieval_config['rerank_with_reliability']:
        # Get reliability scores from blockchain
        source_names = [src['name'] for src in data_source_configs]
        scores = get_scores_from_blockchain(source_names)
        reliability_scores = {name: scores.get(name, {}).get('reliability', 0.0) for name in source_names}
        
        selected = reranker.rerank_with_reliability(
            query_text,
            candidates,
            top_k=top_k,
            reliability_scores=reliability_scores,
            reliability_weight=retrieval_config['reliability_weight'],
            reliability_meta_key="source",
        )
    else:
        selected = reranker.rerank(query_text, candidates, top_k=top_k)
    if stream_mode:
        yield _format_sse_event('reranking', {'status': 'completed', 'selected_count': len(selected)})
    
    selected = selected[::-1]  # Reverse to make last context most reliable
    
    # Build prompt
    context_blocks: List[str] = []
    ranks: List[str] = []
    for c in selected:
        text = str(c["text"]).strip()
        if text and not text.endswith('.'):
            text += '.'
        if text:
            context_blocks.append("[Context] " + text + "\n\n")
        ranks.append(c["meta"]["source"])
    
    context = "".join(context_blocks)
    prompt = context + "Question: " + query_text + "\n\nAnswer: "
    
    # Generate response and analyze sentence-level importance
    if stream_mode:
        yield _format_sse_event('mc_shap', {'status': 'starting'})
        
        # Run analysis in a separate thread and monitor progress queue in real-time
        import threading
        analysis_result_container = {'result': None, 'exception': None}
        
        def run_analysis():
            try:
                if sentence_importance_config['sentence_importance_method'] == 'mc_shap':
                    analysis_result_container['result'] = analyze_with_mc_shapley(
                        prompt, 
                        retrieval_config.get('max_tokens', 900), 
                        progress_queue=progress_queue
                    )
                elif sentence_importance_config['sentence_importance_method'] == 'rora':
                    analysis_result_container['result'] = analyze_with_rora(
                        query_text,
                        ground_truth,
                        context_blocks
                    )
                else:
                    raise ValueError("No sentence importance evaluation method enabled")
            except Exception as e:
                analysis_result_container['exception'] = e
        
        analysis_thread = threading.Thread(target=run_analysis, daemon=True)
        analysis_thread.start()
        
        # Monitor progress queue and yield events in real-time
        import time
        while analysis_thread.is_alive() or (progress_queue and not progress_queue.empty()):
            # Check for progress messages
            try:
                    if progress_queue:
                        msg = progress_queue.get(timeout=0.1)
                        if msg.get('message') == 'baseline_calculated':
                            yield _format_sse_event('mc_shap_baseline', {'baseline': msg.get('baseline', '')})
                        elif msg.get('message') == 'Processing':
                            yield _format_sse_event('mc_shap_progress', {
                                'count': msg.get('count', 0),
                                'total': msg.get('total', 0),
                                'progress': msg.get('count', 0) / msg.get('total', 1) if msg.get('total', 0) > 0 else 0.0
                            })
            except:
                # Queue empty or timeout - continue monitoring
                time.sleep(0.05)
                continue
        
        # Wait for analysis thread to complete
        analysis_thread.join(timeout=1.0)
        
        # Get final result
        if analysis_result_container['exception']:
            yield _format_sse_event('error', {'error': f'Analysis failed: {str(analysis_result_container["exception"])}'})
            return
        
        analysis_result = analysis_result_container['result']
        if not analysis_result:
            yield _format_sse_event('error', {'error': 'Failed to generate response'})
            return
    else:
        # Non-streaming mode: direct analysis
        if sentence_importance_config['sentence_importance_method'] == 'mc_shap':
            analysis_result = analyze_with_mc_shapley(prompt, retrieval_config.get('max_tokens', 900))
        elif sentence_importance_config['sentence_importance_method'] == 'rora':
            analysis_result = analyze_with_rora(query_text, ground_truth, context_blocks)
        else:
            raise ValueError("No sentence importance evaluation method enabled")
        
        if not analysis_result:
            raise ValueError("Failed to generate response")
        
    
    response_text = analysis_result['response']
    sentences_importance = analysis_result['importance']
    
    # Compute importance scores per context
    if stream_mode:
        yield _format_sse_event('computing_importance', {'status': 'starting'})
    importance_score = compute_importance_scores(
        sentences_importance,
        ranks,
        retrieval_config['aggregate_type']
    )
    if stream_mode:
        yield _format_sse_event('computing_importance', {'status': 'completed', 'sources_count': len(importance_score)})
    
    # Check correctness
    correctness = any(
        response_text.lower().strip() == answer.lower().strip() for answer in ground_truth
    ) if ground_truth else None

    # Check grounding (if response can be found in selected contexts)
    norm_response = _norm_join(_normalize_for_match(response_text))
    selected_contexts_norm = [_norm_join(_normalize_for_match(c["text"])) for c in selected]
    grounded_by_sources = [norm_response in ctx for ctx in selected_contexts_norm]

    # Prepare info for contract (JSON string with response, correctness, sampled_sources, importance_score)
    info_dict = {
        "response": response_text,
        "correctness": correctness,
        "sampled_sources": sampled_sources,
        "importance_score": importance_score
    }
    info_json = json.dumps(info_dict, sort_keys=True)

    # Update scores if requested
    if update_scores and blockchain_client and blockchain_config:
        try:
            if stream_mode:
                yield _format_sse_event('updating_scores', {'status': 'starting'})
            # Get current scores from blockchain
            current_scores = get_scores_from_blockchain(sampled_sources)
            
            # Aggregate score updates per source (a source may appear multiple times in importance_score)
            # Scale importance_score by 100x and round to integer
            source_updates = {}  # source_id -> {reliability_delta, usefulness_delta, signature}
            
            # Calculate deltas for each context/source
            for i, item in enumerate(importance_score):
                # item is [source_id, score]
                source_id = item[0]
                score_value = item[1]
                
                # Only process sources that were sampled and have signatures
                if source_id not in sampled_sources or source_id not in source_signatures:
                    if source_id not in source_signatures:
                        logger.debug(f"Skipping {source_id}: no signature available")
                    continue
                
                # Scale importance by 100x and round to integer
                scaled_importance = int(round(score_value * 100))

                # Initialize source update if not exists
                if source_id not in source_updates:
                    source_updates[source_id] = {
                        'reliability_delta': 0,
                        'usefulness_delta': 0,
                        'signature': source_signatures[source_id]
                    }
                
                # Calculate deltas based on grounding and correctness
                # Logic from open_book_qa_decentralized.py:
                # - If grounded: update reliability (signed by correctness) and usefulness
                # - If not grounded: penalize usefulness only
                if i < len(grounded_by_sources) and grounded_by_sources[i]:
                    # Grounded: update both reliability and usefulness
                    if correctness:
                        # Correct answer: positive update
                        source_updates[source_id]['reliability_delta'] += scaled_importance
                        source_updates[source_id]['usefulness_delta'] += scaled_importance
                    else:
                        # Incorrect answer: negative reliability update
                        source_updates[source_id]['reliability_delta'] -= scaled_importance
                        source_updates[source_id]['usefulness_delta'] += scaled_importance  # Still reward usefulness for grounded
                else:
                    # Not grounded: penalize usefulness only
                    source_updates[source_id]['usefulness_delta'] -= scaled_importance
            
            # Build arrays for contract call (one entry per unique source)
            if len(source_updates) > 0:
                update_source_ids = []
                update_reliability_scores = []
                update_usefulness_scores = []
                signatures_list = []
                
                for source_id, updates in source_updates.items():
                    # Get current scores
                    current_rel = int(current_scores.get(source_id, {}).get('reliability', 0.0))
                    current_use = int(current_scores.get(source_id, {}).get('usefulness', 0.0))
                    
                    # Calculate final scores (current + deltas)
                    new_reliability = current_rel + updates['reliability_delta']
                    new_usefulness = current_use + updates['usefulness_delta']
                    
                    update_source_ids.append(source_id)
                    update_reliability_scores.append(new_reliability)
                    update_usefulness_scores.append(new_usefulness)
                    signatures_list.append(updates['signature'])
                
                # Build message for signing (same format as data sources expect)
                message_dict = {
                    "query": query_text,
                    "selected_sources": selected_sources
                }
                message_json = json.dumps(message_dict, sort_keys=True)
                
                # Call contract to update scores
                private_key = blockchain_config.get('private_key')
                if private_key:
                    try:
                        tx_hash = blockchain_client.feedback_and_update_score_records(
                            caller_private_key=private_key,
                            message=message_json,
                            signatures=signatures_list,
                            update_source_ids=update_source_ids,
                            update_reliability_scores=update_reliability_scores,
                            update_usefulness_scores=update_usefulness_scores,
                            info=info_json
                        )
                        logger.info(f"Score update transaction submitted: {tx_hash}")
                        if stream_mode:
                            yield _format_sse_event('updating_scores', {'status': 'completed', 'tx_hash': tx_hash})
                    except Exception as e:
                        logger.error(f"Failed to update scores on blockchain: {str(e)}")
                        if stream_mode:
                            yield _format_sse_event('updating_scores', {'status': 'error', 'error': str(e)})
                else:
                    logger.warning("No private key configured for blockchain updates")
                    if stream_mode:
                        yield _format_sse_event('updating_scores', {'status': 'error', 'error': 'No private key configured'})
            else:
                logger.warning(f"Cannot update scores: no sources with signatures available")
                if stream_mode:
                    yield _format_sse_event('updating_scores', {'status': 'skipped', 'reason': 'No signatures available'})
        except Exception as e:
            logger.error(f"Error updating scores: {str(e)}", exc_info=True)
            if stream_mode:
                yield _format_sse_event('updating_scores', {'status': 'error', 'error': str(e)})
    
    # Final result
    result_data = {
        "response": response_text,
        "correctness": correctness,
        "sampled_sources": sampled_sources,
        "importance_score": importance_score
    }
    
    # Include updated scores if scores were updated
    if update_scores and blockchain_client:
        try:
            updated_scores = get_scores_from_blockchain(sampled_sources)
            result_data['updated_scores'] = updated_scores
        except Exception as e:
            logger.warning(f"Failed to fetch updated scores: {str(e)}")
    
    # Yield the final result
    if stream_mode:
        yield _format_sse_event('final_result', result_data)
    else:
        # In non-streaming mode, yield the raw data (endpoint will extract it)
        yield result_data


@app.route('/query_analyze', methods=['POST'])
def query_analyze():
    """Query endpoint that returns response with analysis.
    
    Request body:
        {
            "query": "search query string",
            "ground_truth": ["expected answer 1", "expected answer 2"],
            "update_scores": true/false
        }
    
    Query parameters:
        stream: If set to 'true', returns SSE stream instead of JSON
    
    Returns:
        If stream=true: SSE events with progress updates and final result
        Otherwise: JSON with response, correctness, sampled_sources, importance_score, updated_scores
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "Missing required field 'query'"}), 400
        
        # Check if streaming is requested
        stream_mode = request.args.get('stream', 'false').lower() == 'true'
        
        query_text = data['query']
        ground_truth = data.get('ground_truth', [])
        update_scores = data.get('update_scores', False)
        
        if stream_mode:
            # SSE streaming mode
            progress_queue = Queue()
            return Response(
                _query_analyze(query_text, ground_truth, update_scores, stream_mode=True, progress_queue=progress_queue),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            # Non-streaming mode: extract the final result from the generator
            generator = _query_analyze(query_text, ground_truth, update_scores, stream_mode=False)
            # The generator yields the final result as the last value
            response_data = None
            for result in generator:
                response_data = result
            if response_data is None:
                return jsonify({"error": "No result generated"}), 500
            return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error during query_analyze: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Load configuration
    import os
    config_path = os.getenv('CONFIG_PATH', 'configs/config.yaml')
    config = load_config(config_path)
    
    # Initialize model
    initialize_model(config['model'])

    # Set global configs (need retrieval_config before initializing sentence importance)
    data_source_configs = config['data_sources']
    retrieval_config = config['retrieval']
    sentence_importance_config = config['sentence_importance']
    # initialize sentence importance evaluation
    # This will set aggregate_type and max_tokens in retrieval_config based on the selected method
    initialize_sentence_importance(sentence_importance_config)

    # initialize reranker
    logger.info("Initializing reranker")
    reranker = Reranker(
        method=config['reranker']['method'],
        model_name=config['reranker']['model_name'],
        hybrid_alpha=config['reranker']['hybrid_alpha'],
        orig_score_weight=config['reranker']['orig_score_weight'],
        device=config['reranker']['device']
    )
    
    # Initialize blockchain client
    blockchain_config = config.get('blockchain', {})
    if blockchain_config and DragScoresClient:
        try:
            # Get project root (parent of drag_llm_service)
            # In Docker: /app, so project_root is /app
            # Locally: two levels up from app/server.py
            if os.path.exists('/app'):
                project_root = '/app'
            else:
                project_root = blockchain_config.get('project_root', '../..')
                # Resolve project root path
                if not os.path.isabs(project_root):
                    config_dir = Path(config_path).parent
                    project_root = str(config_dir / project_root)
            
            provider_url = blockchain_config.get('provider_url', 'http://127.0.0.1:8545')
            contract_address = blockchain_config.get('contract_address')
            
            blockchain_client = DragScoresClient(
                project_root=project_root,
                provider_url=provider_url,
                contract_address=contract_address
            )
            
            # Verify connection
            try:
                hello_msg = blockchain_client.hello()
                logger.info(f"Blockchain client connected: {hello_msg}")
            except Exception as e:
                logger.warning(f"Blockchain client initialized but connection test failed: {str(e)}")
                blockchain_client = None
        except Exception as e:
            logger.error(f"Failed to initialize blockchain client: {str(e)}")
            blockchain_client = None
    else:
        logger.warning("No blockchain configuration found, scores will default to 0.0")
        blockchain_client = None
    
    # Start server
    server_config = config['server']
    # Enable debug mode from environment variable or config, default to True
    debug_mode = os.getenv('FLASK_DEBUG', str(server_config.get('debug', True))).lower() in ('true', '1', 'yes')
    logger.info(f"Starting server on {server_config['host']}:{server_config['port']} (debug={debug_mode})")
    app.run(
        host=server_config['host'],
        port=server_config['port'],
        debug=debug_mode
    )
