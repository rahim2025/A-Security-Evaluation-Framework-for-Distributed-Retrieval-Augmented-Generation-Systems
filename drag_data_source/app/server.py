"""API server for the data source retrieval service.
"""
import json
import yaml
import sys
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add drag_python_client to path (both for local and Docker contexts)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'drag_python_client')))
sys.path.insert(0, '/app/drag_python_client')

from flask import Flask, request, jsonify
from src.retriever.retriever import FastRetriever, Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import drag contract library
try:
    from drag_python_client import DragScoresClient, sign_message_personal
    DRAG_CLIENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"drag_python_client not available: {str(e)}. Contract validation will be disabled.")
    DRAG_CLIENT_AVAILABLE = False
    DragScoresClient = None
    sign_message_personal = None

app = Flask(__name__)
DS_API_KEY = os.getenv("API_KEY", "")

@app.before_request
def check_api_key():
    if not DS_API_KEY:
        return  # no key configured — allow all (backwards compat)
    if request.endpoint in ("health_check",):
        return  # health check always accessible
    if request.headers.get("X-API-Key", "") != DS_API_KEY:
        logger.warning(f"[SECURITY] Unauthorized /query attempt from {request.remote_addr}")
        return jsonify({"error": "Unauthorized", "message": "Valid X-API-Key header required"}), 401


API_KEY = os.getenv("API_KEY", "")

@app.before_request
def check_api_key():
    if not API_KEY:
        return
    if request.endpoint in ("health_check",):
        return
    if request.headers.get("X-API-Key", "") != API_KEY:
        return jsonify({"error": "Unauthorized", "message": "X-API-Key header required"}), 401

# Global instances
retriever = None
dataset_name = None
contract_client = None
blockchain_config = None


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def initialize_retriever(config: Dict) -> None:
    """Initialize the FastRetriever with documents from the configured path."""
    global retriever, dataset_name
    
    logger.info("Initializing retriever...")
    
    # Load documents from JSONL file
    jsonl_path = config['data']['jsonl_path']
    dataset_name = config['data']['dataset_name']
    
    logger.info(f"Loading documents from {jsonl_path}")
    docs: List[Document] = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec_id = rec.get("id") or rec.get("htmlid", "")
            
            # Extract text from either "documents" or "html" field
            rec_text = rec.get("documents") or rec.get("html", "")
            
            meta = {
                "record_id": rec_id,
                "dataset_name": dataset_name
            }
            docs.append(Document(id=rec_id, text=rec_text, meta=meta))
    
    logger.info(f"Loaded {len(docs)} documents")
    
    # Initialize retriever with configuration
    retriever_config = config['retriever']
    retriever = FastRetriever(
        model_name=retriever_config['model_name'],
        normalize=retriever_config['normalize'],
        use_faiss=retriever_config['use_faiss'],
        device=retriever_config['device'],
        batch_size=retriever_config['batch_size']
    )
    
    # Fit the retriever with documents
    retriever.fit(docs)
    logger.info("Retriever initialized successfully")


def initialize_contract_client(config: Dict) -> None:
    """Initialize the DragScores contract client."""
    global contract_client, blockchain_config
    
    if not DRAG_CLIENT_AVAILABLE:
        logger.warning("Drag contract client not available. Skipping initialization.")
        return
    
    blockchain_config = config.get('blockchain', {})
    if not blockchain_config:
        logger.warning("No blockchain configuration found. Contract validation will be disabled.")
        return
    
    provider_url = blockchain_config.get('provider_url')
    contract_address = blockchain_config.get('contract_address')
    
    try:
        # Get project root (parent of drag_data_source)
        # In Docker: /app, so project_root is /app
        # Locally: two levels up from app/server.py
        if os.path.exists('/app'):
            project_root = '/app'
        else:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        contract_client = DragScoresClient(
            project_root=project_root,
            provider_url=provider_url,
            contract_address=contract_address
        )
        logger.info(f"Contract client initialized. Provider: {provider_url}")
        
        # Test connection
        hello_msg = contract_client.hello()
        logger.info(f"Contract connection verified: {hello_msg}")
        
    except Exception as e:
        logger.error(f"Failed to initialize contract client: {str(e)}")
        logger.warning("Contract validation will be disabled.")
        contract_client = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "dataset": dataset_name})


@app.route('/poison', methods=['POST'])
def inject_poison():
    """Inject poisoned documents into the retriever at runtime.

    Request body:
        {
            "documents": [
                {"id": "poison_0", "text": "poisoned content"},
                ...
            ]
        }

    Returns:
        {"status": "poisoned", "injected_count": N, "total_docs": M}
    """
    global retriever
    if retriever is None:
        return jsonify({"error": "Retriever not initialized"}), 500

    data = request.get_json()
    if not data or 'documents' not in data:
        return jsonify({"error": "Missing 'documents' field"}), 400

    new_docs = []
    for i, doc in enumerate(data['documents']):
        new_docs.append(Document(
            id=str(doc.get('id', f'poison_{len(retriever.docs) + i}')),
            text=str(doc.get('text', '')),
            meta={**(doc.get('meta', {})), 'poisoned': True, 'dataset_name': dataset_name}
        ))

    all_docs = retriever.docs + new_docs
    retriever.fit(all_docs)

    logger.warning(f"[ATTACK] Injected {len(new_docs)} poisoned documents. Total docs: {len(retriever.docs)}")
    return jsonify({
        "status": "poisoned",
        "injected_count": len(new_docs),
        "total_docs": len(retriever.docs)
    }), 200


@app.route('/reset', methods=['POST'])
def reset_retriever():
    """Reset the retriever to its original clean state (removes all poisoned docs)."""
    config_path = os.getenv('CONFIG_PATH', 'configs/config.yaml')
    config = load_config(config_path)
    initialize_retriever(config)
    logger.info(f"[RESET] Retriever restored to clean state with {len(retriever.docs)} docs")
    return jsonify({"status": "reset", "total_docs": len(retriever.docs)}), 200


@app.route('/info', methods=['GET'])
def get_info():
    """Return current retriever stats."""
    if retriever is None:
        return jsonify({"error": "Retriever not initialized"}), 500
    poisoned_count = sum(1 for d in retriever.docs if (d.meta or {}).get('poisoned', False))
    return jsonify({
        "dataset": dataset_name,
        "total_docs": len(retriever.docs),
        "clean_docs": len(retriever.docs) - poisoned_count,
        "poisoned_docs": poisoned_count
    }), 200


@app.route('/query', methods=['POST'])
def query():
    """Query endpoint for document retrieval.
    
    Request body:
        {
            "query": "search query string",
            "k": 10,  // optional, number of documents to return
            "selected_sources": {  // optional, dict of {source_id: (usefulness, reliability)}
                "sources_0": (10000, 10000),
                "sources_20": (10000, 10000)
            }
        }
    
    Returns:
        {
            "results": [...],
            "signature": "0x..."  // if selected_sources provided and validated
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "Missing required field 'query'"}), 400
        
        query_text = data['query']
        k = data.get('k', 10)
        selected_sources = data.get('selected_sources', {})
        
        if retriever is None:
            return jsonify({"error": "Retriever not initialized"}), 500
        
        # Validate selected_sources scores if provided
        signature = None
        if selected_sources:
            if contract_client is None:
                logger.warning("Contract client not initialized. Skipping validation and signing of selected_sources.")
                # Continue without validation/signing - query can still proceed
            else:
                # Validate scores against contract
                validation_error = validate_selected_sources(selected_sources)
                if validation_error:
                    return jsonify({"error": validation_error}), 400
                
                # Construct dict with query and selected_sources
                message_dict = {
                    "query": query_text,
                    "selected_sources": selected_sources
                }
                
                # Convert to JSON string for signing
                message_json = json.dumps(message_dict, sort_keys=True)
                
                # Sign the message using the private key from config
                private_key = os.getenv("PRIVATE_KEY") or blockchain_config.get('private_key')
                if not private_key:
                    logger.warning("Private key not configured. Skipping message signing.")
                else:
                    try:
                        signature_bytes = sign_message_personal(message_json, private_key)
                        signature = "0x" + signature_bytes.hex()
                    except Exception as e:
                        logger.error(f"Error signing message: {str(e)}")
                        # Continue without signature - query can still proceed
        
        # Perform search
        results = retriever.search(query_text, k=k)
        
        response = {"results": results}
        if signature:
            response["signature"] = signature
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error during query: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def validate_selected_sources(selected_sources: Dict) -> Optional[str]:
    """Validate that the provided scores match the on-chain scores.
    
    Args:
        selected_sources: Dict mapping source_id to (usefulness, reliability) tuple or list
        
    Returns:
        None if validation passes, error message string if validation fails
    """
    if not contract_client:
        return "Contract client not available"
    
    source_ids = list(selected_sources.keys())
    
    try:
        # Get scores from contract
        returned_ids, reliability_scores, usefulness_scores = contract_client.get_scores_batch(source_ids)
        
        # Create mapping of returned scores
        contract_scores = {}
        for i, source_id in enumerate(returned_ids):
            if i < len(reliability_scores) and i < len(usefulness_scores):
                contract_scores[source_id] = (usefulness_scores[i], reliability_scores[i])
        
        # Validate each source (handle both tuple and list formats)
        for source_id, score_pair in selected_sources.items():
            if source_id not in contract_scores:
                return f"Source {source_id} not found in contract"
            
            # Convert to tuple if it's a list
            if isinstance(score_pair, list):
                if len(score_pair) < 2:
                    return f"Invalid score format for {source_id}: expected list of [usefulness, reliability]"
                provided_usefulness, provided_reliability = score_pair[0], score_pair[1]
            elif isinstance(score_pair, tuple):
                if len(score_pair) < 2:
                    return f"Invalid score format for {source_id}: expected tuple of (usefulness, reliability)"
                provided_usefulness, provided_reliability = score_pair[0], score_pair[1]
            else:
                return f"Invalid score format for {source_id}: expected tuple or list of (usefulness, reliability)"
            
            contract_usefulness, contract_reliability = contract_scores[source_id]
            
            if provided_usefulness != contract_usefulness:
                return (
                    f"Usefulness score mismatch for {source_id}: "
                    f"provided={provided_usefulness}, contract={contract_usefulness}"
                )
            
            if provided_reliability != contract_reliability:
                return (
                    f"Reliability score mismatch for {source_id}: "
                    f"provided={provided_reliability}, contract={contract_reliability}"
                )
        
        return None  # Validation passed
        
    except Exception as e:
        logger.error(f"Error validating selected_sources: {str(e)}")
        return f"Contract validation error: {str(e)}"


if __name__ == '__main__':
    # Load configuration from environment variable or default
    import os
    config_path = os.getenv('CONFIG_PATH', 'configs/config.yaml')
    config = load_config(config_path)
    
    # Initialize contract client
    initialize_contract_client(config)
    
    # Initialize retriever
    initialize_retriever(config)
    
    # Start server
    server_config = config['server']
    logger.info(f"Starting server on {server_config['host']}:{server_config['port']}")
    app.run(
        host=server_config['host'],
        port=server_config['port'],
        debug=False
    )
