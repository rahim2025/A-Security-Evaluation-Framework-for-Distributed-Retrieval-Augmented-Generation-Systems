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

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path for the defense package (defense/datapoisoning_defense)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from defense.datapoisoning_defense.poison_filter import PoisonFilter
    POISON_FILTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"defense.datapoisoning_defense not available: {e}. /poison will run unfiltered.")
    POISON_FILTER_AVAILABLE = False
    PoisonFilter = None

# Content-integrity gate on /poison -- see defense/datapoisoning_defense/README.md.
# Defaults to ON; set POISON_FILTER_ENABLED=0 to reproduce the undefended baseline
# without code changes (e.g. to re-run attack/datapoisoning's own evaluation as the
# "before" comparison point).
POISON_FILTER_ENABLED = os.getenv("POISON_FILTER_ENABLED", "1") not in ("0", "false", "False")
poison_filter = PoisonFilter() if (POISON_FILTER_AVAILABLE and POISON_FILTER_ENABLED) else None

try:
    from defense.ssm_defense.content_shape_detector import GroundingFarmDetector
    from defense.ssm_defense.sentence_length_detector import SentenceLengthDetector
    GROUNDING_FARM_DETECTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"defense.ssm_defense not available: {e}. /poison will not screen for grounding-farming filler.")
    GROUNDING_FARM_DETECTOR_AVAILABLE = False
    GroundingFarmDetector = None
    SentenceLengthDetector = None

# Content-shape gates on /poison against SSM grounding-farming filler -- see
# defense/ssm_defense/README.md. Independent of PoisonFilter above
# (different attack, same injection endpoint). Two complementary signals:
# hedge-attribution phrasing (catches the attack as implemented) and average
# sentence length (catches an adaptive attacker who reworded the connective
# phrases but still needs the same short-sentence, many-disconnected-facts
# structure). Defaults to ON; set GROUNDING_FARM_DEFENSE_ENABLED=0 to
# reproduce the undefended baseline.
GROUNDING_FARM_DEFENSE_ENABLED = os.getenv("GROUNDING_FARM_DEFENSE_ENABLED", "1") not in ("0", "false", "False")
grounding_farm_detector = (
    GroundingFarmDetector() if (GROUNDING_FARM_DETECTOR_AVAILABLE and GROUNDING_FARM_DEFENSE_ENABLED) else None
)
sentence_length_detector = (
    SentenceLengthDetector() if (GROUNDING_FARM_DETECTOR_AVAILABLE and GROUNDING_FARM_DEFENSE_ENABLED) else None
)

# --- KB-extraction guard on /query (see defense/kb_extraction_defense/README.md) ----------
# Label-free, per-caller guard: discovery-velocity + query-dispersion flags that degrade the
# result count, and a hard per-caller / global budget of distinct documents served per epoch.
# Callers that present ORCHESTRATOR_TOKEN (the llm-service, the only legitimate bulk reader)
# bypass it. Defaults to ON; set EXTRACTION_GUARD_ENABLED=0 to reproduce the undefended
# baseline. Tunables: EXTRACTION_GUARD_CONFIG (JSON of GuardConfig fields).
try:
    from defense.kb_extraction_defense.extraction_guard import ExtractionGuard, GuardConfig, ALLOW, cluster_corpus
    EXTRACTION_GUARD_AVAILABLE = True
except ImportError as e:
    logger.warning(f"defense.kb_extraction_defense not available: {e}. /query will run unguarded.")
    EXTRACTION_GUARD_AVAILABLE = False
    ExtractionGuard = None

EXTRACTION_GUARD_ENABLED = os.getenv("EXTRACTION_GUARD_ENABLED", "1") not in ("0", "false", "False")
EXTRACTION_GUARD_DEBUG = os.getenv("EXTRACTION_GUARD_DEBUG", "0") in ("1", "true", "True")
ORCHESTRATOR_TOKEN = os.getenv("ORCHESTRATOR_TOKEN", "")


def _build_extraction_guard():
    if not (EXTRACTION_GUARD_AVAILABLE and EXTRACTION_GUARD_ENABLED):
        return None
    overrides = json.loads(os.getenv("EXTRACTION_GUARD_CONFIG", "{}") or "{}")
    cfg = GuardConfig(**{**DEFAULT_GUARD_FIELDS, **overrides})
    clusters = None
    if cfg.breadth_thresh is not None and retriever is not None and getattr(retriever, "emb", None) is not None:
        # Unsupervised pseudo-topics from this server's OWN corpus embeddings (no labels, no oracle).
        labels = cluster_corpus(retriever.emb, cfg.n_clusters)
        clusters = {d.id: int(l) for d, l in zip(retriever.docs, labels)}
    return ExtractionGuard(cfg, doc_clusters=clusters)


# Calibrated defaults (defense/kb_extraction_defense/offline_eval.py calibrate); see README.
DEFAULT_GUARD_FIELDS = dict(window=30, min_queries=10, velocity_thresh=None, dispersion_thresh=None,
                            breadth_thresh=6, n_clusters=10, degrade_k=1, cooldown_queries=20, escalate_after_flags=3,
                            client_budget=202, global_budget=None, epoch_seconds=86400.0)
retriever = None            # populated by initialize_retriever(); declared early for the guard
extraction_guard = _build_extraction_guard()


def _is_orchestrator(req) -> bool:
    import hmac
    sent = req.headers.get("X-Orchestrator-Token", "")
    return bool(ORCHESTRATOR_TOKEN) and hmac.compare_digest(sent, ORCHESTRATOR_TOKEN)


def _client_id(req) -> str:
    return req.remote_addr or "unknown"

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
# Overridable per-deployment via RATE_LIMIT_DEFAULT (Flask-Limiter string syntax,
# e.g. "600 per minute" or "1000 per hour") -- defaults to the original hardcoded
# "60 per minute" so existing behavior is unchanged unless explicitly opted into.
# A full SFA sweep can exceed 60/min-per-IP within seconds; raising this for a
# test deployment avoids multi-hour client-side-throttled runs while keeping the
# limiter itself (and its default) intact for anyone who relies on the default.
# See reports/SFA_Security_Analysis_Report.md sec 12.9 for the confirmed root-cause
# writeup of what happens when this limit is hit unthrottled.
RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "60 per minute")
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=[RATE_LIMIT_DEFAULT])
DS_API_KEY = os.getenv("API_KEY", "")

@app.before_request
def check_api_key():
    if not DS_API_KEY:
        return
    if request.endpoint in ("health_check",):
        return
    logging.info("ACCESS ip=%s path=%s key_present=%s",
                 request.remote_addr, request.path,
                 bool(request.headers.get("X-API-Key")))
    if request.headers.get("X-API-Key", "") != DS_API_KEY:
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
    global extraction_guard
    extraction_guard = _build_extraction_guard()  # needs the corpus embeddings (pseudo-topic clusters)


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
        {"status": "poisoned", "injected_count": N, "total_docs": M,
         "rejected_count": R, "rejections": [...]}  (rejected_count/rejections
         only present when the content filter is enabled)
    """
    global retriever
    if retriever is None:
        return jsonify({"error": "Retriever not initialized"}), 500

    data = request.get_json()
    if not data or 'documents' not in data:
        return jsonify({"error": "Missing 'documents' field"}), 400

    incoming = data['documents']

    rejections = []
    if poison_filter is not None:
        allowed_count, volume_reason = poison_filter.check_batch_volume(
            requested_count=len(incoming), current_total_docs=len(retriever.docs)
        )
        if volume_reason:
            dropped = incoming[allowed_count:]
            incoming = incoming[:allowed_count]
            for doc in dropped:
                rejections.append({"id": doc.get('id'), "reasons": [volume_reason]})
            logger.warning(f"[DEFENSE] {volume_reason}")

        verdicts = poison_filter.classify_batch([str(doc.get('text', '')) for doc in incoming])
        kept = []
        for doc, verdict in zip(incoming, verdicts):
            if verdict.accepted:
                kept.append(doc)
            else:
                rejections.append({"id": doc.get('id'), "reasons": verdict.reasons})
        incoming = kept

    if grounding_farm_detector is not None:
        shape_verdicts = grounding_farm_detector.classify_batch([str(doc.get('text', '')) for doc in incoming])
        kept = []
        for doc, verdict in zip(incoming, shape_verdicts):
            if verdict.flagged:
                rejections.append({
                    "id": doc.get('id'),
                    "reasons": [f"grounding_farm_shape(hedge_sentences={verdict.distinct_sentences_with_hedge})"],
                })
            else:
                kept.append(doc)
        incoming = kept

    if sentence_length_detector is not None:
        length_verdicts = sentence_length_detector.classify_batch([str(doc.get('text', '')) for doc in incoming])
        kept = []
        for doc, verdict in zip(incoming, length_verdicts):
            if verdict.flagged:
                rejections.append({
                    "id": doc.get('id'),
                    "reasons": [f"grounding_farm_short_sentences(avg_words={verdict.avg_sentence_length:.1f})"],
                })
            else:
                kept.append(doc)
        incoming = kept

    new_docs = []
    for i, doc in enumerate(incoming):
        new_docs.append(Document(
            id=str(doc.get('id', f'poison_{len(retriever.docs) + i}')),
            text=str(doc.get('text', '')),
            meta={**(doc.get('meta', {})), 'poisoned': True, 'dataset_name': dataset_name}
        ))

    if new_docs:
        retriever.add_documents(new_docs)

    if rejections:
        logger.warning(
            f"[DEFENSE] Rejected {len(rejections)} candidate document(s) at /poison "
            f"(see 'rejections' in response for reasons)."
        )
    logger.warning(f"[ATTACK] Injected {len(new_docs)} poisoned documents. Total docs: {len(retriever.docs)}")

    response = {
        "status": "poisoned",
        "injected_count": len(new_docs),
        "total_docs": len(retriever.docs),
    }
    if poison_filter is not None or grounding_farm_detector is not None or sentence_length_detector is not None:
        response["rejected_count"] = len(rejections)
        response["rejections"] = rejections
    return jsonify(response), 200


@app.route('/poison_filter/stats', methods=['GET'])
def poison_filter_stats():
    """Cumulative detection stats for this process's content-integrity gate."""
    if poison_filter is None:
        return jsonify({"enabled": False}), 200
    return jsonify({"enabled": True, **poison_filter.get_stats()}), 200


@app.route('/grounding_farm_defense/stats', methods=['GET'])
def grounding_farm_defense_stats():
    """Cumulative detection stats for this process's SSM grounding-farming gates
    (hedge-attribution content-shape + average-sentence-length)."""
    if grounding_farm_detector is None and sentence_length_detector is None:
        return jsonify({"enabled": False}), 200
    return jsonify({
        "enabled": True,
        "hedge_attribution": grounding_farm_detector.get_stats() if grounding_farm_detector else None,
        "sentence_length": sentence_length_detector.get_stats() if sentence_length_detector else None,
    }), 200


@app.route('/extraction_guard/stats', methods=['GET'])
def extraction_guard_stats():
    """Cumulative stats for this process's KB-extraction guard."""
    if extraction_guard is None:
        return jsonify({"enabled": False}), 200
    return jsonify({"enabled": True, **extraction_guard.get_stats()}), 200


@app.route('/extraction_guard/reset', methods=['POST'])
def extraction_guard_reset():
    """Clear per-caller guard state only (no corpus reload) -- for evaluation between runs."""
    global extraction_guard
    if extraction_guard is not None:
        extraction_guard = _build_extraction_guard()
    return jsonify({"status": "guard_reset", "enabled": extraction_guard is not None}), 200


@app.route('/reset', methods=['POST'])
def reset_retriever():
    """Reset the retriever to its original clean state (removes all poisoned docs)."""
    global poison_filter, grounding_farm_detector, sentence_length_detector, extraction_guard
    config_path = os.getenv('CONFIG_PATH', 'configs/config.yaml')
    config = load_config(config_path)
    initialize_retriever(config)
    if POISON_FILTER_AVAILABLE and POISON_FILTER_ENABLED:
        poison_filter = PoisonFilter()  # fresh near-dup history/stats for the new clean corpus
    if GROUNDING_FARM_DETECTOR_AVAILABLE and GROUNDING_FARM_DEFENSE_ENABLED:
        grounding_farm_detector = GroundingFarmDetector()  # fresh stats
        sentence_length_detector = SentenceLengthDetector()  # fresh stats
    if extraction_guard is not None:
        extraction_guard = _build_extraction_guard()  # fresh per-caller state
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
        k = min(int(data.get("k", 5)), 5)
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
        
        # Perform search -- through the KB-extraction guard for any non-orchestrator caller.
        decision = None
        guarded = extraction_guard is not None and not _is_orchestrator(request)
        k = max(1, k)
        if guarded:
            cid = _client_id(request)
            decision = extraction_guard.decide(cid, k)
            # FREEZE needs the full top-k so already-served documents can still be re-read.
            k_eff = k if decision.k is None else min(k, decision.k)
            results = retriever.search(query_text, k=k if decision.allowed_ids is not None else k_eff)
            results = extraction_guard.apply(decision, results)
            q_emb = None
            if extraction_guard.cfg.dispersion_thresh is not None:
                q_emb = retriever.model.encode([query_text], convert_to_numpy=True, show_progress_bar=False)[0]
            extraction_guard.observe(cid, [r["id"] for r in results], q_emb)
            if decision.action != ALLOW:
                logger.warning(f"[GUARD] client={cid} action={decision.action} reason={decision.reason} "
                               f"signals={ {k_: round(v_, 2) for k_, v_ in decision.signals.items()} }")
        else:
            results = retriever.search(query_text, k=k)
        
        response = {"results": results}
        if signature:
            response["signature"] = signature
        
        resp = jsonify(response)
        if guarded and EXTRACTION_GUARD_DEBUG:
            resp.headers["X-Guard-Action"] = decision.action
            resp.headers["X-Guard-Reason"] = decision.reason
        return resp, 200
        
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
