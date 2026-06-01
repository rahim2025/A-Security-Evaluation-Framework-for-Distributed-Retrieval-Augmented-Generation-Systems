# Drag LLM Service

A Dockerized LLM orchestrator service for the DRAG decentralized RAG system. This service coordinates multiple data source retrievers, performs reranking, conducts sentence-level importance analysis (MC-Shapley or RORA), and integrates with blockchain for source score management.

## Overview

This service provides REST APIs that:
- Query multiple data sources for documents in parallel
- Sample sources based on usefulness scores from blockchain
- Rerank retrieved documents with reliability weighting
- Generate LLM responses using local or OpenAI models
- Perform sentence-level importance analysis (MC-Shapley or RORA)
- Update source reliability and usefulness scores on blockchain
- Stream real-time progress updates via Server-Sent Events (SSE)
- Retrieve and monitor score update events from blockchain

## Architecture

```
┌─────────────────┐
│   LLM Service   │
│  (Port 9000)    │
└────────┬────────┘
         │
         ├───────┬──────────┬─────────────┐
         │       │          │             │
┌────────▼──┐ ┌──▼─────┐ ┌─▼──────┐ ┌───▼──────┐
│ Data Src  │ │Data Src│ │Data Src│ │    ...   │
│ (Port 8001│ │(8002)  │ │(8003)  │ │          │
└───────────┘ └────────┘ └────────┘ └──────────┘
```

## Configuration

Edit `configs/config.yaml` to configure:

### Model Configuration
- **type**: "local" (HuggingFace) or "openai" (API)
- **local_model_name**: HuggingFace model identifier
- **HF_TOKEN / HUGGINGFACE_HUB_TOKEN**: required for gated HuggingFace models (e.g., Llama)
- **openai_model_name**: API model name
- **openai_api_key**: API key (or use environment variable)

### Data Sources
List of data source services with their URLs:
```yaml
data_sources:
  - name: "sources_0"
    url: "http://localhost:8001"
```

### Retrieval Settings
- **n_retrievers**: Number of sources to query
- **n_contexts**: Documents per source
- **top_k**: Final documents after reranking
- **sample_with_usefulness**: Whether to sample based on usefulness
- **rerank_with_reliability**: Whether to use reliability in reranking

## API Endpoints

### Health Check

**Endpoint:** `GET /health`

Returns basic service health status.

**Response:**
```json
{
  "status": "healthy"
}
```

### Data Sources Health Check

**Endpoint:** `GET /health/data_sources`

Verifies connectivity to all configured data sources and returns their individual status.

**Response:**
```json
{
  "status": "healthy" | "degraded" | "unhealthy",
  "data_sources": {
    "source_name": {
      "status": "healthy" | "unreachable" | "error",
      "url": "http://...",
      "message": "...",
      "dataset": "dataset_name"
    }
  }
}
```

### Score Events

**Endpoint:** `GET /score_events`

Retrieves score update events from the blockchain contract.

**Query Parameters:**
- `source_id` (optional): Filter by source ID
- `source_address` (optional): Filter by source address
- `from_block` (optional): Starting block number (inclusive)
- `to_block` (optional): Ending block number (inclusive)

**Response:**
```json
{
  "events": [
    {
      "sourceAddress": "0x...",
      "sourceID": "sources_0",
      "sourceName": "sources_0",
      "reliabilityScore": 10000,
      "usefulnessScore": 10000,
      "timestamp": 1234567890,
      "info": "...",
      "blockNumber": 12345,
      "transactionHash": "0x...",
      "logIndex": 0
    }
  ]
}
```

### Query (Simple Response)

**Endpoint:** `POST /query`

Performs a query and returns just the response text. This endpoint:
- Samples data sources based on usefulness scores (if configured)
- Queries selected data sources with blockchain scores
- Reranks retrieved documents (optionally with reliability weighting)
- Generates LLM response

**Request Body:**
```json
{
  "query": "your question here"
}
```

**Response:**
```json
{
  "response": "answer text"
}
```

### Query Analyze (Full Analysis)

**Endpoint:** `POST /query_analyze`

Performs a query with full analysis including sentence-level importance scoring and optional blockchain score updates.

**Query Parameters:**
- `stream` (optional): If set to `true`, returns Server-Sent Events (SSE) stream instead of JSON

**Request Body:**
```json
{
  "query": "your question here",
  "ground_truth": ["expected answer 1", "expected answer 2"],
  "update_scores": true
}
```

**Parameters:**
- `query` (required): Search query string
- `ground_truth` (optional): List of expected answers for correctness evaluation
- `update_scores` (optional): Whether to update blockchain scores based on query results (default: `false`)

**Response (Non-streaming):**
```json
{
  "response": "answer text",
  "correctness": true,
  "sampled_sources": ["sources_0", "sources_20"],
  "importance_score": [["sources_0", 0.85], ["sources_20", 0.67]],
  "updated_scores": {
    "sources_0": {
      "reliability": 10500,
      "usefulness": 10200
    }
  }
}
```

**Response (Streaming, `?stream=true`):**

Returns Server-Sent Events (SSE) with the following event types:
- `sampling_sources`: Data sources selected for querying
- `querying_sources`: Progress of querying each data source
- `reranking`: Reranking status
- `mc_shap`: MC-Shapley analysis progress (if enabled)
- `mc_shap_baseline`: Baseline response from MC-Shapley
- `mc_shap_progress`: Progress updates during analysis
- `computing_importance`: Computing importance scores
- `updating_scores`: Blockchain score update status
- `final_result`: Final result with all analysis data
- `error`: Error events

**Example SSE Events:**
```
event: sampling_sources
data: {"sources": ["sources_0", "sources_20"]}

event: querying_sources
data: {"source": "sources_0", "status": "querying"}

event: mc_shap_progress
data: {"count": 50, "total": 100, "progress": 0.5}

event: final_result
data: {"response": "...", "correctness": true, ...}
```

## Building and Running

### Prerequisites

Make sure data source services are running on ports 8001, 8002, 8003.

### Build and Run

```bash
# Build the Docker image
docker build -t drag-llm-service .

# Enable vLLM (GPU-only)
docker build --build-arg INSTALL_VLLM=true -t drag-llm-service .

# Run the container
docker run -d \
  -p 9000:9000 \
  --name drag-llm-service \
  drag-llm-service
```

Or use docker-compose:

```bash
docker-compose up --build -d

# Enable vLLM (GPU-only)
INSTALL_VLLM=true docker-compose up --build -d
```

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app/server.py
```

## Testing

### Test Service

```bash
python test_service.py
```

### Manual Testing

**Health check:**
```bash
curl http://localhost:9000/health
```

**Data sources health check:**
```bash
curl http://localhost:9000/health/data_sources
```

**Get score events from blockchain:**
```bash
curl "http://localhost:9000/score_events?source_id=sources_0"
```

**Simple query:**
```bash
curl -X POST http://localhost:9000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

**Query with analysis (non-streaming):**
```bash
curl -X POST http://localhost:9000/query_analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AI?",
    "ground_truth": ["artificial intelligence"],
    "update_scores": false
  }'
```

**Query with analysis and score updates:**
```bash
curl -X POST http://localhost:9000/query_analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AI?",
    "ground_truth": ["artificial intelligence"],
    "update_scores": true
  }'
```

**Query with analysis (streaming SSE):**
```bash
curl -X POST "http://localhost:9000/query_analyze?stream=true" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AI?",
    "ground_truth": ["artificial intelligence"],
    "update_scores": false
  }'
```

## Features

1. **Multi-Source Retrieval**: Queries multiple data source services in parallel
2. **Smart Sampling**: Samples sources based on usefulness scores from blockchain
3. **Hybrid Reranking**: Combines dense and sparse retrieval with reliability weighting
4. **Blockchain Integration**: Retrieves and updates source scores on blockchain
5. **Sentence-Level Importance**: MC-Shapley or RORA analysis for interpretability
6. **Score Updates**: Automatically updates reliability and usefulness scores based on query results
7. **Streaming Support**: Server-Sent Events (SSE) for real-time progress updates
8. **Source Validation**: Validates source scores and signatures from data sources
9. **Health Monitoring**: Comprehensive health checks for service and data sources
10. **Event Tracking**: Retrieves score update events from blockchain contract

## Dependencies

Key dependencies:
- **vllm**: For local model inference
- **openai**: For API-based models
- **sentence-transformers**: For embeddings
- **scikit-learn**: For TF-IDF and similarity
- **requests**: For querying data sources
- **Flask**: For REST API

## Directory Structure

```
drag_llm_service/
├── app/
│   ├── __init__.py
│   └── server.py          # Flask API server
├── configs/
│   └── config.yaml        # Service configuration
├── src/
│   ├── models/
│   │   ├── open_model.py              # VLLMModel
│   │   ├── model.py                   # Model implementations
│   │   └── huggingface_wrapper_module.py
│   ├── retriever/
│   │   └── reranker.py                # Reranker
│   ├── mc_shap/
│   │   ├── base.py                    # LocalModel, OpenAIModel, BaseSHAP
│   │   ├── mc_shap.py                 # MC SHAP analysis
│   │   └── visualization.py
│   ├── collate_fns/
│   │   ├── collate_fn.py
│   │   └── rora_collate_fn.py
│   ├── metrics/
│   │   ├── accuracy.py
│   │   ├── loss.py
│   │   └── metric.py
│   ├── rora/
│   │   └── rora.py                    # RoRA implementation
│   ├── trainer/
│   │   ├── trainer.py
│   │   └── rora_trainer.py
│   └── utils/
│       ├── helper_functions.py
│       ├── drag_log_sol.py
│       ├── draglog_client.py
│       └── visualization.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── test_service.py
└── README.md
```

## Notes

- The service assumes data source services are running
- For Docker networking, update URLs in config to use container names
- Token SHAP analysis can be slow for long prompts
- Model initialization may take several minutes
