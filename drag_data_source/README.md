# Drag Data Source Services

A Dockerized retrieval service for the DRAG decentralized RAG system.

## Overview

This service provides REST APIs for document retrieval based on semantic similarity. It loads documents from JSONL files, creates embeddings, and allows querying for similar documents.

## Services

The docker-compose setup runs **three independent data source services**:

| Service | Container Name | Port | Dataset | Config File |
|---------|---------------|------|---------|-------------|
| data-source-0 | drag-data-source-0 | 8001 | sources_0.jsonl | config_sources_0.yaml |
| data-source-20 | drag-data-source-20 | 8002 | sources_20.jsonl | config_sources_20.yaml |
| data-source-100 | drag-data-source-100 | 8003 | sources_100.jsonl | config_sources_100.yaml |

## Configuration

Each service uses its own YAML configuration file in `configs/`:
- `config_sources_0.yaml` - sources_0.jsonl
- `config_sources_20.yaml` - sources_20.jsonl
- `config_sources_100.yaml` - sources_100.jsonl

Each config specifies:
- **data.jsonl_path**: Path to the JSONL file containing documents
- **data.dataset_name**: Name/prefix for this data source
- **retriever**: Model and retrieval settings
- **server**: Host and port for the API

## Building and Running

### Quick Start

Start all three services:

```bash
docker-compose up --build -d
```

View logs:

```bash
docker-compose logs -f
```

Stop services:

```bash
docker-compose down
```

### Individual Service Management

Start a specific service:

```bash
docker-compose up data-source-0
```

View logs for one service:

```bash
docker logs -f drag-data-source-0
```

## API Endpoints

All services expose the same REST API:

### Health Check

```bash
GET /health
```

Returns service status and dataset name.

Example:
```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "dataset": "sources_0",
  "status": "healthy"
}
```

### Query Documents

```bash
POST /query
Content-Type: application/json

{
  "query": "your search query",
  "k": 10
}
```

Returns top-k most similar documents with scores.

Example:
```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "k": 5}'
```

Response:
```json
{
  "results": [
    {
      "rank": 1,
      "id": "doc_id",
      "score": 0.95,
      "text": "document text",
      "meta": {
        "dataset_name": "sources_0",
        "record_id": "doc_id"
      }
    },
    ...
  ]
}
```

## Testing

### Test Individual Service

```bash
python test_service.py http://localhost:8001
```

### Test All Services

```bash
python test_all_services.py
```

Example output:
```
data-source-0 (port 8001):
✓ Health: healthy - Dataset: sources_0
✓ Query successful: Found 3 results

data-source-20 (port 8002):
✓ Health: healthy - Dataset: sources_20
✓ Query successful: Found 3 results

data-source-100 (port 8003):
✓ Health: healthy - Dataset: sources_100
✓ Query successful: Found 3 results
```

## Architecture

### Directory Structure

```
drag_data_source/
├── app/
│   ├── __init__.py
│   └── server.py          # Flask API server
├── configs/
│   ├── config.yaml        # Default config
│   ├── config_sources_0.yaml
│   ├── config_sources_20.yaml
│   └── config_sources_100.yaml
├── src/
│   ├── __init__.py
│   └── retriever/
│       ├── __init__.py
│       └── retriever.py   # FastRetriever
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── test_service.py        # Test single service
├── test_all_services.py   # Test all services
└── README.md
```

### Key Features

- **Multiple Configurations**: Each service uses its own config via `CONFIG_PATH` environment variable
- **Independent Containers**: Each service runs in its own isolated container
- **Shared Volume**: All services mount the same data directory
- **Semantic Search**: Uses SentenceTransformer + FAISS for fast retrieval
- **REST API**: Simple JSON-based interface

### Data Flow

1. Container starts → loads YAML config
2. Reads JSONL file from mounted volume
3. Creates embeddings using SentenceTransformer
4. Builds FAISS index for fast search
5. Starts Flask server on configured port
6. Handles `/health` and `/query` requests

## Development

To run locally without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Set config path and run
export CONFIG_PATH=configs/config_sources_0.yaml
python app/server.py
```

## Monitoring

Check container status:

```bash
docker ps --filter name=drag-data-source
```

View all logs:

```bash
docker-compose logs --tail=100
```

View specific service logs:

```bash
docker logs drag-data-source-0 --tail=100 -f
```

## Troubleshooting

### Services not starting

Check logs:
```bash
docker-compose logs
```

### Port conflicts

Edit `docker-compose.yml` to change port mappings.

### Out of memory

Reduce batch size in config files or use a smaller model.

### Slow startup

Embedding ~3000 documents takes ~3-5 minutes. This is normal for the first startup.
