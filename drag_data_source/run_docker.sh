#!/bin/bash
# Quick script to build and run the data source service

set -e

echo "Building Docker image..."
docker build -t drag-data-source .

echo ""
echo "Starting container..."
docker run -d \
  -p 8001:8001 \
  -v "$(pwd)/../drag/data:/data" \
  --name drag-data-source-1 \
  --rm \
  drag-data-source

echo ""
echo "Service started! Container: drag-data-source-1"
echo "Health check: curl http://localhost:8001/health"
echo "Test query: python test_service.py"
echo ""
echo "To view logs: docker logs -f drag-data-source-1"
echo "To stop: docker stop drag-data-source-1"
