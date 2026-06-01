#!/bin/bash
set -e

echo "Starting Hardhat node..."

# Start Hardhat node in background
cd /app/drag_contract
npx hardhat node --hostname 0.0.0.0 > /tmp/hardhat.log 2>&1 &
HARDHAT_PID=$!

echo "Waiting for Hardhat node to be ready..."
# Wait for Hardhat RPC endpoint to be ready
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -X POST -H "Content-Type: application/json" \
        --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
        http://localhost:8545 > /dev/null 2>&1; then
        echo "Hardhat node is ready!"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for Hardhat node... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Error: Hardhat node failed to start within timeout"
    cat /tmp/hardhat.log
    exit 1
fi

# Give it a moment to fully initialize
sleep 3

echo "Deploying contract..."
# Deploy the contract
npm run deploy:local || {
    echo "Error: Contract deployment failed"
    cat /tmp/hardhat.log
    exit 1
}

echo "Contract deployed successfully!"

echo "Running Python initialization..."
# Run Python initialization
# Set PYTHONPATH to include drag_python_client parent directory
export PYTHONPATH="/app/drag_python_client:${PYTHONPATH}"
cd /app
python3 -m drag_python_client.examples.test_local test_default_sources || {
    echo "Error: Python initialization failed"
    exit 1
}

echo "Initialization complete! Hardhat node is running on port 8545"
echo "Logs are available in /tmp/hardhat.log"

# Keep Hardhat node running
wait $HARDHAT_PID

