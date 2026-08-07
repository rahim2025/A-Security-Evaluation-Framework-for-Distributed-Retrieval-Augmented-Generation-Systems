#!/bin/bash
# Healthcheck script for Hardhat node with contract deployment verification

# Check if RPC endpoint is responding
if ! curl -s -X POST -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
    http://localhost:8545 > /dev/null 2>&1; then
    exit 1
fi

# Check if contract is deployed by verifying deployment file exists
DEPLOYMENT_FILE="/app/drag_contract/ignition/deployments/chain-31337/deployed_addresses.json"
if [ ! -f "$DEPLOYMENT_FILE" ]; then
    exit 1
fi

# Verify contract address exists in deployment file
if ! grep -q "DragScores" "$DEPLOYMENT_FILE" 2>/dev/null; then
    exit 1
fi

# Try to call the contract's hello() method to verify it's working
# This requires Python and the client library
export PYTHONPATH="/app/drag_python_client:${PYTHONPATH}"
if ! python3 -c "
import sys
sys.path.insert(0, '/app/drag_python_client')
try:
    from drag_python_client import DragScoresClient
    client = DragScoresClient(project_root='/app', provider_url='http://localhost:8545')
    result = client.hello()
    if result and len(result) > 0:
        sys.exit(0)
    else:
        sys.exit(1)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
    exit 1
fi

exit 0

