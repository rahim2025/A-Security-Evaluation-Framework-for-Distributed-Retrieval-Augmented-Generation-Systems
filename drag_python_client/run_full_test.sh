#!/bin/bash

# Script to spin up Hardhat, deploy contract, and run Python tests
# Usage: ./run_full_test.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTRACT_DIR="$PROJECT_ROOT/drag_contract"
PYTHON_CLIENT_DIR="$PROJECT_ROOT/drag_python_client"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ ! -z "$HARDHAT_PID" ]; then
        echo "Stopping Hardhat node (PID: $HARDHAT_PID)..."
        kill $HARDHAT_PID 2>/dev/null || true
        wait $HARDHAT_PID 2>/dev/null || true
    fi
    # Kill any remaining hardhat processes
    pkill -f "hardhat node" 2>/dev/null || true
    echo -e "${GREEN}Cleanup complete.${NC}"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Check if directories exist
if [ ! -d "$CONTRACT_DIR" ]; then
    echo -e "${RED}Error: drag_contract directory not found at $CONTRACT_DIR${NC}"
    exit 1
fi

if [ ! -d "$PYTHON_CLIENT_DIR" ]; then
    echo -e "${RED}Error: drag_python_client directory not found at $PYTHON_CLIENT_DIR${NC}"
    exit 1
fi

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed${NC}"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if node_modules exists in contract dir
if [ ! -d "$CONTRACT_DIR/node_modules" ]; then
    echo -e "${YELLOW}Installing npm dependencies...${NC}"
    cd "$CONTRACT_DIR"
    npm install
fi

# Check if Python venv exists
if [ ! -d "$PYTHON_CLIENT_DIR/venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    cd "$PYTHON_CLIENT_DIR"
    python3 -m venv venv
fi

# Activate Python venv
echo -e "${GREEN}Activating Python virtual environment...${NC}"
source "$PYTHON_CLIENT_DIR/venv/bin/activate"

# Install Python dependencies if needed
if [ ! -f "$PYTHON_CLIENT_DIR/venv/.installed" ]; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    cd "$PYTHON_CLIENT_DIR"
    pip install -r requirements.txt
    touch "$PYTHON_CLIENT_DIR/venv/.installed"
fi

# Step 1: Start Hardhat node
echo -e "${GREEN}Starting Hardhat node...${NC}"
cd "$CONTRACT_DIR"
npm run node > /tmp/hardhat_node.log 2>&1 &
HARDHAT_PID=$!

# Wait for Hardhat to be ready
echo -e "${YELLOW}Waiting for Hardhat node to be ready...${NC}"
MAX_WAIT=30
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://127.0.0.1:8545 > /dev/null 2>&1; then
        echo -e "${GREEN}Hardhat node is ready!${NC}"
        break
    fi
    sleep 1
    WAITED=$((WAITED + 1))
    echo -n "."
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "\n${RED}Error: Hardhat node did not start in time${NC}"
    echo "Check logs at /tmp/hardhat_node.log"
    exit 1
fi

# Give it a bit more time to fully initialize
sleep 2

# Step 2: Deploy contract
echo -e "${GREEN}Deploying contract...${NC}"
cd "$CONTRACT_DIR"
if ! npm run deploy:local; then
    echo -e "${RED}Error: Contract deployment failed${NC}"
    exit 1
fi

echo -e "${GREEN}Contract deployed successfully!${NC}"

# Step 3: Run Python test
echo -e "${GREEN}Running Python test...${NC}"
cd "$PYTHON_CLIENT_DIR"
python3 -m drag_python_client.examples.test_local

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ All tests passed!${NC}"
else
    echo -e "\n${RED}✗ Tests failed${NC}"
    exit 1
fi

