#!/usr/bin/env bash
# setup_processes.sh — bring up the full Reliable-dRAG stack as plain processes
# (no Docker). For GPU boxes where nested Docker/compose isn't available (vast.ai).
# Mirrors docker-compose.yml + drag_contract/entrypoint.sh, as host processes.
#
# Place this at the REPO ROOT (next to docker-compose.yml). Then:
#   export HF_TOKEN=hf_xxx          # needed for gated models (Gemma/Llama); Qwen is open
#   chmod +x setup_processes.sh
#   ./setup_processes.sh            # brings the stack up, stays running in background
#   ./setup_processes.sh stop       # stops all stack processes
#
# Once it prints "STACK UP", run the sweeps (stack already up => SKIP_DOCKER=1):
#   SKIP_DOCKER=1 SEED=42 ./run_tanzim_experiments.sh
#
# Per-service logs land in ./logs/.  Watch vLLM boot with: tail -f logs/llm.log
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
mkdir -p logs
PYBIN="python3"
export PYTHONPATH="$REPO/drag_python_client:$REPO:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$REPO/.hf-cache}"     # keep model/dataset cache on the big disk
API_KEY="reliable-derag-secret-2026"              # matches docker-compose.yml

if [ "${1:-}" = "stop" ]; then
  pkill -f "hardhat node" || true
  pkill -f "app/server.py" || true
  echo "stopped stack processes"; exit 0
fi

wait_http () {  # url name [tries]
  local url="$1" name="$2" tries="${3:-80}" i=0
  until curl -s -o /dev/null "$url"; do
    i=$((i+1)); [ "$i" -ge "$tries" ] && { echo "ERROR: $name never came up at $url (see logs/)"; return 1; }
    sleep 3
  done
  echo "  $name is up ($url)"
}

echo "== [1/6] system deps (Node 20, build tools) =="
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
apt-get update -y && apt-get install -y --no-install-recommends git curl build-essential ninja-build || true

echo "== [2/6] python deps (vLLM installed last, mirroring the Dockerfile) =="
$PYBIN -m pip install --upgrade pip
# If pip errors with 'externally-managed-environment', append --break-system-packages to each line.
$PYBIN -m pip install -r drag_data_source/requirements.txt
$PYBIN -m pip install -r drag_python_client/requirements.txt
$PYBIN -m pip install -r drag_llm_service/requirements.txt
$PYBIN -m pip install -r drag_llm_service/requirements-vllm.txt

echo "== [3/6] hardhat node + contract deploy + on-chain score seed =="
( cd drag_contract && npm install )
( cd drag_contract && nohup npx hardhat node --hostname 127.0.0.1 > "$REPO/logs/hardhat.log" 2>&1 & )
until curl -s -X POST -H 'Content-Type: application/json' \
      --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
      http://localhost:8545 >/dev/null 2>&1; do sleep 2; done
echo "  hardhat RPC ready"; sleep 3
( cd drag_contract && npm run deploy:local )
$PYBIN -m drag_python_client.examples.test_local test_default_sources

echo "== [4/6] generate the three per-source configs (repo ships only config.yaml) =="
gen_cfg () {  # name port
  cat > "drag_data_source/configs/config_$1.yaml" <<YAML
data:
  jsonl_path: "$REPO/data/polluted_token/$1.jsonl"
  dataset_name: "$1"
blockchain:
  address: null
  private_key: null
  provider_url: "http://localhost:8545"
  contract_address: null
retriever:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  normalize: true
  use_faiss: true
  device: "cpu"
  batch_size: 256
server:
  port: $2
  host: "0.0.0.0"
YAML
}
gen_cfg sources_0 8001
gen_cfg sources_20 8002
gen_cfg sources_100 8003

echo "== [5/6] launch the 3 data sources =="
start_ds () {  # name port privkey
  ( cd drag_data_source && \
    CONFIG_PATH="$REPO/drag_data_source/configs/config_$1.yaml" \
    API_KEY="$API_KEY" PRIVATE_KEY="$3" RATE_LIMIT_DEFAULT="600 per minute" \
    nohup $PYBIN app/server.py > "$REPO/logs/$1.log" 2>&1 & )
  wait_http "http://localhost:$2/health" "data-source $1"
}
start_ds sources_0   8001 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
start_ds sources_20  8002 0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d
start_ds sources_100 8003 0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a

echo "== [6/6] launch the LLM service (vLLM, uses the GPU) =="
( cd drag_llm_service && \
  HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
  API_KEY="$API_KEY" FLASK_DEBUG=false \
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  nohup $PYBIN app/server.py > "$REPO/logs/llm.log" 2>&1 & )
echo "  vLLM is pulling + loading the model from HuggingFace (first boot = several minutes)."
echo "  watch:  tail -f logs/llm.log"
wait_http "http://localhost:9000/health" "llm-service" 200

echo
echo "STACK UP."
echo "  health: curl localhost:9000/health ; curl localhost:8001/info"
echo "  run:    SKIP_DOCKER=1 SEED=42 ./run_tanzim_experiments.sh"
echo "  stop:   ./setup_processes.sh stop"
