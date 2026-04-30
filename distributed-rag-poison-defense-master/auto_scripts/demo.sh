#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DRAG_DIR=$(dirname $SCRIPT_DIR)
SCRIPT_LOG_FILE="$DRAG_DIR/logs/script.log"

# --- Default Parameter Settings ---
DEFAULT_CONFIGS=(
  "./config/rag.yaml"
  "./config/llm/llama32_3b.yaml"
  "./config/data/mmlu.yaml"
)
DEFAULT_RAG_TEST_MODE=False
DEFAULT_LOG_LEVEL=INFO

# --- Function to build the Python command ---
build_command() {
  local cmd="python simulator.py"

  # Add default configs
  for config in "${DEFAULT_CONFIGS[@]}"; do
    cmd+=" --config=\"$config\""
  done

  # Add default parameters
  cmd+=" --rag.test_mode=$DEFAULT_RAG_TEST_MODE"
  cmd+=" --log_level=$DEFAULT_LOG_LEVEL"

  # Add overrides (if any) from function arguments
  cmd+=" $@" 

  echo "$cmd"
}

# --- Function to run the experiment ---
run_experiment() {
  local command=$(build_command "$@")

  echo ""
  echo "# Running experiment:"
  echo "> Command: $command"
  echo ""

  echo "Below are the logs from the process..."
  eval "$command" # Use 'eval' to execute the built command
  sleep 1 # This is important.

}

main() {
  # --- Main execution block ---
  echo ""
  echo "##############################"
  echo "## All experiments launched ##"
  echo "##############################"
  echo ""

  cd "$DRAG_DIR"

  # --- Define and run experiments ---

  # Example 1: Basic run with default parameters
  run_experiment

  # Example 2: Varying search algorithms
  for algo in "TARW" "RW" "FL"; do
    run_experiment "--rag.search_algorithm=\"$algo\""
  done

  # Example 3: Varying number of peers
  for peers in 20 50 100; do
    run_experiment "--rag.num_peers=$peers"
  done

  # Example 4: Different data configurations
  for data_config in "./config/data/medical.yaml" "./config/data/news.yaml"; do
    run_experiment "--config=\"$data_config\""
  done

  # Example 5: Combinations (be cautious with too many combinations)
  for algo in "TARW" "RW"; do
    for peers in 20 50; do
       run_experiment "--rag.search_algorithm=\"$algo\"" "--rag.num_peers=$peers"
    done
  done

  echo ""
  echo "##############################"
  echo "## All experiments finished ##"
  echo "##############################"
  echo ""
}

rm -f "$SCRIPT_LOG_FILE" # Delete previous log if it exists.
main > "$SCRIPT_LOG_FILE" 2>&1 &
