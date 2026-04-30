#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DRAG_DIR=$(dirname $SCRIPT_DIR)

cd $DRAG_DIR

# Run baseline (clean network)
echo "Running baseline evaluation..."
python simulator.py \
    --config="./config/rag.yaml" \
    --config="./config/llm/llama32_3b.yaml" \
    --config="./config/data/mmlu.yaml" \
    --config="./config/security.yaml" \
    --security.enable_attack=False \
    --rag.test_mode=False \
    --log_level=INFO

# Run with data poisoning attack - varying ratios
for ratio in 0.1 0.2 0.3; do
    echo "Running attack with poisoning ratio: $ratio"
    python simulator.py \
        --config="./config/rag.yaml" \
        --config="./config/llm/llama32_3b.yaml" \
        --config="./config/data/mmlu.yaml" \
        --config="./config/security.yaml" \
        --security.enable_attack=True \
        --security.poisoning_ratio=$ratio \
        --security.attack_strategy='random' \
        --security.poison_type='wrong_answer' \
        --rag.test_mode=False \
        --log_level=INFO
    sleep 2
done

# Run with different attack strategies
for strategy in 'high_degree' 'topic_specific'; do
    echo "Running attack with strategy: $strategy"
    python simulator.py \
        --config="./config/rag.yaml" \
        --config="./config/llm/llama32_3b.yaml" \
        --config="./config/data/mmlu.yaml" \
        --config="./config/security.yaml" \
        --security.enable_attack=True \
        --security.poisoning_ratio=0.2 \
        --security.attack_strategy=$strategy \
        --security.poison_type='wrong_answer' \
        --rag.test_mode=False \
        --log_level=INFO
    sleep 2
done

echo "All attack simulations completed!"