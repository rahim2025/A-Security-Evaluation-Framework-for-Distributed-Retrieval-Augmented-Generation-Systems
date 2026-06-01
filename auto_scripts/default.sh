#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DRAG_DIR=$(dirname $SCRIPT_DIR)

cd $DRAG_DIR
nohup python simulator.py \
    --config="./config/rag.yaml" \
    --config="./config/llm/llama32_3b.yaml" \
    --config="./config/data/mmlu.yaml" \
    --rag.test_mode=False \
    --log_level=INFO > $DRAG_DIR/logs/script.log 2>&1 &
