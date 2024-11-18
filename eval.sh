#!/bin/bash

# Define the paths to datasets, models, and output
CACHE_DIR="/home/ahmed.attia/Documents/DeTikZify/cache"
TRAINSET_PATH="/home/ahmed.attia/Documents/DeTikZify/detikzify/datikz/data/train-*.parquet"
TESTSET_PATH="/home/ahmed.attia/Documents/DeTikZify/detikzify/datikz/data/50-test-00000-of-00001.parquet"
OUTPUT_PATH="/home/ahmed.attia/Documents/DeTikZify/results/scores.json"
TIMEOUT=10  # Set timeout in seconds
MODELS=(
    "ds-1.3b-mcts-dng=nllg/detikzify-ds-1.3b"
)
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
# Construct the model path arguments
MODEL_ARGS=""
for MODEL in "${MODELS[@]}"; do
    MODEL_ARGS+="--path $MODEL "
done

# Run the evaluation script
python examples/eval.py \
    --cache_dir "$CACHE_DIR" \
    --trainset "$TRAINSET_PATH"\
    --testset "$TESTSET_PATH" \
    --output "$OUTPUT_PATH" \
    --timeout "$TIMEOUT" \
    $USE_SKETCHES \
    $MODEL_ARGS
