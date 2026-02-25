#!/usr/bin/env bash
# Stage B: Distill a local Process Reward Model
#
# Steps:
# 1. Generate trajectories from GRPO model
# 2. Score them with Reward LM API
# 3. Train local PRM on scored data

set -euo pipefail

source ~/miniconda3/bin/activate vlm-rl

MODEL="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
GRPO_CHECKPOINT="${2:-checkpoints/grpo}"
PROVIDER="${3:-anthropic}"
MAX_SCORE_SAMPLES="${4:-5000}"

echo "=== Stage B: Local PRM Distillation ==="

# Step 1: Generate trajectories
echo ""
echo "--- Step 1: Generating trajectories ---"
LORA_ARG=""
if [ -d "$GRPO_CHECKPOINT" ]; then
    LORA_ARG="--lora-path $GRPO_CHECKPOINT"
fi

python -m src.training.generate_trajectories \
    --model "$MODEL" \
    $LORA_ARG \
    --dataset data/processed/trl/geoqa \
    --output data/processed/dpo_pairs.json \
    --n 4 \
    --provider "$PROVIDER"

# Step 2: Score traces with API (if not already done by generate_trajectories)
echo ""
echo "--- Step 2: Scoring traces with API ---"
python -m src.training.distill_prm score \
    --input data/processed/dpo_pairs.trajectories.json \
    --output data/processed/scored_traces.json \
    --provider "$PROVIDER" \
    --max-samples "$MAX_SCORE_SAMPLES"

# Step 3: Train local PRM
echo ""
echo "--- Step 3: Training local PRM ---"
python -m src.training.distill_prm train \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --scored-data data/processed/scored_traces.json \
    --output-dir checkpoints/prm \
    --epochs 3 \
    --batch-size 2 \
    --lr 1e-5

echo ""
echo "=== PRM distillation complete ==="
echo "Checkpoint: checkpoints/prm/"
