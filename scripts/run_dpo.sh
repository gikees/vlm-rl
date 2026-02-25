#!/usr/bin/env bash
# Run offline DPO training with Reward LM-scored preference pairs
# Run after generating trajectories and scoring them

set -euo pipefail

source ~/miniconda3/bin/activate vlm-rl

MODEL="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
SFT_CHECKPOINT="${2:-checkpoints/sft}"
PAIRS_PATH="${3:-data/processed/dpo_pairs.json}"

echo "=== DPO Training (Stage A) ==="
echo "Model: $MODEL"
echo "Pairs: $PAIRS_PATH"

if [ ! -f "$PAIRS_PATH" ]; then
    echo "ERROR: DPO pairs not found at $PAIRS_PATH"
    echo "Run the trajectory generation first:"
    echo "  python -m src.training.generate_trajectories --help"
    exit 1
fi

SFT_ARG=""
if [ -d "$SFT_CHECKPOINT" ]; then
    echo "Starting from SFT checkpoint: $SFT_CHECKPOINT"
    SFT_ARG="--sft-checkpoint $SFT_CHECKPOINT"
else
    echo "No SFT checkpoint found, starting from base model"
fi

accelerate launch \
    --config_file configs/accelerate_2x4090.yaml \
    -m src.training.dpo \
    --model "$MODEL" \
    --pairs-path "$PAIRS_PATH" \
    --output-dir checkpoints/dpo \
    $SFT_ARG \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 5e-7 \
    --lora-r 16 \
    --beta 0.1

echo ""
echo "=== DPO training complete ==="
echo "Checkpoint: checkpoints/dpo/"

# Evaluate DPO model
echo ""
echo "--- Evaluating DPO model ---"
python -m src.eval.baseline \
    --model "$MODEL" \
    --lora-path checkpoints/dpo \
    --dataset data/processed/trl/geoqa \
    --output results/dpo_eval.json
