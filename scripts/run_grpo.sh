#!/usr/bin/env bash
# Run GRPO training with TRL
# Run on GPU server (2x RTX 4090)

set -euo pipefail

source ~/miniconda3/bin/activate vlm-rl

MODEL="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
SFT_CHECKPOINT="${2:-checkpoints/sft}"

echo "=== GRPO Training ==="
echo "Model: $MODEL"
echo "SFT checkpoint: $SFT_CHECKPOINT"

# Check if SFT checkpoint exists
if [ ! -d "$SFT_CHECKPOINT" ]; then
    echo "WARNING: SFT checkpoint not found at $SFT_CHECKPOINT"
    echo "Training from base model instead."
    SFT_ARG=""
else
    SFT_ARG="--sft-checkpoint $SFT_CHECKPOINT"
fi

# GRPO with TRL + DeepSpeed
accelerate launch \
    --config_file configs/accelerate_2x4090.yaml \
    -m src.training.grpo \
    --model "$MODEL" \
    --data-path data/processed/trl/geoqa \
    --output-dir checkpoints/grpo \
    $SFT_ARG \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 4 \
    --lr 1e-6 \
    --lora-r 16 \
    --num-generations 8 \
    --max-completion-length 1024 \
    --beta 0.0 \
    --temperature 1.3

echo ""
echo "=== GRPO training complete ==="
echo "Checkpoint: checkpoints/grpo/"

# Evaluate GRPO model
echo ""
echo "--- Evaluating GRPO model ---"
python -m src.eval.baseline \
    --model "$MODEL" \
    --lora-path checkpoints/grpo \
    --dataset data/processed/trl/geoqa \
    --output results/grpo_eval.json
