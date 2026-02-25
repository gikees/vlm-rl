#!/usr/bin/env bash
# Run SFT warm-up training
# Run on GPU server (2x RTX 4090)

set -euo pipefail

source ~/miniconda3/bin/activate vlm-rl

MODEL="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"

echo "=== SFT Warm-up Training ==="
echo "Model: $MODEL"

# Use accelerate for multi-GPU with DeepSpeed
accelerate launch \
    --config_file configs/accelerate_2x4090.yaml \
    -m src.training.sft \
    --model "$MODEL" \
    --data-path data/processed/trl/geoqa \
    --output-dir checkpoints/sft \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-5 \
    --lora-r 16 \
    --max-seq-length 2048

echo ""
echo "=== SFT training complete ==="
echo "Checkpoint: checkpoints/sft/"

# Evaluate SFT model
echo ""
echo "--- Evaluating SFT model ---"
python -m src.eval.baseline \
    --model "$MODEL" \
    --lora-path checkpoints/sft \
    --dataset data/processed/trl/geoqa \
    --output results/sft_eval.json
