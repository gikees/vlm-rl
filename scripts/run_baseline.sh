#!/usr/bin/env bash
# Run baseline evaluations (zero-shot + CoT)
# Run on GPU server

set -euo pipefail

source .venv/bin/activate

MODEL="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
MAX_SAMPLES="${2:-}"  # leave empty for full eval

echo "=== Baseline Evaluation ==="
echo "Model: $MODEL"

EXTRA_ARGS=""
if [ -n "$MAX_SAMPLES" ]; then
    EXTRA_ARGS="--max-samples $MAX_SAMPLES"
fi

# Zero-shot (no CoT)
echo ""
echo "--- Zero-shot (no CoT) ---"
python -m src.eval.baseline \
    --model "$MODEL" \
    --dataset data/processed/trl/geoqa \
    --output results/baseline_zeroshot.json \
    --no-cot \
    $EXTRA_ARGS

# With CoT prompting
echo ""
echo "--- With Chain-of-Thought ---"
python -m src.eval.baseline \
    --model "$MODEL" \
    --dataset data/processed/trl/geoqa \
    --output results/baseline_cot.json \
    $EXTRA_ARGS

# Best-of-N (K=8)
echo ""
echo "--- Best-of-8 ---"
python -m src.eval.best_of_n \
    --model "$MODEL" \
    --dataset data/processed/trl/geoqa \
    --output results/baseline_bon8.json \
    --n 8 \
    $EXTRA_ARGS

echo ""
echo "=== Baseline evaluation complete ==="
echo "Results in results/"
