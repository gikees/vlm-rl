#!/usr/bin/env bash
# Run full evaluation across all checkpoints and benchmarks
# Run on GPU server

set -euo pipefail

source .venv/bin/activate

MODEL="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
MAX_SAMPLES="${2:-}"

echo "=== Full Evaluation Suite ==="

EXTRA_ARGS=""
if [ -n "$MAX_SAMPLES" ]; then
    EXTRA_ARGS="--max-samples $MAX_SAMPLES"
fi

# Evaluate each checkpoint
for tag in baseline sft grpo; do
    echo ""
    echo "=== Evaluating: $tag ==="

    if [ "$tag" = "baseline" ]; then
        LORA_ARG=""
    elif [ "$tag" = "sft" ]; then
        if [ ! -d "checkpoints/sft" ]; then
            echo "Skipping SFT — checkpoint not found"
            continue
        fi
        LORA_ARG="--lora-path checkpoints/sft"
    elif [ "$tag" = "grpo" ]; then
        if [ ! -d "checkpoints/grpo" ]; then
            echo "Skipping GRPO — checkpoint not found"
            continue
        fi
        LORA_ARG="--lora-path checkpoints/grpo"
    fi

    python -m src.eval.benchmark \
        --model "$MODEL" \
        $LORA_ARG \
        --tag "$tag" \
        --output-dir results \
        $EXTRA_ARGS
done

echo ""
echo "=== All evaluations complete ==="
echo "Results saved in results/"
echo ""
echo "To compare results:"
echo "  python -m src.eval.analysis --compare baseline:results/summary_baseline.json sft:results/summary_sft.json grpo:results/summary_grpo.json"
