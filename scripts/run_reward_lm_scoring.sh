#!/usr/bin/env bash
# Score reasoning traces with Reward LM for error analysis
# and offline DPO training data generation
# Run after baseline/GRPO evaluation

set -euo pipefail

source ~/miniconda3/bin/activate vlm-rl

RESULTS_FILE="${1:-results/grpo_eval.json}"
MAX_SAMPLES="${2:-50}"
PROVIDER="${3:-anthropic}"

echo "=== Reward LM Error Analysis ==="
echo "Results: $RESULTS_FILE"
echo "Provider: $PROVIDER"
echo "Max samples: $MAX_SAMPLES"

python -m src.eval.analysis \
    --results "$RESULTS_FILE" \
    --output results/error_analysis.json \
    --max-samples "$MAX_SAMPLES" \
    --provider "$PROVIDER"

echo ""
echo "=== Analysis complete ==="
echo "See results/error_analysis.json for details"
