#!/usr/bin/env bash
# Environment setup for GPU server (2x RTX 4090)
# Run once after cloning the repo

set -euo pipefail

echo "=== VLM-RL Environment Setup ==="

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Are you on a GPU server?"
fi
nvidia-smi || true

# Create conda env from environment.yaml
echo "Creating conda environment..."
conda env create -f environment.yaml

echo ""
echo "=== Manual steps ==="
echo "1. Activate: source ~/miniconda3/bin/activate vlm-rl"
echo "2. Log in to HuggingFace: huggingface-cli login"
echo "3. Log in to Weights & Biases: wandb login"
echo "4. Set API keys in .env if using Reward LM:"
echo "   export ANTHROPIC_API_KEY=..."
echo "   export DASHSCOPE_API_KEY=..."
echo ""
echo "Setup complete! Activate with: source ~/miniconda3/bin/activate vlm-rl"
