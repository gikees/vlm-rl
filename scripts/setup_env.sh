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

# Create venv
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA 12.1)
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
echo "Installing project dependencies..."
pip install -e .

# Install flash-attention (needed for efficient inference/training)
echo "Installing flash-attention..."
pip install flash-attn --no-build-isolation

# Install EasyR1
echo "Installing EasyR1..."
pip install easyr1

# Install TRL (fallback framework)
echo "Installing TRL..."
pip install trl

# Install vLLM (for fast generation during GRPO)
echo "Installing vLLM..."
pip install vllm

# Login to services
echo ""
echo "=== Manual steps ==="
echo "1. Log in to HuggingFace: huggingface-cli login"
echo "2. Log in to Weights & Biases: wandb login"
echo "3. Set API keys in .env if using Reward LM:"
echo "   export ANTHROPIC_API_KEY=..."
echo "   export DASHSCOPE_API_KEY=..."
echo ""
echo "Setup complete! Activate with: source .venv/bin/activate"
