#!/usr/bin/env bash
# Download and preprocess all datasets
# Run on GPU server after setup_env.sh

set -euo pipefail

echo "=== Downloading Datasets ==="

# Activate venv
source .venv/bin/activate

# Download training datasets
echo "Downloading training datasets..."
python -m src.data.download --training-only --output-dir data/raw

# Download eval datasets
echo "Downloading evaluation datasets..."
python -m src.data.download --dataset mathvista --output-dir data/raw
python -m src.data.download --dataset hallusionbench --output-dir data/raw

# Preprocess into training formats
echo ""
echo "=== Preprocessing Datasets ==="
python -m src.data.prepare --raw-dir data/raw --output-dir data/processed

echo ""
echo "=== Dataset Summary ==="
echo "Raw data:       data/raw/"
echo "TRL format:     data/processed/trl/"
echo "EasyR1 format:  data/processed/verl/"
echo ""
echo "Done! Ready for training."
