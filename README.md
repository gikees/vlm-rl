# VLM-RL: Reinforcement Learning for Visual Reasoning

Fine-tuning Qwen2.5-VL-7B with GRPO to improve visual chain-of-thought reasoning, using both outcome rewards and a Reward Language Model for process-level reward signals.

## Setup

```bash
# On GPU server (2x RTX 4090)
bash scripts/setup_env.sh

# Download datasets
bash scripts/download_data.sh
```

## Training Pipeline

### 1. Baseline evaluation
```bash
bash scripts/run_baseline.sh
```

### 2. SFT warm-up
```bash
bash scripts/run_sft.sh
```

### 3. GRPO training
```bash
bash scripts/run_grpo.sh
```

### 4. Full evaluation
```bash
bash scripts/run_eval.sh
```

## Project Structure

```
configs/          Training configs (EasyR1 YAML)
src/
  data/           Dataset download & preprocessing
  rewards/        Reward functions (outcome, format, reward LM)
  training/       SFT and GRPO training scripts
  eval/           Evaluation and error analysis
  utils/          Shared utilities
scripts/          Shell scripts for setup and running
notebooks/        Analysis notebooks
```

## Key Results

| Model | GeoQA | MathVista |
|---|---|---|
| Qwen2.5-VL-7B (zero-shot) | TBD | TBD |
| + SFT | TBD | TBD |
| + GRPO (outcome) | TBD | TBD |
| + GRPO (outcome + process) | TBD | TBD |

## References

- [EasyR1](https://github.com/hiyouga/EasyR1) — GRPO framework
- [R1-V](https://github.com/StarsfieldAI/R1-V) — VLM reasoning with RL
- [Vision-R1](https://arxiv.org/abs/2503.06749) — GRPO for VLMs
