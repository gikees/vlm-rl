# Experiments Log

## Experiment 1: Baseline evaluation on GeoQA

**Date**: 2026-02-27

**Goal**: Establish base model performance on GeoQA before any fine-tuning (SFT or GRPO). These numbers are the "before" in our before/after comparison.

**Setup**:
- Model: Qwen2.5-VL-7B-Instruct (no fine-tuning)
- Dataset: GeoQA (500 random samples, seed=42, from 8,031 total)
- Hardware: 2x RTX 4090, ~10s/sample with greedy decoding
- Two conditions:
  - **CoT**: model prompted to use `<think>/<answer>` format
  - **Zero-shot**: direct answer, no chain-of-thought

**Results**:

| Method | Accuracy | Format Compliance | Time |
|---|---|---|---|
| CoT | 42.40% (212/500) | 96.60% | ~80 min |
| Zero-shot | 41.40% (207/500) | 100.00% | ~80 min |

**Observations**:
- CoT and zero-shot are nearly identical (42.4% vs 41.4%) - CoT prompting doesn't help much on the base model for these geometry problems
- Format compliance is high for both: 96.6% CoT (model follows `<think>/<answer>` format), 100% zero-shot (no format required)
- The model tends to give verbose answers inside `<answer>` tags (e.g., "The length of CE is 28 inches" instead of "28"), which required improving `answers_match` to extract numbers
- ~42% is a reasonable geometry baseline - these are non-trivial problems requiring diagram understanding

**Next steps**:
- Start SFT training on GeoQA-8K (Phase 3)
- Re-evaluate with SFT model to measure improvement

## Experiment 2: SFT warm-up training

**Date**: 2026-02-27

**Goal**: Teach the model the `<think>/<answer>` format via supervised fine-tuning before GRPO.

**Setup**:
- Base model: Qwen2.5-VL-7B-Instruct
- Dataset: GeoQA-8K (8,031 samples, full training set)
- LoRA rank 16, alpha 32, dropout 0.05
- Targets: q/k/v/o/gate/up/down projections (47.6M trainable params, 0.57%)
- lr 2e-5, cosine schedule, warmup 10%, 3 epochs
- Batch size 1, grad accum 8 (effective batch 8)
- Max sequence length 2048, bf16, gradient checkpointing
- Hardware: 2x RTX 4090

**Training stats**:
- Runtime: ~3h11m (~3.6s/step, 3,012 steps)
- Train loss: 1.89 avg, converging to ~1.6 by epoch 3
- Checkpoint: `checkpoints/sft`

**SFT Eval Results**:

| Method | Accuracy | Format Compliance |
|---|---|---|
| Base (zero-shot) | 41.40% | 100.00% |
| Base (CoT) | 42.40% | 96.60% |
| **SFT (CoT)** | **55.40%** | **100.00%** |

**Observations**:
- +13pp accuracy improvement over base CoT (42.4% -> 55.4%)
- Format compliance improved to 100% (from 96.6%) - SFT successfully taught the format
- Inference is much faster (~1.3s/it vs ~10s/it) since the model generates shorter, more focused responses
- Loss converged well (1.89 avg -> ~1.6 by epoch 3), no signs of overfitting

**Next steps**:
- Proceed to GRPO training (Phase 4) using the SFT checkpoint as starting point
