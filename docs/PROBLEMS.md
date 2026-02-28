# Problems Log

## 1. Ground truth contains `<answer>` tags

**Found during**: Phase 2 baseline eval sanity check

GeoQA raw `solution` field contains `<answer> 145° </answer>` instead of just `145°`. This gets stored as-is in the TRL `solution` field. The eval script extracts the model's answer from `<answer>` tags but compares it against the raw solution string which still has tags, so `answers_match("145°", "<answer> 145° </answer>")` always fails.

**Fix**: Added `clean_solution()` in `prepare.py` that strips `<answer>` tags using `extract_answer()`. Applied to all datasets (geoqa, clevr, multimodal-r1-8k, geometry3k) in both TRL and verl formats.

## 2. Model answers in natural language

**Found during**: Phase 2 baseline eval sanity check

The model wraps answers in natural language (e.g., "The length of CE is 28 inches") inside `<answer>` tags instead of just the value. `extract_answer` gets the full sentence, which fails numeric comparison against ground truth.

**Fix**: Improved `answers_match` in `formatting.py` to extract numbers from verbose answers as a fallback (e.g., "The measure is 70.0 degrees" matches ground truth "70"). SFT/GRPO should also reduce this behavior over time.

## 3. GRPO loss is zero - no learning signal

**Found during**: Phase 4 GRPO training

GRPO loss stayed at ~0.0001 for the entire run. `frac_reward_zero_std` was 0.6-1.0 throughout, meaning 60-100% of prompts had all K=4 completions receiving the same reward. When all completions get the same reward, the group-relative advantage is 0 and there's nothing to optimize.

Root cause: the SFT model is too deterministic. Entropy dropped to 0.0003-0.05 (near zero). Even with temperature=1.0, the model generates nearly identical completions for each prompt. The binary outcome reward (0 or 1) plus fixed format reward (0.5) is too coarse to differentiate them.

**Fix applied** (three-pronged):
1. **DAPO loss + diversity**: switched `loss_type` from `"grpo"` to `"dapo"` (filters zero-variance groups, no length bias), dropped KL penalty (`beta=0.0`), raised `temperature` to 1.3, doubled `num_generations` to 8
2. **Partial credit outcome reward**: replaced binary 0/1 with graduated scoring based on numerical proximity (1.0 exact, 0.8 within 5%, 0.4 within 15%, 0.2 within 30%, 0.0 otherwise)
3. **Reasoning depth format reward**: added word count dimension to thinking block — 5 reward levels (0.0/0.1/0.3/0.4/0.5) instead of 3, creating variance among well-formatted outputs