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

Temperature tuning: 1.3 collapsed back to deterministic after ~3K steps, 2.0 destroyed coherence (entropy 10+, all gibberish). 1.5 is the sweet spot.

## 4. Qwen2.5-VL image token mismatch crash

**Found during**: Phase 4 GRPO full training run

Certain images cause `ValueError: Image features and image tokens do not match` in Qwen2.5-VL's forward pass. This crashes the entire training run.

**Fix**: Added `RobustGRPOTrainer` subclass that catches the ValueError in `training_step` and returns zero loss for the affected batch, allowing training to continue.

## 5. GRPO eval accuracy gap — compounding diagnostic failures

**Found during**: Phase 4 GRPO eval comparison

GRPO eval shows 39.6% accuracy (198/500) vs 55.4% for SFT. We initially attributed this to LaTeX parsing failures (250/293 "wrong" answers), but that analysis was flawed — the SSH analysis script used the wrong field name (`predicted_answer` instead of `predicted`), causing ALL non-correct samples to be counted as "no answer extracted" regardless of actual failure mode.

Multiple compounding problems:
1. **Wrong field name in analysis** — made it impossible to diagnose real failure modes
2. **Response truncation** — eval stored only first 500 chars of response (`response[:500]`), hiding what the model actually generated
3. **No truncation detection** — 15/500 samples had `format_ok=false` (no complete `<answer>` tags), suggesting `max_new_tokens=1024` was too low, but no way to confirm
4. **max_new_tokens mismatch** — GRPO training uses 1024, but format reward incentivizes 50+ word thinking blocks, so model learned to be verbose; SFT trained on 2048-token sequences
5. **Unit stripping regression** — the `m` unit match in `normalize_answer` could strip valid trailing characters (accuracy dropped 40.6% → 39.6% after LaTeX fix)

**Fix (LaTeX parsing, applied earlier)**: Enhanced answer matching in `formatting.py`:
1. `normalize_answer` now strips LaTeX delimiters (`\(`, `\)`, `$`), unwraps `\text{}`/`\mathrm{}`, removes `^\circ` notation, and strips trailing units
2. Added `_eval_simple_latex()` — regex-based evaluator for `\frac{a}{b}`, `\sqrt{a}`, `a\sqrt{b}`, `\pi`, and plain fractions (no sympy dependency)
3. Added `_extract_from_equation()` — extracts RHS from equations like `x = 48` → `48`
4. Enhanced `_extract_number` extraction order: direct float → LaTeX eval → equation RHS → embedded LaTeX scan → bare number fallback

**Fix (eval diagnostics)**: Overhauled `baseline.py`:
1. Store full response (removed `[:500]` truncation) + `response_length` field
2. Added `was_truncated` detection (checks if `generated_ids` length hit `max_new_tokens`)
3. Added `predicted_raw` field for debugging answer extraction
4. Increased default `max_new_tokens` from 1024 to 2048
5. Added diagnostic summary: correct/wrong/no-answer/truncated counts + avg response lengths
6. Fixed unit regex: short ambiguous units (`m`, `ft`, `in`) now require whitespace separator to avoid false matches