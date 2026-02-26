# Problems Log

## 1. Ground truth contains `<answer>` tags

**Found during**: Phase 2 baseline eval sanity check

GeoQA raw `solution` field contains `<answer> 145° </answer>` instead of just `145°`. This gets stored as-is in the TRL `solution` field. The eval script extracts the model's answer from `<answer>` tags but compares it against the raw solution string which still has tags, so `answers_match("145°", "<answer> 145° </answer>")` always fails.

**Fix**: Added `clean_solution()` in `prepare.py` that strips `<answer>` tags using `extract_answer()`. Applied to all datasets (geoqa, clevr, multimodal-r1-8k, geometry3k) in both TRL and verl formats.

## 2. Model answers in natural language

**Found during**: Phase 2 baseline eval sanity check

The model wraps answers in natural language (e.g., "The length of CE is 28 inches") inside `<answer>` tags instead of just the value. `extract_answer` gets the full sentence, which fails numeric comparison against ground truth.

**Status**: Not a bug - expected behavior from a base model. SFT/GRPO should improve this. Could also improve `answers_match` to be more flexible, but not critical now.
