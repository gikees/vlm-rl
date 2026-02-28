# Decisions Log

## 1. Store raw question text in TRL records

**Context**: The TRL format only stored the prompt as a JSON-serialized message list. Eval scripts needed access to the raw question text but couldn't find it.

**Decision**: Add a `question` field to TRL records containing the original question (before `<image>` tag insertion). Eval scripts already had `row.get("problem", row.get("question", ""))` fallbacks, so adding the field was sufficient.

**Alternative considered**: Could have deserialized the prompt JSON in eval scripts and extracted the text. But storing the raw question is simpler and avoids coupling eval scripts to the prompt format.

## 2. Extract numbers from verbose answers

**Context**: Base model wraps answers in natural language (e.g., "The length of CE is 28 inches" instead of "28"). Direct comparison fails even when the numeric answer is correct.

**Decision**: Added `_extract_number()` fallback in `answers_match` that finds the last number in both predicted and ground truth strings. Uses last number because that's typically the final answer value in a sentence.

**Risk**: Could produce false positives if the predicted answer contains a different number than intended. Acceptable trade-off for base model evaluation - SFT/GRPO will train the model to output clean answers.

## 3. Eval on 500-sample subset instead of full 8K

**Context**: Full GeoQA eval (8,031 samples) takes ~22 hours at ~10s/sample with greedy decoding on 2x RTX 4090.

**Decision**: Use 500 random samples (shuffled, seed=42) for baseline comparisons. ~80 min per eval run. 500 samples gives reasonable statistical significance for comparing methods.

**Revisit**: Run full eval only for final paper numbers (Phase 7).
