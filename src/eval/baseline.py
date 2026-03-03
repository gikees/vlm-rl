"""Baseline evaluation of Qwen2.5-VL on visual reasoning benchmarks.

Runs zero-shot and CoT-prompted evaluation, recording accuracy per task type.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.data.formatting import (
    COT_PROMPT_SUFFIX,
    SYSTEM_PROMPT,
    answers_match,
    extract_answer,
    has_valid_format,
)


def run_inference(model, processor, messages, image=None, max_new_tokens=2048):
    """Run single inference with the model.

    Returns (response_text, was_truncated) tuple.
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if image is not None:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for eval
        )

    # Decode only the generated tokens
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    was_truncated = generated_ids.shape[1] >= max_new_tokens
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response, was_truncated


def evaluate_dataset(
    model,
    processor,
    dataset_path: str,
    use_cot: bool = True,
    max_samples: int | None = None,
    max_new_tokens: int = 2048,
) -> dict:
    """Evaluate model on a dataset.

    Returns dict with accuracy, per-sample results, and format compliance stats.
    """
    ds = load_from_disk(dataset_path)
    if isinstance(ds, dict):
        # Try test split first, then validation, then train
        for split in ["test", "validation", "train"]:
            if split in ds:
                ds = ds[split]
                break

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    results = []
    correct = 0
    format_ok = 0

    for i, row in enumerate(tqdm(ds, desc="Evaluating")):
        question = row.get("problem", row.get("question", ""))
        gt_answer = str(row.get("solution", row.get("answer", "")))
        image = row.get("image", row.get("images", None))
        if isinstance(image, list) and len(image) > 0:
            image = image[0]

        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if image is not None:
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": question + (COT_PROMPT_SUFFIX if use_cot else "")},
            ]
        else:
            content = question + (COT_PROMPT_SUFFIX if use_cot else "")
        messages.append({"role": "user", "content": content})

        # Generate
        try:
            response, was_truncated = run_inference(
                model, processor, messages, image, max_new_tokens
            )
        except Exception as e:
            response = f"ERROR: {e}"
            was_truncated = False

        # Score
        predicted = extract_answer(response) if use_cot else response.strip()
        is_correct = answers_match(predicted, gt_answer) if predicted else False
        is_format_ok = has_valid_format(response) if use_cot else True

        if is_correct:
            correct += 1
        if is_format_ok:
            format_ok += 1

        results.append({
            "idx": i,
            "question": question[:200],
            "ground_truth": gt_answer,
            "predicted": predicted,
            "predicted_raw": extract_answer(response) if use_cot else response.strip(),
            "response": response,
            "response_length": len(response),
            "correct": is_correct,
            "format_ok": is_format_ok,
            "was_truncated": was_truncated,
            "ability": row.get("ability", "unknown"),
        })

    total = len(results)
    return {
        "accuracy": correct / total if total > 0 else 0,
        "format_compliance": format_ok / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--lora-path", type=str, default=None, help="LoRA checkpoint to evaluate")
    parser.add_argument("--dataset", type=str, default="data/processed/trl/geoqa")
    parser.add_argument("--output", type=str, default="results/baseline.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-cot", action="store_true", help="Disable chain-of-thought prompting")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    # Load model
    if args.lora_path:
        from src.utils.model import load_model_for_inference
        model, processor = load_model_for_inference(
            args.model, lora_path=args.lora_path
        )
    else:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

    # Evaluate
    metrics = evaluate_dataset(
        model, processor, args.dataset,
        use_cot=not args.no_cot,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print(f"Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"Format compliance: {metrics['format_compliance']:.2%}")

    # Detailed diagnostic summary
    res = metrics["results"]
    n_correct = sum(1 for r in res if r["correct"])
    n_wrong = sum(1 for r in res if not r["correct"] and r["predicted"])
    n_no_answer = sum(1 for r in res if not r["correct"] and not r["predicted"])
    n_truncated = sum(1 for r in res if r.get("was_truncated"))

    correct_lens = [r["response_length"] for r in res if r["correct"]]
    incorrect_lens = [r["response_length"] for r in res if not r["correct"]]

    print(f"\n--- Diagnostic breakdown ---")
    print(f"Correct:    {n_correct}")
    print(f"Wrong:      {n_wrong}")
    print(f"No answer:  {n_no_answer}")
    print(f"Truncated:  {n_truncated}")
    if correct_lens:
        print(f"Avg response length (correct):   {sum(correct_lens)/len(correct_lens):.0f} chars")
    if incorrect_lens:
        print(f"Avg response length (incorrect): {sum(incorrect_lens)/len(incorrect_lens):.0f} chars")


if __name__ == "__main__":
    main()
