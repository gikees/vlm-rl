"""Best-of-N evaluation: generate K responses and pick the best.

Critical non-RL baseline — reviewers will ask for this comparison.
Reuses the same generation infrastructure as GRPO but just picks the
highest-reward completion rather than optimizing the policy.
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


def generate_n_responses(
    model,
    processor,
    messages,
    image,
    n: int = 8,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> list[str]:
    """Generate N diverse responses for a single prompt."""
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

    responses = []
    for _ in range(n):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
            )
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        responses.append(response)

    return responses


def score_response(response: str, ground_truth: str) -> float:
    """Score a response: outcome + format reward."""
    score = 0.0
    predicted = extract_answer(response)
    if predicted and answers_match(predicted, ground_truth):
        score += 1.0
    if has_valid_format(response):
        score += 0.5
    return score


def evaluate_best_of_n(
    model,
    processor,
    dataset_path: str,
    n: int = 8,
    max_samples: int | None = None,
    temperature: float = 0.7,
) -> dict:
    """Run Best-of-N evaluation.

    For each prompt, generate N responses and pick the highest-scoring one.
    Reports both best-of-N accuracy and majority-vote accuracy.
    """
    ds = load_from_disk(dataset_path)
    if isinstance(ds, dict):
        for split in ["test", "validation", "train"]:
            if split in ds:
                ds = ds[split]
                break

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    results = []
    bon_correct = 0
    majority_correct = 0

    for i, row in enumerate(tqdm(ds, desc=f"Best-of-{n}")):
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
                {"type": "text", "text": question + COT_PROMPT_SUFFIX},
            ]
        else:
            content = question + COT_PROMPT_SUFFIX
        messages.append({"role": "user", "content": content})

        # Generate N responses
        try:
            responses = generate_n_responses(model, processor, messages, image, n, temperature=temperature)
        except Exception as e:
            responses = [f"ERROR: {e}"]

        # Score each response
        scores = [score_response(r, gt_answer) for r in responses]
        best_idx = scores.index(max(scores))
        best_response = responses[best_idx]
        best_predicted = extract_answer(best_response)
        bon_is_correct = best_predicted and answers_match(best_predicted, gt_answer)

        # Majority vote
        predictions = [extract_answer(r) for r in responses]
        valid_preds = [p for p in predictions if p is not None]
        if valid_preds:
            from collections import Counter
            majority_pred = Counter(valid_preds).most_common(1)[0][0]
            majority_is_correct = answers_match(majority_pred, gt_answer)
        else:
            majority_is_correct = False

        if bon_is_correct:
            bon_correct += 1
        if majority_is_correct:
            majority_correct += 1

        results.append({
            "idx": i,
            "question": question[:200],
            "ground_truth": gt_answer,
            "n_responses": len(responses),
            "best_predicted": best_predicted,
            "best_score": max(scores),
            "bon_correct": bon_is_correct,
            "majority_correct": majority_is_correct,
            "any_correct": any(s >= 1.0 for s in scores),
            "scores": scores,
        })

    total = len(results)
    any_correct = sum(1 for r in results if r["any_correct"])

    return {
        "n": n,
        "total": total,
        "best_of_n_accuracy": bon_correct / total if total else 0,
        "majority_vote_accuracy": majority_correct / total if total else 0,
        "any_correct_rate": any_correct / total if total else 0,
        "bon_correct": bon_correct,
        "majority_correct": majority_correct,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Best-of-N evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="data/processed/trl/geoqa")
    parser.add_argument("--output", type=str, default="results/best_of_n.json")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if args.lora_path:
        from src.utils.model import load_model_for_inference
        model, processor = load_model_for_inference(args.model, lora_path=args.lora_path)
    else:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

    metrics = evaluate_best_of_n(
        model, processor, args.dataset,
        n=args.n,
        max_samples=args.max_samples,
        temperature=args.temperature,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nBest-of-{args.n} Results:")
    print(f"  Best-of-N accuracy:    {metrics['best_of_n_accuracy']:.2%}")
    print(f"  Majority vote accuracy: {metrics['majority_vote_accuracy']:.2%}")
    print(f"  Any correct rate:      {metrics['any_correct_rate']:.2%}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
