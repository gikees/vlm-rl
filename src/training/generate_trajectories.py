"""Generate and score trajectories for offline DPO training.

Stage A of the Reward LM pipeline:
1. Generate K completions per prompt from the GRPO model
2. Score each with the Reward LM API
3. Build preference pairs (chosen/rejected) for DPO/KTO
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.data.formatting import COT_PROMPT_SUFFIX, SYSTEM_PROMPT, extract_answer, answers_match
from src.rewards.reward_lm import RewardLMScorer, scores_to_reward


def generate_trajectories(
    model,
    processor,
    dataset_path: str,
    n: int = 4,
    max_samples: int | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
) -> list[dict]:
    """Generate N trajectories per prompt.

    Returns list of dicts, each with prompt, ground_truth, and N responses.
    """
    ds = load_from_disk(dataset_path)
    if isinstance(ds, dict):
        ds = ds["train"]
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    trajectories = []

    for i, row in enumerate(tqdm(ds, desc=f"Generating {n} trajectories")):
        question = row.get("problem", row.get("question", ""))
        gt = str(row.get("solution", row.get("answer", "")))
        image = row.get("image", row.get("images", None))
        if isinstance(image, list) and len(image) > 0:
            image = image[0]

        if "<image>" not in question:
            question = "<image>\n" + question

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if image is not None:
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": question + COT_PROMPT_SUFFIX},
            ]
        else:
            content = question + COT_PROMPT_SUFFIX
        messages.append({"role": "user", "content": content})

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if image is not None:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
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
            gen_ids = output_ids[:, inputs.input_ids.shape[1]:]
            resp = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            responses.append(resp)

        trajectories.append({
            "idx": i,
            "question": question,
            "ground_truth": gt,
            "responses": responses,
            "prompt": messages,
        })

    return trajectories


def score_and_build_pairs(
    trajectories: list[dict],
    scorer: RewardLMScorer,
    outcome_weight: float = 0.5,
    process_weight: float = 0.5,
) -> list[dict]:
    """Score trajectories with Reward LM and build DPO preference pairs.

    For each prompt, picks the highest-scoring response as 'chosen'
    and the lowest-scoring as 'rejected'.
    """
    pairs = []

    for traj in tqdm(trajectories, desc="Scoring trajectories"):
        question = traj["question"]
        gt = traj["ground_truth"]
        responses = traj["responses"]

        scored = []
        for resp in responses:
            # Outcome score
            predicted = extract_answer(resp)
            outcome = 1.0 if predicted and answers_match(predicted, gt) else 0.0

            # Process score from Reward LM
            rlm_scores = scorer.score(question, gt, resp)
            process = scores_to_reward(rlm_scores)

            total = outcome_weight * outcome + process_weight * process

            scored.append({
                "response": resp,
                "outcome_score": outcome,
                "process_score": process,
                "total_score": total,
                "rlm_scores": rlm_scores,
            })

        # Sort by total score
        scored.sort(key=lambda x: x["total_score"], reverse=True)

        # Build preference pair: best vs worst
        if len(scored) >= 2 and scored[0]["total_score"] > scored[-1]["total_score"]:
            pairs.append({
                "prompt": traj["prompt"],
                "question": question,
                "ground_truth": gt,
                "chosen": scored[0]["response"],
                "rejected": scored[-1]["response"],
                "chosen_score": scored[0]["total_score"],
                "rejected_score": scored[-1]["total_score"],
                "chosen_diagnosis": scored[0]["rlm_scores"].get("diagnosis", ""),
                "rejected_diagnosis": scored[-1]["rlm_scores"].get("diagnosis", ""),
            })

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate and score trajectories for DPO")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="data/processed/trl/geoqa")
    parser.add_argument("--output", type=str, default="data/processed/dpo_pairs.json")
    parser.add_argument("--n", type=int, default=4, help="Trajectories per prompt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--provider", type=str, default="anthropic")
    parser.add_argument("--rlm-model", type=str, default=None)
    args = parser.parse_args()

    # Load model
    if args.lora_path:
        from src.utils.model import load_model_for_inference
        model, processor = load_model_for_inference(args.model, lora_path=args.lora_path)
    else:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        model.eval()

    # Generate
    trajectories = generate_trajectories(
        model, processor, args.dataset,
        n=args.n, max_samples=args.max_samples,
    )

    # Save raw trajectories
    traj_path = Path(args.output).with_suffix(".trajectories.json")
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    with open(traj_path, "w") as f:
        json.dump(trajectories, f, indent=2, default=str)
    print(f"Raw trajectories saved to {traj_path}")

    # Score and build pairs
    scorer = RewardLMScorer(provider=args.provider, model=args.rlm_model)
    pairs = score_and_build_pairs(trajectories, scorer)

    # Save DPO pairs
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2, default=str)
    print(f"DPO pairs saved to {output_path} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
