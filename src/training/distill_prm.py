"""Stage B: Distill a local Process Reward Model from API-labeled data.

Pipeline:
1. Load reasoning traces scored by the Reward LM API (from generate_trajectories.py)
2. Train a small model (e.g. Qwen2.5-VL-3B) to predict the structured scores
3. Use this local PRM during GRPO instead of expensive API calls

The PRM is trained as a regression task: given (question, response) -> scores.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)


class PRMHead(nn.Module):
    """Lightweight head that maps LLM hidden states to reward scores."""

    def __init__(self, hidden_size: int, n_scores: int = 3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_scores),
            nn.Sigmoid(),  # Output in [0, 1], multiply by 5 to get [0, 5]
        )

    def forward(self, hidden_states):
        # Use last token's hidden state
        return self.head(hidden_states[:, -1, :]) * 5.0


def load_scored_data(trajectories_path: str) -> list[dict]:
    """Load API-scored trajectories and extract training data for the PRM.

    Expected input: output of generate_trajectories.py with rlm_scores.
    """
    with open(trajectories_path) as f:
        data = json.load(f)

    # Handle both raw trajectories and DPO pairs format
    samples = []

    if isinstance(data, list) and data and "responses" in data[0]:
        # Raw trajectories format (from generate_trajectories)
        # Need to also load the scored pairs for rlm_scores
        print("Note: raw trajectories format. Use DPO pairs file for scored data.")
        return samples

    if isinstance(data, list) and data and "chosen" in data[0]:
        # DPO pairs format - has rlm_scores via chosen/rejected
        for pair in data:
            # We'd need the full scored data. For now, this is a placeholder.
            # In practice, save the full scored trajectories from generate_trajectories.
            pass

    return samples


def load_prm_training_data(scored_traces_path: str) -> Dataset:
    """Load pre-scored traces into a training dataset.

    Expected JSON format (one per line or as list):
    {
        "question": "...",
        "response": "...",
        "perception_score": 0-5,
        "reasoning_score": 0-5,
        "groundedness_score": 0-5
    }
    """
    with open(scored_traces_path) as f:
        data = json.load(f)

    # Flatten if nested
    records = []
    if isinstance(data, dict) and "error_details" in data:
        # Output from analysis.py
        for item in data["error_details"]:
            records.append({
                "text": f"Question: {item.get('question', '')}\n\nResponse: {item.get('response', '')}",
                "perception_score": item.get("perception_score", 0) / 5.0,
                "reasoning_score": item.get("reasoning_score", 0) / 5.0,
                "groundedness_score": item.get("groundedness_score", 0) / 5.0,
            })
    elif isinstance(data, list):
        for item in data:
            records.append({
                "text": f"Question: {item.get('question', '')}\n\nResponse: {item.get('response', '')}",
                "perception_score": item.get("perception_score", 0) / 5.0,
                "reasoning_score": item.get("reasoning_score", 0) / 5.0,
                "groundedness_score": item.get("groundedness_score", 0) / 5.0,
            })

    return Dataset.from_list(records)


def score_traces_with_api(
    traces_path: str,
    output_path: str,
    provider: str = "anthropic",
    model: str | None = None,
    max_samples: int | None = None,
):
    """Score a batch of reasoning traces with the API for PRM training data.

    Input: JSON with list of {question, response, ground_truth}
    Output: Same list augmented with perception/reasoning/groundedness scores
    """
    from src.rewards.reward_lm import RewardLMScorer

    with open(traces_path) as f:
        traces = json.load(f)

    if max_samples:
        traces = traces[:max_samples]

    scorer = RewardLMScorer(provider=provider, model=model)
    scored = []

    for trace in tqdm(traces, desc="Scoring with API"):
        scores = scorer.score(
            question=trace.get("question", ""),
            ground_truth=trace.get("ground_truth", ""),
            response=trace.get("response", ""),
        )
        scored.append({**trace, **scores})

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(scored, f, indent=2)

    print(f"Scored {len(scored)} traces, saved to {output}")
    return scored


def train_prm(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    scored_data_path: str = "data/processed/scored_traces.json",
    output_dir: str = "checkpoints/prm",
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-5,
    lora_r: int = 8,
):
    """Train local PRM on API-scored data.

    Uses the base VLM with a regression head to predict structured scores.
    """
    dataset = load_prm_training_data(scored_data_path)
    print(f"PRM training on {len(dataset)} scored traces")

    if len(dataset) == 0:
        print("No training data found. Run score_traces_with_api first.")
        return

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb",
        run_name="vlm-rl-prm",
        remove_unused_columns=False,
    )

    # Custom trainer with MSE loss on scores
    class PRMTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            text_inputs = processor(
                text=inputs["text"],
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(model.device)

            outputs = model(**text_inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]

            # Use mean pooling of last layer
            scores_pred = self._prm_head(hidden)

            targets = torch.stack([
                inputs["perception_score"],
                inputs["reasoning_score"],
                inputs["groundedness_score"],
            ], dim=1).to(model.device).float()

            loss = nn.functional.mse_loss(scores_pred, targets)

            if return_outputs:
                return loss, {"predictions": scores_pred}
            return loss

    # Initialize PRM head
    hidden_size = model.config.hidden_size
    prm_head = PRMHead(hidden_size).to(model.device)

    trainer = PRMTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer._prm_head = prm_head

    trainer.train()

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    torch.save(prm_head.state_dict(), Path(output_dir) / "prm_head.pt")
    processor.save_pretrained(output_dir)
    print(f"PRM saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Distill local PRM from API scores")
    subparsers = parser.add_subparsers(dest="command")

    # Score traces command
    score_parser = subparsers.add_parser("score", help="Score traces with API")
    score_parser.add_argument("--input", type=str, required=True)
    score_parser.add_argument("--output", type=str, required=True)
    score_parser.add_argument("--provider", type=str, default="anthropic")
    score_parser.add_argument("--model", type=str, default=None)
    score_parser.add_argument("--max-samples", type=int, default=None)

    # Train PRM command
    train_parser = subparsers.add_parser("train", help="Train local PRM")
    train_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    train_parser.add_argument("--scored-data", type=str, default="data/processed/scored_traces.json")
    train_parser.add_argument("--output-dir", type=str, default="checkpoints/prm")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--lr", type=float, default=1e-5)
    train_parser.add_argument("--lora-r", type=int, default=8)

    args = parser.parse_args()

    if args.command == "score":
        score_traces_with_api(
            args.input, args.output,
            provider=args.provider, model=args.model,
            max_samples=args.max_samples,
        )
    elif args.command == "train":
        train_prm(
            model_name=args.model,
            scored_data_path=args.scored_data,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_r=args.lora_r,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
