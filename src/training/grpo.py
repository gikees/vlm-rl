"""GRPO training for visual reasoning using TRL.

Group Relative Policy Optimization on Qwen2.5-VL with LoRA.
This is the TRL-based implementation (fallback/alternative to EasyR1).
"""

import argparse
import json
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

from src.rewards.outcome import outcome_reward
from src.rewards.format import format_reward


def load_grpo_dataset(data_path: str, max_samples: int | None = None):
    """Load dataset for GRPO training.

    GRPO only needs prompts (no target responses) — the model generates
    completions which are scored by reward functions.
    """
    ds = load_from_disk(data_path)
    if isinstance(ds, dict):
        ds = ds["train"]

    # Deserialize prompt and flatten multimodal content to plain text.
    # GRPOTrainer's prepare_multimodal_messages() handles image injection
    # from the dataset's "image" column, so we only need text here.
    # All-string content avoids Arrow mixed-type serialization errors.
    def simplify_prompt(example):
        messages = json.loads(example["prompt"])
        for msg in messages:
            if isinstance(msg["content"], list):
                text_parts = [item["text"] for item in msg["content"] if item.get("type") == "text"]
                msg["content"] = " ".join(text_parts).strip()
        return {"prompt": messages}

    ds = ds.map(simplify_prompt)

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    return ds


def train_grpo(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    data_path: str = "data/processed/trl/geoqa",
    output_dir: str = "checkpoints/grpo",
    sft_checkpoint: str | None = None,
    num_epochs: int = 1,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-6,
    lora_r: int = 16,
    num_generations: int = 8,
    max_completion_length: int = 1024,
    beta: float = 0.0,
    temperature: float = 2.0,
    max_samples: int | None = None,
    max_pixels: int = 401408,
    use_vllm: bool = False,
):
    """Run GRPO training with TRL.

    Args:
        model_name: base model to fine-tune
        data_path: path to preprocessed dataset
        output_dir: where to save checkpoints
        sft_checkpoint: optional SFT LoRA checkpoint to start from
        num_epochs: training epochs
        batch_size: per-device batch size
        gradient_accumulation_steps: grad accumulation
        learning_rate: learning rate
        lora_r: LoRA rank
        num_generations: K completions per prompt for GRPO
        max_completion_length: max tokens in generated completion
        beta: KL divergence coefficient
        temperature: sampling temperature for generation
        max_samples: limit training samples (for debugging)
        max_pixels: max image pixels (reduce if OOM)
        use_vllm: use vLLM for generation (faster but needs more setup)
    """
    # Dataset
    dataset = load_grpo_dataset(data_path, max_samples=max_samples)
    print(f"GRPO training on {len(dataset)} samples, K={num_generations}")

    # LoRA config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # GRPO config
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={"device_map": "auto", "torch_dtype": "bfloat16", "attn_implementation": "sdpa"},
        # GRPO-specific
        num_generations=num_generations,
        generation_batch_size=num_generations,
        max_completion_length=max_completion_length,
        beta=beta,
        temperature=temperature,
        loss_type="dapo",
        # Logging & saving
        logging_steps=5,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=5,
        report_to="wandb",
        run_name="vlm-rl-grpo",
        # Generation
        use_vllm=use_vllm,
    )

    # Load model
    model_path = sft_checkpoint if sft_checkpoint else model_name
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if sft_checkpoint:
        # Load base model + merge SFT LoRA, then apply fresh LoRA for GRPO
        # Use device_map="auto" to distribute across GPUs (avoids DataParallel)
        from peft import PeftModel
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, sft_checkpoint)
        model = model.merge_and_unload()
        print(f"Loaded SFT checkpoint from {sft_checkpoint}")
    else:
        model = model_name

    # Reward functions — use both outcome and format
    reward_fns = [outcome_reward, format_reward]

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fns,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"GRPO model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="GRPO training for visual reasoning")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data-path", type=str, default="data/processed/trl/geoqa")
    parser.add_argument("--output-dir", type=str, default="checkpoints/grpo")
    parser.add_argument("--sft-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--use-vllm", action="store_true")
    args = parser.parse_args()

    train_grpo(
        model_name=args.model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        sft_checkpoint=args.sft_checkpoint,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=args.temperature,
        max_samples=args.max_samples,
        max_pixels=args.max_pixels,
        use_vllm=args.use_vllm,
    )


if __name__ == "__main__":
    main()
