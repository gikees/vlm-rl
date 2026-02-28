"""Supervised fine-tuning (SFT) on visual reasoning datasets.

Warm-up training to teach the model the <think>/<answer> format
before GRPO reinforcement learning.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments


class MultimodalSFTCollator:
    """Data collator for multimodal SFT.

    Handles JSON prompt deserialization, image injection,
    chat template application, and tokenization per batch.
    Avoids Arrow schema issues from mixed content types.
    """

    def __init__(self, processor, max_seq_length=2048):
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __call__(self, examples):
        batch_texts = []
        all_images = []

        for example in examples:
            messages = json.loads(example["prompt"])
            solution = example["solution"]
            image = example.get("image")

            # Inject actual image into message content
            if image is not None:
                for msg in messages:
                    if isinstance(msg["content"], list):
                        for item in msg["content"]:
                            if item.get("type") == "image" and "image" not in item:
                                item["image"] = image

            # Build target response in <think>/<answer> format
            target = (
                f"<think>\nLet me analyze this problem step by step.\n"
                f"Looking at the image, I need to find the answer to this question.\n"
                f"After careful analysis, the answer is {solution}.\n"
                f"</think>\n<answer>{solution}</answer>"
            )

            full_messages = messages + [{"role": "assistant", "content": target}]

            # Apply chat template
            text = self.processor.apply_chat_template(full_messages, tokenize=False)
            batch_texts.append(text)

            # Extract image inputs via qwen_vl_utils
            image_inputs, _ = process_vision_info(full_messages)
            if image_inputs:
                all_images.extend(image_inputs)

        # Tokenize
        inputs = self.processor(
            text=batch_texts,
            images=all_images if all_images else None,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # Labels = input_ids, mask padding tokens
        labels = inputs.input_ids.clone()
        labels[inputs.attention_mask == 0] = -100
        inputs["labels"] = labels

        return inputs


def train_sft(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    data_path: str = "data/processed/trl/geoqa",
    output_dir: str = "checkpoints/sft",
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    lora_r: int = 16,
    max_samples: int | None = None,
    max_seq_length: int = 2048,
):
    """Run SFT training with LoRA."""

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    # LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset (no preprocessing - collator handles formatting)
    ds = load_from_disk(data_path)
    if isinstance(ds, dict):
        ds = ds["train"]
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    print(f"Training on {len(ds)} samples")

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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="wandb",
        run_name="vlm-rl-sft",
    )

    collator = MultimodalSFTCollator(processor, max_seq_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"SFT model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="SFT warm-up training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data-path", type=str, default="data/processed/trl/geoqa")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    train_sft(
        model_name=args.model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
