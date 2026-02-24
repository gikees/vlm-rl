"""Model loading utilities for Qwen2.5-VL."""

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM",
}


def load_model_and_processor(
    model_name: str = DEFAULT_MODEL,
    use_lora: bool = True,
    lora_config: dict | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "flash_attention_2",
):
    """Load Qwen2.5-VL model with optional LoRA.

    Args:
        model_name: HuggingFace model ID
        use_lora: whether to apply LoRA adapters
        lora_config: LoRA configuration dict (uses defaults if None)
        torch_dtype: model precision
        device_map: device placement strategy
        attn_implementation: attention implementation (flash_attention_2 recommended)

    Returns:
        (model, processor) tuple
    """
    print(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )

    if use_lora:
        config = lora_config or DEFAULT_LORA_CONFIG
        peft_config = LoraConfig(**config)
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, processor


def load_model_for_inference(
    model_name: str = DEFAULT_MODEL,
    lora_path: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    """Load model for inference, optionally merging LoRA weights.

    Args:
        model_name: base model ID
        lora_path: path to LoRA adapter weights (if any)
        torch_dtype: model precision
        device_map: device placement
    """
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if lora_path:
        from peft import PeftModel
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.merge_and_unload()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    model.eval()
    return model, processor
