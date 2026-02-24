"""Prompt and response formatting for visual reasoning tasks.

Handles the <think>...</think><answer>...</answer> format used throughout training.
"""

import re

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves visual reasoning problems step by step. "
    "First, carefully observe the image and describe what you see. "
    "Then, reason through the problem logically. "
    "Output your thinking process inside <think>...</think> tags, "
    "then provide your final answer inside <answer>...</answer> tags."
)

COT_PROMPT_SUFFIX = (
    "\n\nPlease reason step by step. Put your reasoning in <think>...</think> tags "
    "and your final answer in <answer>...</answer> tags."
)


def format_prompt_for_chat(question: str, system: str = SYSTEM_PROMPT) -> list[dict]:
    """Format a question into chat message format for the model.

    Returns messages list compatible with HuggingFace chat_template.
    The <image> tag in the question signals where the image goes.
    """
    messages = [{"role": "system", "content": system}]

    # Build user content with image placeholder if present
    if "<image>" in question:
        content = []
        parts = question.split("<image>")
        if parts[0].strip():
            content.append({"type": "text", "text": parts[0].strip()})
        content.append({"type": "image"})
        remainder = "<image>".join(parts[1:]).strip()
        if remainder:
            content.append({"type": "text", "text": remainder + COT_PROMPT_SUFFIX})
        else:
            content.append({"type": "text", "text": COT_PROMPT_SUFFIX.strip()})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({
            "role": "user",
            "content": question + COT_PROMPT_SUFFIX,
        })

    return messages


def extract_answer(response: str) -> str | None:
    """Extract the final answer from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_thinking(response: str) -> str | None:
    """Extract the thinking/reasoning from <think>...</think> tags."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def has_valid_format(response: str) -> bool:
    """Check if response follows <think>...</think><answer>...</answer> format."""
    has_think = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL))
    return has_think and has_answer


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Handles common variations: extra whitespace, degree symbols,
    trailing periods, percentage signs.
    """
    answer = answer.strip().lower()
    # Remove trailing period
    answer = answer.rstrip(".")
    # Remove degree symbol
    answer = answer.replace("°", "")
    # Remove percentage sign
    answer = answer.rstrip("%")
    # Normalize whitespace
    answer = " ".join(answer.split())
    return answer


def answers_match(predicted: str, ground_truth: str, tolerance: float = 1e-4) -> bool:
    """Check if predicted answer matches ground truth.

    Tries numeric comparison first, then falls back to string matching.
    """
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Try numeric comparison
    try:
        pred_val = float(pred_norm)
        gt_val = float(gt_norm)
        return abs(pred_val - gt_val) < tolerance
    except ValueError:
        pass

    # String comparison
    return pred_norm == gt_norm
