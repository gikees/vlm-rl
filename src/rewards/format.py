"""Format reward: checks if response follows <think>/<answer> structure.

Encourages the model to produce structured chain-of-thought reasoning.
"""

import re

from src.data.formatting import has_valid_format


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """EasyR1/verl reward function signature.

    Returns 0.5 if format is correct, with partial credit.
    """
    return _score_format(solution_str)


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """TRL GRPOTrainer reward function signature."""
    rewards = []
    for completion in completions:
        if isinstance(completion, dict):
            completion = completion.get("content", "")
        elif isinstance(completion, list):
            completion = completion[-1].get("content", "") if completion else ""
        rewards.append(_score_format(completion))
    return rewards


def _score_format(response: str) -> float:
    """Score the format quality of a response.

    Returns:
        0.0: no structure at all
        0.1: has one of <think> or <answer> but not both
        0.3: both tags, <20 words of thinking
        0.4: both tags, 20-49 words of thinking
        0.5: both tags, 50+ words of thinking
    """
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    has_answer = bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL))

    if think_match and has_answer:
        word_count = len(think_match.group(1).split())
        if word_count >= 50:
            return 0.5
        if word_count >= 20:
            return 0.4
        return 0.3
    elif think_match or has_answer:
        return 0.1
    return 0.0
