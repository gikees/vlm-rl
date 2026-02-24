"""Outcome reward: binary correctness of final answer.

Compatible with both EasyR1 (verl) and TRL reward function signatures.
"""

from src.data.formatting import answers_match, extract_answer


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """EasyR1/verl reward function signature.

    Returns 1.0 if the extracted answer matches ground truth, 0.0 otherwise.
    """
    predicted = extract_answer(solution_str)
    if predicted is None:
        return 0.0
    return 1.0 if answers_match(predicted, ground_truth) else 0.0


def outcome_reward(completions: list[str], solution: list[str], **kwargs) -> list[float]:
    """TRL GRPOTrainer reward function signature.

    Args:
        completions: model-generated responses
        solution: ground truth answers

    Returns:
        list of rewards (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []
    for completion, gt in zip(completions, solution):
        # Handle dict completions from TRL (content field)
        if isinstance(completion, dict):
            completion = completion.get("content", "")
        elif isinstance(completion, list):
            completion = completion[-1].get("content", "") if completion else ""

        predicted = extract_answer(completion)
        if predicted is None:
            rewards.append(0.0)
        elif answers_match(predicted, gt):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards
