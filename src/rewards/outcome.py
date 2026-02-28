"""Outcome reward: binary correctness of final answer.

Compatible with both EasyR1 (verl) and TRL reward function signatures.
"""

from src.data.formatting import _extract_number, answers_match, extract_answer, normalize_answer


def _partial_credit(predicted: str, ground_truth: str) -> float:
    """Score answer with partial credit based on numerical proximity.

    Returns:
        1.0: exact match
        0.8: within 5% relative error
        0.4: within 15%
        0.2: within 30%
        0.0: otherwise or non-numeric
    """
    if answers_match(predicted, ground_truth):
        return 1.0

    pred_num = _extract_number(normalize_answer(predicted))
    gt_num = _extract_number(normalize_answer(ground_truth))

    if pred_num is None or gt_num is None:
        return 0.0

    if gt_num == 0:
        return 0.0

    rel_error = abs(pred_num - gt_num) / abs(gt_num)
    if rel_error <= 0.05:
        return 0.8
    if rel_error <= 0.15:
        return 0.4
    if rel_error <= 0.30:
        return 0.2
    return 0.0


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """EasyR1/verl reward function signature.

    Returns graduated score based on answer proximity.
    """
    predicted = extract_answer(solution_str)
    if predicted is None:
        return 0.0
    return _partial_credit(predicted, ground_truth)


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
        else:
            rewards.append(_partial_credit(predicted, gt))
    return rewards
