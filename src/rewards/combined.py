"""Combined reward function: outcome + format + optional process reward.

R = alpha * outcome + beta * format + gamma * process
"""

from src.rewards.format import format_reward
from src.rewards.outcome import outcome_reward


def combined_reward(
    completions: list[str],
    solution: list[str],
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.0,
    process_rewards: list[float] | None = None,
    **kwargs,
) -> list[float]:
    """TRL-compatible combined reward function.

    Args:
        completions: model-generated responses
        solution: ground truth answers
        alpha: weight for outcome reward (correctness)
        beta: weight for format reward (<think>/<answer> tags)
        gamma: weight for process reward (from Reward LM, if provided)
        process_rewards: pre-computed process rewards (avoids API calls during training)
    """
    outcome_scores = outcome_reward(completions, solution, **kwargs)
    format_scores = format_reward(completions, **kwargs)

    if process_rewards is None:
        process_rewards = [0.0] * len(completions)

    rewards = []
    for o, f, p in zip(outcome_scores, format_scores, process_rewards):
        rewards.append(alpha * o + beta * f + gamma * p)
    return rewards


def make_reward_fn(alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.0):
    """Create a parameterized reward function for TRL GRPOTrainer.

    Usage:
        reward_fn = make_reward_fn(alpha=1.0, beta=0.5)
        trainer = GRPOTrainer(..., reward_funcs=reward_fn)
    """
    def reward_fn(completions, solution, **kwargs):
        return combined_reward(
            completions, solution,
            alpha=alpha, beta=beta, gamma=gamma,
            **kwargs,
        )
    return reward_fn
