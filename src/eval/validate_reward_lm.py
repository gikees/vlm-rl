"""Validate the Reward LM scoring prompt for consistency.

Critical early task: score ~50-100 reasoning traces and check if the
structured scores are consistent and can reliably distinguish
perception vs reasoning errors.

Run this BEFORE using the Reward LM in training.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.rewards.reward_lm import RewardLMScorer, scores_to_reward


# Hand-crafted test cases with known error types
VALIDATION_CASES = [
    {
        "question": "How many triangles are in the figure?",
        "ground_truth": "3",
        "response": "<think>I can see the geometric figure. Let me count the triangles. I see 2 large triangles and 1 small triangle formed by the intersection. That gives us 3 triangles total.</think>\n<answer>3</answer>",
        "expected_diagnosis": "correct",
        "description": "correct answer with good reasoning",
    },
    {
        "question": "How many triangles are in the figure?",
        "ground_truth": "3",
        "response": "<think>Looking at the figure, I can see there are 5 circles and 2 squares. The circles are arranged in a pentagon shape. So there must be 5 triangles.</think>\n<answer>5</answer>",
        "expected_diagnosis": "perception_error",
        "description": "wrong perception (sees circles instead of triangles)",
    },
    {
        "question": "Find angle x if the two lines are parallel and the transversal creates a 60 degree angle.",
        "ground_truth": "120",
        "response": "<think>I can see two parallel lines cut by a transversal. The transversal makes a 60 degree angle with one line. Since the lines are parallel, the corresponding angle is also 60 degrees. So x = 60.</think>\n<answer>60</answer>",
        "expected_diagnosis": "reasoning_error",
        "description": "correct perception but wrong reasoning (should use supplementary angles)",
    },
    {
        "question": "What is the area of the shaded region?",
        "ground_truth": "25",
        "response": "<think>The shaded region is a hexagon with side length 4. I can also see a small dragon drawn in the corner. Using the formula for a regular hexagon, the area is pi * 4^2 = 50.24.</think>\n<answer>50</answer>",
        "expected_diagnosis": "both",
        "description": "hallucinated element and wrong formula",
    },
    {
        "question": "What is the radius of the circle?",
        "ground_truth": "5",
        "response": "The radius is 5.",
        "expected_diagnosis": "format_error",
        "description": "correct answer but no think/answer tags",
    },
]


def run_validation(
    scorer: RewardLMScorer,
    cases: list[dict] | None = None,
    n_repeats: int = 3,
) -> dict:
    """Run validation: score each case multiple times to check consistency.

    Args:
        scorer: RewardLMScorer instance
        cases: test cases (uses built-in VALIDATION_CASES if None)
        n_repeats: how many times to score each case (for consistency check)
    """
    if cases is None:
        cases = VALIDATION_CASES

    results = []

    for case in tqdm(cases, desc="Validating"):
        repeat_scores = []
        repeat_diagnoses = []

        for _ in range(n_repeats):
            scores = scorer.score(
                question=case["question"],
                ground_truth=case["ground_truth"],
                response=case["response"],
            )
            repeat_scores.append(scores)
            repeat_diagnoses.append(scores.get("diagnosis", "unknown"))

        # Check consistency
        diagnoses_match = len(set(repeat_diagnoses)) == 1
        perception_scores = [s["perception_score"] for s in repeat_scores]
        reasoning_scores = [s["reasoning_score"] for s in repeat_scores]
        groundedness_scores = [s["groundedness_score"] for s in repeat_scores]

        result = {
            "description": case["description"],
            "expected_diagnosis": case["expected_diagnosis"],
            "predicted_diagnoses": repeat_diagnoses,
            "diagnosis_consistent": diagnoses_match,
            "diagnosis_correct": repeat_diagnoses[0] == case["expected_diagnosis"],
            "perception_scores": perception_scores,
            "perception_std": float(np.std(perception_scores)),
            "reasoning_scores": reasoning_scores,
            "reasoning_std": float(np.std(reasoning_scores)),
            "groundedness_scores": groundedness_scores,
            "groundedness_std": float(np.std(groundedness_scores)),
        }
        results.append(result)

    # Summary
    n = len(results)
    consistency_rate = sum(1 for r in results if r["diagnosis_consistent"]) / n
    accuracy_rate = sum(1 for r in results if r["diagnosis_correct"]) / n
    avg_perception_std = np.mean([r["perception_std"] for r in results])
    avg_reasoning_std = np.mean([r["reasoning_std"] for r in results])
    avg_groundedness_std = np.mean([r["groundedness_std"] for r in results])

    summary = {
        "n_cases": n,
        "n_repeats": n_repeats,
        "diagnosis_consistency": consistency_rate,
        "diagnosis_accuracy": accuracy_rate,
        "avg_perception_std": float(avg_perception_std),
        "avg_reasoning_std": float(avg_reasoning_std),
        "avg_groundedness_std": float(avg_groundedness_std),
        "results": results,
    }

    return summary


def print_validation_report(summary: dict):
    """Print a readable validation report."""
    print(f"\n{'='*60}")
    print("REWARD LM VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Cases: {summary['n_cases']}, Repeats per case: {summary['n_repeats']}")
    print(f"Diagnosis consistency: {summary['diagnosis_consistency']:.0%}")
    print(f"Diagnosis accuracy:    {summary['diagnosis_accuracy']:.0%}")
    print(f"\nScore stability (lower std = more consistent):")
    print(f"  Perception std:    {summary['avg_perception_std']:.2f}")
    print(f"  Reasoning std:     {summary['avg_reasoning_std']:.2f}")
    print(f"  Groundedness std:  {summary['avg_groundedness_std']:.2f}")

    print(f"\n{'='*60}")
    print("PER-CASE RESULTS")
    print(f"{'='*60}")
    for r in summary["results"]:
        status = "PASS" if r["diagnosis_correct"] and r["diagnosis_consistent"] else "FAIL"
        print(f"\n[{status}] {r['description']}")
        print(f"  Expected: {r['expected_diagnosis']}")
        print(f"  Got:      {r['predicted_diagnoses']}")
        print(f"  Scores:   P={r['perception_scores']} R={r['reasoning_scores']} G={r['groundedness_scores']}")


def main():
    parser = argparse.ArgumentParser(description="Validate Reward LM scoring prompt")
    parser.add_argument("--provider", type=str, default="anthropic")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--output", type=str, default="results/reward_lm_validation.json")
    args = parser.parse_args()

    scorer = RewardLMScorer(provider=args.provider, model=args.model)
    summary = run_validation(scorer, n_repeats=args.n_repeats)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_validation_report(summary)
    print(f"\nFull results saved to {output_path}")

    # Guidance
    if summary["diagnosis_consistency"] < 0.8:
        print("\nWARNING: Low consistency. Consider simplifying the scoring rubric.")
    if summary["diagnosis_accuracy"] < 0.6:
        print("\nWARNING: Low accuracy. Iterate on the scoring prompt before using in training.")


if __name__ == "__main__":
    main()
