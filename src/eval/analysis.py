"""Error analysis and perception vs reasoning taxonomy.

Uses the Reward LM's structured scores to categorize failure modes
and quantify what bottlenecks RL improvement.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.rewards.reward_lm import RewardLMScorer, scores_to_reward


def analyze_errors(
    results_path: str,
    scorer: RewardLMScorer | None = None,
    max_samples: int | None = None,
) -> dict:
    """Analyze errors in evaluation results using Reward LM scoring.

    Args:
        results_path: path to detailed evaluation results JSON
        scorer: RewardLMScorer instance (creates one if None)
        max_samples: limit scoring to N incorrect samples

    Returns dict with error taxonomy and statistics.
    """
    with open(results_path) as f:
        eval_results = json.load(f)

    results = eval_results.get("results", eval_results)
    if isinstance(results, dict):
        results = results.get("results", [])

    # Separate correct and incorrect
    correct_samples = [r for r in results if r.get("correct")]
    incorrect_samples = [r for r in results if not r.get("correct")]

    print(f"Total: {len(results)}, Correct: {len(correct_samples)}, Incorrect: {len(incorrect_samples)}")

    if not incorrect_samples:
        return {"error": "No incorrect samples to analyze"}

    if max_samples and len(incorrect_samples) > max_samples:
        incorrect_samples = incorrect_samples[:max_samples]

    if scorer is None:
        scorer = RewardLMScorer()

    # Score incorrect samples
    scored_errors = []
    diagnosis_counts = Counter()
    perception_scores = []
    reasoning_scores = []
    groundedness_scores = []

    for sample in tqdm(incorrect_samples, desc="Scoring errors"):
        scores = scorer.score(
            question=sample.get("question", ""),
            ground_truth=sample.get("ground_truth", ""),
            response=sample.get("response", ""),
        )

        scored_errors.append({**sample, **scores})
        diagnosis_counts[scores.get("diagnosis", "unknown")] += 1
        perception_scores.append(scores.get("perception_score", 0))
        reasoning_scores.append(scores.get("reasoning_score", 0))
        groundedness_scores.append(scores.get("groundedness_score", 0))

    n = len(scored_errors)
    analysis = {
        "total_samples": len(results),
        "correct": len(correct_samples),
        "incorrect": len(incorrect_samples),
        "scored_errors": n,
        "diagnosis_distribution": dict(diagnosis_counts),
        "avg_perception_score": sum(perception_scores) / n if n else 0,
        "avg_reasoning_score": sum(reasoning_scores) / n if n else 0,
        "avg_groundedness_score": sum(groundedness_scores) / n if n else 0,
        "error_details": scored_errors,
    }

    # Compute bottleneck: which dimension is weakest?
    avgs = {
        "perception": analysis["avg_perception_score"],
        "reasoning": analysis["avg_reasoning_score"],
        "groundedness": analysis["avg_groundedness_score"],
    }
    analysis["primary_bottleneck"] = min(avgs, key=avgs.get)
    analysis["score_summary"] = avgs

    return analysis


def compare_runs(
    run_paths: dict[str, str],
    output_path: str = "results/comparison.json",
):
    """Compare error analyses across multiple runs.

    Args:
        run_paths: dict of {run_name: analysis_json_path}
    """
    comparison = {}
    for name, path in run_paths.items():
        with open(path) as f:
            data = json.load(f)

        comparison[name] = {
            "accuracy": data["correct"] / data["total_samples"] if data["total_samples"] else 0,
            "diagnosis_distribution": data.get("diagnosis_distribution", {}),
            "score_summary": data.get("score_summary", {}),
            "primary_bottleneck": data.get("primary_bottleneck", "unknown"),
        }

    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Print comparison table
    print(f"\n{'='*70}")
    print("RUN COMPARISON")
    print(f"{'='*70}")
    print(f"{'Run':<20} {'Acc':>8} {'Perc':>8} {'Reas':>8} {'Ground':>8} {'Bottleneck':<15}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*15}")

    for name, data in comparison.items():
        scores = data.get("score_summary", {})
        print(
            f"{name:<20} "
            f"{data['accuracy']:>7.2%} "
            f"{scores.get('perception', 0):>7.2f} "
            f"{scores.get('reasoning', 0):>7.2f} "
            f"{scores.get('groundedness', 0):>7.2f} "
            f"{data['primary_bottleneck']:<15}"
        )
    print(f"{'='*70}")

    return comparison


def print_error_taxonomy(analysis: dict):
    """Print a readable error taxonomy from analysis results."""
    print(f"\n{'='*60}")
    print("ERROR TAXONOMY")
    print(f"{'='*60}")

    dist = analysis.get("diagnosis_distribution", {})
    total = sum(dist.values())

    print(f"\nTotal errors analyzed: {total}")
    print(f"Primary bottleneck: {analysis.get('primary_bottleneck', 'unknown')}")
    print(f"\nScore averages (0-5):")
    for dim, score in analysis.get("score_summary", {}).items():
        bar = "#" * int(score) + "." * (5 - int(score))
        print(f"  {dim:<15} {score:.2f}/5.0  [{bar}]")

    print(f"\nDiagnosis breakdown:")
    for diagnosis, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total else 0
        print(f"  {diagnosis:<20} {count:>4} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Error analysis with Reward LM")
    parser.add_argument("--results", type=str, required=True, help="Path to evaluation results JSON")
    parser.add_argument("--output", type=str, default="results/error_analysis.json")
    parser.add_argument("--max-samples", type=int, default=50, help="Max errors to score (API cost)")
    parser.add_argument("--provider", type=str, default="anthropic")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--compare", nargs="+", default=None,
                        help="Compare multiple analysis files: name1:path1 name2:path2")
    args = parser.parse_args()

    if args.compare:
        run_paths = {}
        for pair in args.compare:
            name, path = pair.split(":")
            run_paths[name] = path
        compare_runs(run_paths, args.output)
        return

    scorer = RewardLMScorer(provider=args.provider, model=args.model)
    analysis = analyze_errors(args.results, scorer=scorer, max_samples=args.max_samples)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print_error_taxonomy(analysis)
    print(f"\nFull analysis saved to {output_path}")


if __name__ == "__main__":
    main()
