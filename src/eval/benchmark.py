"""Full benchmark evaluation suite.

Runs evaluation across all benchmarks and generates comparison tables.
"""

import argparse
import json
from pathlib import Path

from src.eval.baseline import evaluate_dataset


BENCHMARK_DATASETS = {
    "geoqa": {
        "path": "data/processed/trl/geoqa",
        "description": "Geometry diagram reasoning",
        "category": "geometry",
    },
    "mathvista": {
        "path": "data/raw/mathvista",
        "description": "Broad visual math",
        "category": "math",
    },
    "geometry3k": {
        "path": "data/raw/geometry3k",
        "description": "Geometry problems with diagrams",
        "category": "geometry",
    },
}


def run_benchmarks(
    model,
    processor,
    benchmarks: list[str] | None = None,
    max_samples: int | None = None,
    use_cot: bool = True,
    output_dir: str = "results",
) -> dict:
    """Run evaluation on multiple benchmarks.

    Returns dict of {benchmark_name: metrics}.
    """
    if benchmarks is None:
        benchmarks = list(BENCHMARK_DATASETS.keys())

    all_results = {}
    for name in benchmarks:
        if name not in BENCHMARK_DATASETS:
            print(f"Skipping unknown benchmark: {name}")
            continue

        info = BENCHMARK_DATASETS[name]
        dataset_path = info["path"]

        if not Path(dataset_path).exists():
            print(f"Skipping {name}: dataset not found at {dataset_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {name} — {info['description']}")
        print(f"{'='*60}")

        metrics = evaluate_dataset(
            model, processor, dataset_path,
            use_cot=use_cot,
            max_samples=max_samples,
        )

        all_results[name] = {
            "accuracy": metrics["accuracy"],
            "format_compliance": metrics["format_compliance"],
            "total": metrics["total"],
            "correct": metrics["correct"],
            "category": info["category"],
        }

        # Save per-benchmark detailed results
        output_path = Path(output_dir) / f"{name}_detailed.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"  Accuracy: {metrics['accuracy']:.2%}")

    return all_results


def print_comparison_table(results: dict[str, dict]):
    """Print a formatted comparison table."""
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"{'Benchmark':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*8}")

    for name, metrics in sorted(results.items()):
        acc = f"{metrics['accuracy']:.2%}"
        print(f"{name:<20} {acc:>10} {metrics['correct']:>10} {metrics['total']:>8}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Full benchmark evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-cot", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--tag", type=str, default="baseline", help="Tag for this run (e.g. sft, grpo)")
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # Load model
    if args.lora_path:
        from src.utils.model import load_model_for_inference
        model, processor = load_model_for_inference(
            args.model, lora_path=args.lora_path
        )
    else:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

    # Run benchmarks
    results = run_benchmarks(
        model, processor,
        benchmarks=args.benchmarks,
        max_samples=args.max_samples,
        use_cot=not args.no_cot,
        output_dir=args.output_dir,
    )

    # Save summary
    summary_path = Path(args.output_dir) / f"summary_{args.tag}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print_comparison_table(results)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
