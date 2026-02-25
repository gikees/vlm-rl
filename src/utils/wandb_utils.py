"""Weights & Biases logging utilities for experiment tracking."""

import json
from pathlib import Path


def log_eval_results(run_name: str, results_path: str, project: str = "vlm-rl"):
    """Log evaluation results to an existing or new WandB run."""
    import wandb

    with open(results_path) as f:
        data = json.load(f)

    run = wandb.init(project=project, name=run_name, job_type="eval")

    # Log summary metrics
    summary = {k: v for k, v in data.items() if k != "results"}
    wandb.log(summary)

    # Log per-sample results as a table
    if "results" in data:
        columns = ["idx", "question", "ground_truth", "predicted", "correct", "ability"]
        table_data = []
        for r in data["results"]:
            table_data.append([
                r.get("idx", ""),
                r.get("question", "")[:100],
                r.get("ground_truth", ""),
                r.get("predicted", ""),
                r.get("correct", False),
                r.get("ability", ""),
            ])
        wandb.log({"eval_results": wandb.Table(columns=columns, data=table_data)})

    run.finish()


def log_error_analysis(run_name: str, analysis_path: str, project: str = "vlm-rl"):
    """Log error analysis results to WandB."""
    import wandb

    with open(analysis_path) as f:
        data = json.load(f)

    run = wandb.init(project=project, name=run_name, job_type="analysis")

    # Log diagnosis distribution as bar chart
    if "diagnosis_distribution" in data:
        dist = data["diagnosis_distribution"]
        wandb.log({
            "diagnosis_distribution": wandb.plot.bar(
                wandb.Table(
                    columns=["diagnosis", "count"],
                    data=[[k, v] for k, v in dist.items()],
                ),
                "diagnosis", "count",
                title="Error Diagnosis Distribution",
            )
        })

    # Log score summary
    if "score_summary" in data:
        wandb.log({"score_summary": data["score_summary"]})

    wandb.log({
        "total_samples": data.get("total_samples", 0),
        "accuracy": data.get("correct", 0) / max(data.get("total_samples", 1), 1),
        "primary_bottleneck": data.get("primary_bottleneck", "unknown"),
    })

    run.finish()


def log_comparison(
    results_dir: str = "results",
    project: str = "vlm-rl",
):
    """Log a comparison table across all runs to WandB."""
    import wandb

    results_dir = Path(results_dir)
    summaries = {}

    for f in sorted(results_dir.glob("summary_*.json")):
        tag = f.stem.replace("summary_", "")
        with open(f) as fh:
            summaries[tag] = json.load(fh)

    if not summaries:
        print("No summary files found")
        return

    run = wandb.init(project=project, name="comparison", job_type="comparison")

    # Build comparison table
    benchmarks = set()
    for data in summaries.values():
        benchmarks.update(data.keys())

    columns = ["run"] + sorted(benchmarks)
    table_data = []
    for tag, data in summaries.items():
        row = [tag]
        for b in sorted(benchmarks):
            if b in data:
                row.append(f"{data[b]['accuracy']:.2%}")
            else:
                row.append("-")
        table_data.append(row)

    wandb.log({"comparison": wandb.Table(columns=columns, data=table_data)})
    run.finish()
