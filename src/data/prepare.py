"""Prepare datasets into formats for EasyR1 and TRL training.

EasyR1 (verl) expects parquet with columns:
    data_source, prompt, ability, reward_model, extra_info, images

TRL GRPOTrainer expects datasets with:
    prompt (chat messages), solution, image
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_from_disk

from src.data.formatting import format_prompt_for_chat


# --- GeoQA ---


def prepare_geoqa(raw_dir: Path, output_dir: Path):
    """Prepare GeoQA dataset for training.

    Source: AI-ModelScope/GEOQA_R1V_Train_8K
    Fields: image, problem, solution
    """
    ds = load_from_disk(str(raw_dir / "geoqa"))
    if isinstance(ds, dict):
        ds = ds["train"]

    print(f"GeoQA: {len(ds)} samples")

    # TRL format
    trl_records = []
    for i, row in enumerate(ds):
        question = row.get("problem", row.get("question", ""))
        if "<image>" not in question:
            question = "<image>\n" + question

        trl_records.append({
            "prompt": format_prompt_for_chat(question),
            "solution": str(row.get("solution", row.get("answer", ""))),
            "image": row.get("image"),
            "data_source": "geoqa",
            "ability": "geometry",
            "idx": i,
        })

    trl_ds = Dataset.from_list(trl_records)
    trl_path = output_dir / "trl" / "geoqa"
    trl_path.mkdir(parents=True, exist_ok=True)
    trl_ds.save_to_disk(str(trl_path))
    print(f"  TRL format saved to {trl_path}")

    # EasyR1/verl parquet format
    verl_records = []
    for i, row in enumerate(ds):
        question = row.get("problem", row.get("question", ""))
        if "<image>" not in question:
            question = "<image>\n" + question

        verl_records.append({
            "data_source": "geoqa",
            "prompt": format_prompt_for_chat(question),
            "ability": "geometry",
            "reward_model": {"ground_truth": str(row.get("solution", row.get("answer", "")))},
            "extra_info": {"split": "train", "index": i},
        })

    verl_path = output_dir / "verl" / "geoqa_train.parquet"
    verl_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(verl_records).to_parquet(str(verl_path))
    print(f"  verl format saved to {verl_path}")


# --- CLEVR Counting ---


def prepare_clevr(raw_dir: Path, output_dir: Path, max_samples: int = 10000):
    """Prepare CLEVR counting dataset.

    Source: lmms-lab/CLEVR-70k-Counting
    """
    ds = load_from_disk(str(raw_dir / "clevr"))
    if isinstance(ds, dict):
        ds = ds["train"]

    # Subsample for manageable size
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    print(f"CLEVR: {len(ds)} samples (subsampled to {max_samples})")

    trl_records = []
    for i, row in enumerate(ds):
        question = row.get("problem", row.get("question", ""))
        if "<image>" not in question:
            question = "<image>\n" + question

        trl_records.append({
            "prompt": format_prompt_for_chat(question),
            "solution": str(row.get("solution", row.get("answer", row.get("count", "")))),
            "image": row.get("image"),
            "data_source": "clevr",
            "ability": "counting",
            "idx": i,
        })

    trl_ds = Dataset.from_list(trl_records)
    trl_path = output_dir / "trl" / "clevr"
    trl_path.mkdir(parents=True, exist_ok=True)
    trl_ds.save_to_disk(str(trl_path))
    print(f"  TRL format saved to {trl_path}")


# --- Multimodal Open R1 8K ---


def prepare_multimodal_r1(raw_dir: Path, output_dir: Path):
    """Prepare multimodal-open-r1-8k-verified dataset.

    Source: lmms-lab/multimodal-open-r1-8k-verified
    Fields: image, problem, solution, original_question, original_answer
    """
    ds = load_from_disk(str(raw_dir / "multimodal-r1-8k"))
    if isinstance(ds, dict):
        ds = ds["train"]

    print(f"Multimodal-R1-8K: {len(ds)} samples")

    trl_records = []
    for i, row in enumerate(ds):
        question = row.get("problem", row.get("original_question", ""))
        if "<image>" not in question:
            question = "<image>\n" + question

        # Use original_answer as ground truth (cleaner than full solution)
        answer = str(row.get("original_answer", row.get("solution", "")))

        trl_records.append({
            "prompt": format_prompt_for_chat(question),
            "solution": answer,
            "image": row.get("image"),
            "data_source": "multimodal-r1-8k",
            "ability": "math",
            "idx": i,
        })

    trl_ds = Dataset.from_list(trl_records)
    trl_path = output_dir / "trl" / "multimodal-r1-8k"
    trl_path.mkdir(parents=True, exist_ok=True)
    trl_ds.save_to_disk(str(trl_path))
    print(f"  TRL format saved to {trl_path}")


# --- Geometry3K (EasyR1 native) ---


def prepare_geometry3k(raw_dir: Path, output_dir: Path):
    """Prepare geometry3k dataset (already near-EasyR1 format).

    Source: hiyouga/geometry3k
    Fields: images, problem, answer
    """
    ds = load_from_disk(str(raw_dir / "geometry3k"))

    for split_name in ["train", "validation", "test"]:
        if isinstance(ds, dict) and split_name in ds:
            split_ds = ds[split_name]
        elif split_name == "train":
            split_ds = ds
        else:
            continue

        print(f"Geometry3K ({split_name}): {len(split_ds)} samples")

        verl_records = []
        for i, row in enumerate(split_ds):
            question = row.get("problem", "")
            if "<image>" not in question:
                question = "<image>\n" + question

            verl_records.append({
                "data_source": "geometry3k",
                "prompt": format_prompt_for_chat(question),
                "ability": "geometry",
                "reward_model": {"ground_truth": str(row.get("answer", ""))},
                "extra_info": {"split": split_name, "index": i},
            })

        verl_path = output_dir / "verl" / f"geometry3k_{split_name}.parquet"
        verl_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(verl_records).to_parquet(str(verl_path))
        print(f"  verl format saved to {verl_path}")


def prepare_all(raw_dir: Path, output_dir: Path):
    """Prepare all datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    preparers = {
        "geoqa": prepare_geoqa,
        "clevr": prepare_clevr,
        "multimodal-r1-8k": prepare_multimodal_r1,
        "geometry3k": prepare_geometry3k,
    }

    for name, prep_fn in preparers.items():
        try:
            if name == "clevr":
                prep_fn(raw_dir, output_dir, max_samples=10000)
            else:
                prep_fn(raw_dir, output_dir)
        except Exception as e:
            print(f"Failed to prepare {name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for training")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--clevr-max", type=int, default=10000)
    args = parser.parse_args()

    if args.dataset:
        preparers = {
            "geoqa": prepare_geoqa,
            "clevr": lambda r, o: prepare_clevr(r, o, args.clevr_max),
            "multimodal-r1-8k": prepare_multimodal_r1,
            "geometry3k": prepare_geometry3k,
        }
        if args.dataset not in preparers:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        preparers[args.dataset](args.raw_dir, args.output_dir)
    else:
        prepare_all(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
