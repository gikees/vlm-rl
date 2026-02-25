"""Download and cache datasets from HuggingFace."""

import argparse
from pathlib import Path

from datasets import load_dataset


DATASETS = {
    # Training datasets
    "geoqa": "leonardPKU/GEOQA_R1V_Train_8K",
    "clevr": "MMInstruction/Clevr_CoGenT_TrainA_70K_Complex",
    "multimodal-r1-8k": "lmms-lab/multimodal-open-r1-8k-verified",
    # Eval datasets
    "mathvista": "AI4Math/MathVista",
    "mathverse": "AI4Math/MathVerse",
    "hallusionbench": "lmms-lab/HallusionBench",
    # EasyR1 pre-formatted
    "geometry3k": "hiyouga/geometry3k",
}


def download_dataset(name: str, output_dir: Path, split: str | None = None):
    """Download a single dataset and save to disk."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(DATASETS.keys())}")

    hf_id = DATASETS[name]
    print(f"Downloading {name} from {hf_id}...")

    ds = load_dataset(hf_id, split=split)
    save_path = output_dir / name
    ds.save_to_disk(str(save_path))
    print(f"Saved {name} to {save_path}")
    return ds


def download_all(output_dir: Path, training_only: bool = False):
    """Download all datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = DATASETS.keys()
    if training_only:
        targets = ["geoqa", "clevr", "multimodal-r1-8k", "geometry3k"]

    for name in targets:
        try:
            download_dataset(name, output_dir)
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            print("Continuing with other datasets...")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for VLM-RL")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", type=str, default=None, help="Download a specific dataset")
    parser.add_argument("--training-only", action="store_true", help="Only download training sets")
    args = parser.parse_args()

    if args.dataset:
        download_dataset(args.dataset, args.output_dir)
    else:
        download_all(args.output_dir, training_only=args.training_only)


if __name__ == "__main__":
    main()
