"""
Data preparation script for ASAP dataset
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ASAPDataset  # noqa: E402

PROMPT_IDS = tuple(range(1, 9))
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_VAL_SIZE = 0.1


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Prepare ASAP dataset for training"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/raw",
        help="Path to raw ASAP dataset or training TSV file"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed",
        help="Output directory for processed data splits"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=DEFAULT_VAL_SIZE,
        help="Validation set size (fraction of training data)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits"
    )

    parser.add_argument(
        "--format",
        choices=("csv", "parquet"),
        default="csv",
        help="File format to persist processed splits"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if splits already exist"
    )

    return parser.parse_args()


def resolve_training_file(input_path: str) -> Path:
    """Resolve the path to the ASAP training TSV file"""
    path = Path(input_path)

    if path.is_dir():
        candidate = path / "training_set_rel3.tsv"
        if candidate.exists():
            return candidate

        matches = list(path.glob("**/training_set_rel3.tsv"))
        if matches:
            print(
                f"Found training file in subdirectory: {matches[0]}"
            )
            return matches[0]

        raise FileNotFoundError(
            "Could not find 'training_set_rel3.tsv' inside the provided directory"
        )

    if path.suffix.lower() not in {".tsv"}:
        raise ValueError(
            "Input path must point to a TSV file or a directory containing 'training_set_rel3.tsv'"
        )

    return path


def ensure_output_dir(path: Path) -> None:
    """Ensure the output directory exists"""
    path.mkdir(parents=True, exist_ok=True)


def split_exists(prompt_dir: Path, fmt: str) -> bool:
    """Check whether all expected splits exist for a prompt"""
    return all((prompt_dir / f"{split_name}.{fmt}").exists() for split_name in SPLIT_NAMES)


def save_dataframe(df: pd.DataFrame, file_path: Path, fmt: str) -> None:
    """Persist a dataframe to disk using the desired format"""
    if fmt == "csv":
        df.to_csv(file_path, index=False)
    elif fmt == "parquet":
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def load_dataframe(file_path: Path, fmt: str) -> pd.DataFrame:
    """Load an existing dataframe from disk"""
    if fmt == "csv":
        return pd.read_csv(file_path)
    if fmt == "parquet":
        return pd.read_parquet(file_path)
    raise ValueError(f"Unsupported format: {fmt}")


def sanitize_dict(data: Dict) -> Dict:
    """Convert numpy types to native Python types for JSON serialization"""
    sanitized = {}
    for key, value in data.items():
        key_str = str(key)
        if isinstance(value, dict):
            sanitized[key_str] = sanitize_dict(value)
        elif isinstance(value, (np.generic,)):  
            sanitized[key_str] = value.item()
        else:
            sanitized[key_str] = value
    return sanitized


def compute_split_sizes(prompt_dir: Path, fmt: str) -> Dict[str, int]:
    """Get the number of rows for each split without regenerating them"""
    sizes: Dict[str, int] = {}
    for split_name in SPLIT_NAMES:
        split_file = prompt_dir / f"{split_name}.{fmt}"
        if not split_file.exists():
            sizes[split_name] = 0
            continue
        df = load_dataframe(split_file, fmt)
        sizes[split_name] = int(len(df))
    return sizes


def generate_splits(
    dataset: ASAPDataset,
    prompt_id: int,
    val_size: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate train, validation and test splits for a given target prompt"""
    source_prompts = [pid for pid in PROMPT_IDS if pid != prompt_id]
    train_df, val_df, test_df = dataset.get_cross_prompt_split(
        source_prompts=source_prompts,
        target_prompt=prompt_id,
        val_size=val_size,
        random_seed=seed
    )
    return train_df, val_df, test_df


def persist_prompt_splits(
    output_dir: Path,
    dataset: ASAPDataset,
    val_size: float,
    seed: int,
    fmt: str,
    force: bool
) -> Dict[str, Dict[str, object]]:
    """Persist splits for all prompts and return summary metadata"""
    split_summary: Dict[str, Dict[str, object]] = {}

    for prompt_id in PROMPT_IDS:
        prompt_dir = output_dir / f"prompt_{prompt_id}"
        ensure_output_dir(prompt_dir)

        entry_key = f"prompt_{prompt_id}"
        entry_summary: Dict[str, object] = {
            "target_prompt": prompt_id,
            "source_prompts": [pid for pid in PROMPT_IDS if pid != prompt_id],
        }

        already_exists = split_exists(prompt_dir, fmt)
        if already_exists and not force:
            print(f"✓ Splits already exist for prompt {prompt_id}; skipping regeneration")
            entry_summary["sizes"] = compute_split_sizes(prompt_dir, fmt)
            entry_summary["regenerated"] = False
            split_summary[entry_key] = entry_summary
            continue

        print(f"→ Generating splits for prompt {prompt_id} (val_size={val_size})")
        train_df, val_df, test_df = generate_splits(dataset, prompt_id, val_size, seed)

        for split_name, df in zip(SPLIT_NAMES, (train_df, val_df, test_df)):
            file_path = prompt_dir / f"{split_name}.{fmt}"
            save_dataframe(df, file_path, fmt)

        entry_summary["sizes"] = {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df))
        }
        entry_summary["regenerated"] = True
        split_summary[entry_key] = entry_summary

    return split_summary


def main():
    """Main data preparation function"""
    args = parse_args()

    training_file = resolve_training_file(args.input_path)
    output_dir = Path(args.output_path)
    ensure_output_dir(output_dir)

    print("============================================")
    print("ASAP DATA PREPARATION")
    print("============================================")
    print(f"Input file : {training_file}")
    print(f"Output dir : {output_dir}")
    print(f"Format     : {args.format}")
    print(f"Val size   : {args.val_size}")
    print(f"Seed       : {args.seed}")
    if args.force:
        print("Force regen: enabled")
    print("============================================\n")

    asap_dataset = ASAPDataset(str(training_file))
    asap_dataset.load_data()

    split_summary = persist_prompt_splits(
        output_dir=output_dir,
        dataset=asap_dataset,
        val_size=args.val_size,
        seed=args.seed,
        fmt=args.format,
        force=args.force
    )

    metadata = {
        "source_file": str(training_file.resolve()),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "format": args.format,
        "val_size": args.val_size,
        "seed": args.seed,
        "score_stats": sanitize_dict(asap_dataset.score_stats),
        "splits": split_summary
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✓ Data preparation completed successfully")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
