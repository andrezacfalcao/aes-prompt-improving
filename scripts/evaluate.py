"""
Evaluation script for cross-prompt automated essay scoring
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ASAPDataset, EssayDataset  # noqa: E402
from src.data.tokenizer import get_tokenizer  # noqa: E402
from src.models.transformer_aes import TransformerAES  # noqa: E402
from src.evaluation.metrics import evaluate_predictions, print_metrics  # noqa: E402

PROMPT_IDS = tuple(range(1, 9))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on ASAP dataset"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--target_prompt",
        type=int,
        required=True,
        help="Target prompt ID for evaluation (1-8)"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/training_set_rel3.tsv",
        help="Path to ASAP TSV file"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained model name"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Evaluation batch size"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="results.json",
        help="Output file for results"
    )

    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Directory containing precomputed dataset splits"
    )

    parser.add_argument(
        "--processed_format",
        type=str,
        choices=("csv", "parquet"),
        default=None,
        help="Format of processed splits (overrides metadata if provided)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used when rebuilding splits)"
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA even if available"
    )

    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict:
    """Load model checkpoint from disk"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain 'model_state_dict'. "
            "Ensure it was saved by the training script."
        )
    return checkpoint


def resolve_processed_format(
    processed_dir: Path,
    explicit_format: Optional[str]
) -> Optional[str]:
    """Infer processed split format from metadata or explicit flag"""
    if explicit_format:
        return explicit_format

    metadata_path = processed_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return metadata.get("format")
        except Exception as exc:
            print(f"Warning: failed to read metadata at {metadata_path}: {exc}")
    return None


def load_processed_dataframe(prompt_dir: Path, fmt: str) -> Optional[pd.DataFrame]:
    """Load a processed dataframe for the specified prompt"""
    test_file = prompt_dir / f"test.{fmt}"
    if not test_file.exists():
        return None

    if fmt == "csv":
        return pd.read_csv(test_file)
    if fmt == "parquet":
        return pd.read_parquet(test_file)

    raise ValueError(f"Unsupported processed format: {fmt}")


def build_test_dataframe(
    asap_dataset: ASAPDataset,
    target_prompt: int,
    processed_dir: Path,
    processed_format: Optional[str]
) -> pd.DataFrame:
    """Load test dataframe from processed splits or rebuild from raw data"""
    prompt_dir = processed_dir / f"prompt_{target_prompt}"
    if processed_format and prompt_dir.exists():
        df = load_processed_dataframe(prompt_dir, processed_format)
        if df is not None:
            print(f"✓ Loaded processed test split for prompt {target_prompt} ({len(df)} essays)")
            return df
        else:
            print(
                f"Warning: test split not found at {prompt_dir} with format '{processed_format}'. "
                "Falling back to on-the-fly generation."
            )

    print(f"→ Building test split on the fly for prompt {target_prompt}")
    if asap_dataset.data is None:
        raise ValueError("ASAP dataset must be loaded before building splits")

    test_df = asap_dataset.data[asap_dataset.data["essay_set"] == target_prompt].copy()
    if len(test_df) == 0:
        raise ValueError(f"No essays found for prompt {target_prompt}")

    test_df["essay"] = test_df["essay"].apply(asap_dataset.preprocess_text)
    if "normalized_score" not in test_df.columns:
        test_df["normalized_score"] = test_df.apply(
            lambda row: asap_dataset.normalize_score(row["domain1_score"], row["essay_set"]),
            axis=1
        )

    return test_df


def build_dataloader(
    test_df: pd.DataFrame,
    tokenizer_name: str,
    max_length: int,
    batch_size: int
) -> DataLoader:
    """Build evaluation dataloader for the target prompt"""
    tokenizer = get_tokenizer(tokenizer_name)
    test_dataset = EssayDataset(test_df, tokenizer, max_length)

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )


def run_evaluation(args) -> Dict[str, float]:
    """Run model evaluation and return metrics"""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    processed_dir = Path(args.processed_dir)
    processed_format = resolve_processed_format(processed_dir, args.processed_format)

    if processed_format:
        print(f"Using processed splits from {processed_dir} (format='{processed_format}')")
    else:
        print("Processed splits not configured; using raw dataset")

    print("Loading dataset...")
    asap_dataset = ASAPDataset(args.data_path)
    asap_dataset.load_data()

    test_df = build_test_dataframe(
        asap_dataset,
        args.target_prompt,
        processed_dir,
        processed_format
    )

    dataloader = build_dataloader(
        test_df,
        args.model_name,
        args.max_length,
        args.batch_size
    )

    print("Loading model...")
    model = TransformerAES(model_name=args.model_name)
    model = model.to(device)

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_predictions = []
    all_labels = []
    all_prompt_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['score'].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs['predictions'].detach().cpu().numpy()

            all_predictions.extend(preds)
            all_labels.extend(labels)
            all_prompt_ids.extend(batch['prompt_id'].cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_prompt_ids = np.array(all_prompt_ids)

    predictions_original = [
        asap_dataset.denormalize_score(pred, args.target_prompt)
        for pred in all_predictions
    ]
    labels_original = [
        asap_dataset.denormalize_score(label, args.target_prompt)
        for label in all_labels
    ]

    metrics = evaluate_predictions(
        np.array(labels_original),
        np.array(predictions_original)
    )

    print_metrics(metrics, title="Evaluation Metrics")

    results = {
        "target_prompt": args.target_prompt,
        "checkpoint": str(checkpoint_path.resolve()),
        "model_name": args.model_name,
        "metrics": metrics
    }

    return results


def main():
    """Main evaluation function"""
    args = parse_args()

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Target prompt: {args.target_prompt}")

    results = run_evaluation(args)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
