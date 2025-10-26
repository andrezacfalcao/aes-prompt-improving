"""
Training script for cross-prompt automated essay scoring
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ASAPDataset, EssayDataset  # noqa: E402
from src.data.tokenizer import get_tokenizer  # noqa: E402
from src.models.transformer_aes import TransformerAES  # noqa: E402
from src.evaluation.metrics import evaluate_predictions, print_metrics  # noqa: E402


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train transformer model for cross-prompt AES"
    )

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/training_set_rel3.tsv",
        help="Path to ASAP TSV file"
    )

    parser.add_argument(
        "--target_prompt",
        type=int,
        required=True,
        help="Target prompt ID for testing (1-8)"
    )

    # Model
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
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Warmup steps for learning rate scheduler"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping"
    )

    # Validation
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Validation set size (fraction of training data)"
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience (epochs)"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory"
    )

    parser.add_argument(
        "--experiment_prefix",
        type=str,
        default=None,
        help="Optional prefix for checkpoint/log directories"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training"
    )

    parser.add_argument(
        "--processed_dir",
        type=str,
        default=None,
        help="Directory containing preprocessed splits (from prepare_data.py)"
    )

    parser.add_argument(
        "--processed_format",
        type=str,
        choices=("csv", "parquet"),
        default=None,
        help="Format of processed splits (overrides metadata)"
    )

    parser.add_argument(
        "--use_processed_splits",
        action="store_true",
        help="Use processed splits instead of regenerating them on-the-fly"
    )

    parser.add_argument(
        "--train_subset",
        type=int,
        default=None,
        help="Limit number of training samples (for quick experiments)"
    )

    parser.add_argument(
        "--val_subset",
        type=int,
        default=None,
        help="Limit number of validation samples"
    )

    parser.add_argument(
        "--test_subset",
        type=int,
        default=None,
        help="Limit number of test samples"
    )

    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip training if results.json already exists"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers"
    )

    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable DataLoader pin_memory (useful for CUDA)"
    )

    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Disable tqdm progress bars"
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def sanitize_model_name(model_name: str) -> str:
    """Create a filesystem-friendly tag for a model name"""
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", model_name)
    sanitized = sanitized.strip("_").lower()
    return sanitized or "model"


def limit_dataframe(df: pd.DataFrame, subset_size: Optional[int], seed: int) -> pd.DataFrame:
    """Optionally limit dataframe to a subset of rows"""
    if subset_size is None or subset_size <= 0 or subset_size >= len(df):
        return df
    return df.sample(n=subset_size, random_state=seed).reset_index(drop=True)


def infer_processed_format(processed_dir: Path, explicit: Optional[str]) -> Optional[str]:
    """Infer processed split format from metadata or explicit flag"""
    if explicit:
        return explicit

    metadata_path = processed_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            fmt = metadata.get("format")
            if fmt in {"csv", "parquet"}:
                return fmt
        except Exception as exc:
            print(f"Warning: failed to read metadata at {metadata_path}: {exc}")
    return None


def load_processed_split(processed_dir: Path, prompt_id: int, split: str, fmt: str) -> Optional[pd.DataFrame]:
    """Load a specific split for a prompt from the processed directory"""
    split_path = processed_dir / f"prompt_{prompt_id}" / f"{split}.{fmt}"
    if not split_path.exists():
        return None

    if fmt == "csv":
        return pd.read_csv(split_path)
    if fmt == "parquet":
        return pd.read_parquet(split_path)
    raise ValueError(f"Unsupported processed format: {fmt}")


def load_processed_splits(
    processed_dir: Path,
    prompt_id: int,
    fmt: str
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Load train/val/test splits from processed directory"""
    train_df = load_processed_split(processed_dir, prompt_id, "train", fmt)
    val_df = load_processed_split(processed_dir, prompt_id, "val", fmt)
    test_df = load_processed_split(processed_dir, prompt_id, "test", fmt)

    if train_df is None or val_df is None or test_df is None:
        return None
    return train_df, val_df, test_df


def train_epoch(
    model: TransformerAES,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    show_progress: bool = True
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc="Training", disable=not show_progress)

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['score'].to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss'] / gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss'] / gradient_accumulation_steps
            loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps

        if (step + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        if show_progress:
            progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss


def evaluate(
    model: TransformerAES,
    dataloader: DataLoader,
    device: torch.device,
    asap_dataset: ASAPDataset,
    prompt_id: int,
    show_progress: bool = True,
    return_details: bool = False
):
    """Evaluate model"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_essay_ids = []
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Evaluating", disable=not show_progress)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['score'].to(device)

            outputs = model(input_ids, attention_mask, labels)

            all_predictions.extend(outputs['predictions'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if isinstance(batch['essay_id'], list):
                all_essay_ids.extend(batch['essay_id'])
            else:
                essay_ids = batch['essay_id']
                if torch.is_tensor(essay_ids):
                    all_essay_ids.extend(essay_ids.cpu().numpy().tolist())
                elif isinstance(essay_ids, np.ndarray):
                    all_essay_ids.extend(essay_ids.tolist())
                else:
                    all_essay_ids.append(essay_ids)
            total_loss += outputs['loss'].item()

    avg_loss = total_loss / max(len(dataloader), 1)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    predictions_original = [
        asap_dataset.denormalize_score(pred, prompt_id)
        for pred in all_predictions
    ]
    labels_original = [
        asap_dataset.denormalize_score(label, prompt_id)
        for label in all_labels
    ]

    metrics = evaluate_predictions(
        np.array(labels_original),
        np.array(predictions_original)
    )
    metrics['loss'] = avg_loss

    if return_details:
        details = {
            'essay_ids': all_essay_ids,
            'predictions_norm': all_predictions,
            'labels_norm': all_labels,
            'predictions_original': predictions_original,
            'labels_original': labels_original,
        }
        return metrics, details

    return metrics


def get_experiment_paths(args) -> Tuple[Path, Path, str]:
    """Determine output/log directories and experiment name"""
    model_tag = sanitize_model_name(args.model_name)
    prefix = f"{args.experiment_prefix}_" if args.experiment_prefix else ""
    experiment_name = f"{prefix}{model_tag}_p{args.target_prompt}"

    output_dir = Path(args.output_dir) / experiment_name
    log_dir = Path(args.log_dir) / experiment_name

    return output_dir, log_dir, experiment_name


def maybe_skip_experiment(args, output_dir: Path) -> bool:
    """Return True if experiment should be skipped"""
    if not args.skip_if_exists:
        return False

    results_file = output_dir / "results.json"
    if results_file.exists():
        print(f"✓ Results already exist at {results_file}; skipping as requested")
        return True
    return False


def prepare_splits(
    args,
    asap_dataset: ASAPDataset
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Prepare train/val/test DataFrames either from processed splits or raw data"""
    split_sizes: Dict[str, int] = {}

    if args.use_processed_splits and args.processed_dir:
        processed_dir = Path(args.processed_dir)
        fmt = infer_processed_format(processed_dir, args.processed_format)
        if fmt:
            splits = load_processed_splits(processed_dir, args.target_prompt, fmt)
            if splits is not None:
                print(
                    f"✓ Loaded processed splits for prompt {args.target_prompt} from {processed_dir}"
                )
                train_df, val_df, test_df = splits
                split_sizes = {
                    'train': len(train_df),
                    'val': len(val_df),
                    'test': len(test_df)
                }
                return train_df, val_df, test_df, split_sizes
            else:
                print(
                    f"Warning: processed splits not found for prompt {args.target_prompt}. "
                    "Falling back to dynamic split."
                )
        else:
            print("Warning: could not infer processed split format; falling back to dynamic split")

    print("Creating cross-prompt split dynamically...")
    source_prompts = [pid for pid in range(1, 9) if pid != args.target_prompt]
    train_df, val_df, test_df = asap_dataset.get_cross_prompt_split(
        source_prompts=source_prompts,
        target_prompt=args.target_prompt,
        val_size=args.val_size,
        random_seed=args.seed
    )

    split_sizes = {
        'train': len(train_df),
        'val': len(val_df),
        'test': len(test_df)
    }
    return train_df, val_df, test_df, split_sizes


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    args
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch dataloaders from dataframes"""
    train_dataset = EssayDataset(train_df, tokenizer, args.max_length)
    val_dataset = EssayDataset(val_df, tokenizer, args.max_length) if len(val_df) > 0 else None
    test_dataset = EssayDataset(test_df, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    return train_loader, val_loader, test_loader


def main():
    """Main training function"""
    args = parse_args()
    set_seed(args.seed)

    show_progress = not args.disable_tqdm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    output_dir, log_dir, experiment_name = get_experiment_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if maybe_skip_experiment(args, output_dir):
        return

    with open(output_dir / 'args.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nExperiment: {experiment_name}")
    print(f"Target prompt: {args.target_prompt}")
    print(f"Source prompts: {[i for i in range(1, 9) if i != args.target_prompt]}")
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}\n")

    asap_dataset = ASAPDataset(args.data_path)
    asap_dataset.load_data()

    train_df, val_df, test_df, split_sizes = prepare_splits(args, asap_dataset)

    train_df = limit_dataframe(train_df, args.train_subset, args.seed)
    val_df = limit_dataframe(val_df, args.val_subset, args.seed) if len(val_df) > 0 else val_df
    test_df = limit_dataframe(test_df, args.test_subset, args.seed)

    tokenizer = get_tokenizer(args.model_name)
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, tokenizer, args)

    print(f"\nDataset sizes (after optional subsets):")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val: {len(val_loader.dataset) if val_loader else 0}")
    print(f"  Test: {len(test_loader.dataset)}\n")

    model = TransformerAES(
        model_name=args.model_name,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    total_train_steps = max(len(train_loader), 1) * args.num_epochs // max(args.gradient_accumulation_steps, 1)
    total_train_steps = max(total_train_steps, 1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_train_steps
    )

    scaler = torch.cuda.amp.GradScaler() if args.fp16 and torch.cuda.is_available() else None

    writer = SummaryWriter(log_dir=str(log_dir))

    best_qwk = -1.0
    patience_counter = 0
    history_records: List[Dict[str, float]] = []

    for epoch in range(args.num_epochs):
        print(f"{'='*60}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*60}")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            args.gradient_accumulation_steps,
            args.max_grad_norm,
            scaler,
            show_progress=show_progress
        )

        print(f"Train Loss: {train_loss:.4f}")

        if val_loader is not None and len(val_loader.dataset) > 0:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                asap_dataset,
                args.target_prompt,
                show_progress=show_progress
            )
            print(f"\nValidation Metrics:")
            print_metrics(val_metrics)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('QWK/val', val_metrics['qwk'], epoch)
            writer.add_scalar('Pearson/val', val_metrics['pearson'], epoch)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

            current_qwk = val_metrics['qwk']
        else:
            current_qwk = train_loss * -1  # ensure model saves at least once when no val set
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        record: Dict[str, float] = {
            'epoch': float(epoch),
            'train_loss': float(train_loss),
            'lr': float(scheduler.get_last_lr()[0])
        }

        if val_loader is not None and len(val_loader.dataset) > 0:
            for key, value in val_metrics.items():
                record[f'val_{key}'] = float(value)

        history_records.append(record)

        if current_qwk > best_qwk:
            best_qwk = current_qwk
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_qwk': best_qwk,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"✓ Saved best model (metric: {best_qwk:.4f})")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.early_stopping_patience}")

        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

        print()

    history_path = output_dir / 'history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_records, f, indent=2)

    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}\n")

    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        asap_dataset,
        args.target_prompt,
        show_progress=show_progress
    )
    print("Test Metrics:")
    print_metrics(test_metrics)

    results = {
        'experiment_name': experiment_name,
        'target_prompt': args.target_prompt,
        'source_prompts': [i for i in range(1, 9) if i != args.target_prompt],
        'best_val_qwk': best_qwk,
        'test_metrics': test_metrics,
        'total_epochs': epoch + 1,
        'split_sizes': split_sizes,
        'actual_sizes': {
            'train': len(train_loader.dataset),
            'val': len(val_loader.dataset) if val_loader else 0,
            'test': len(test_loader.dataset)
        },
        'subset': {
            'train_subset': args.train_subset,
            'val_subset': args.val_subset,
            'test_subset': args.test_subset
        }
    }

    with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")
    print(f"Model saved to {output_dir / 'best_model.pt'}")
    print(f"TensorBoard logs: {log_dir}")

    writer.close()


if __name__ == "__main__":
    main()
