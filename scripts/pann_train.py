#!/usr/bin/env python3
"""
PANN Training Script

Trains Prompt-Aware Neural Network (PANN) for cross-prompt AES.
Leave-one-out evaluation: Train on 7 prompts, test on 1.

Usage:
    python scripts/pann_train.py --test_prompt 2
    python scripts/pann_train.py --test_prompt 2 --epochs 10
    python scripts/pann_train.py --test_prompt 2 --quick_test  # Fast test (1 epoch)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import ASAPDataset, PANNDataModule
from src.evaluation.metrics import evaluate_predictions, print_metrics
from src.models.pann import PromptAwareAES, PANNConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PANN model")

    parser.add_argument(
        "--test_prompt",
        type=int,
        required=True,
        choices=list(range(1, 9)),
        help="Prompt ID for testing (1-8)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/training_set_rel3.tsv",
        help="Path to ASAP dataset"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="BERT model name"
    )
    parser.add_argument(
        "--freeze_bert",
        action="store_true",
        help="Freeze BERT encoder"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience"
    )

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
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Quick test mode (1 epoch, small data)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, optimizer, scheduler, device, max_grad_norm):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    n_batches = 0

    for batch in train_loader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        prompt_input_ids = batch['prompt_input_ids'].to(device)
        prompt_attention_mask = batch['prompt_attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            labels=labels
        )

        loss = outputs['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss


def evaluate_epoch(model, val_loader, device, asap_dataset, test_prompt):
    """Evaluate on validation set."""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            prompt_input_ids = batch['prompt_input_ids'].to(device)
            prompt_attention_mask = batch['prompt_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask
            )

            predictions = outputs['predictions']

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    import numpy as np
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Compute metrics on normalized scores
    metrics = evaluate_predictions(
        all_labels,
        all_predictions,
        round_for_qwk=False
    )

    return metrics


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    if args.quick_test:
        args.epochs = 1
        args.batch_size = 16
        print("\n⚡ QUICK TEST MODE: 1 epoch, batch_size=16")

    output_dir = Path(args.output_dir) / f"pann_p{args.test_prompt}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) / f"pann_p{args.test_prompt}"
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    print("\n" + "="*70)
    print("PANN Training - Cross-Prompt AES")
    print("="*70)
    print(f"Test prompt: {args.test_prompt}")
    print(f"Train prompts: {[p for p in range(1, 9) if p != args.test_prompt]}")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*70 + "\n")

    print("Loading ASAP dataset...")
    asap_dataset = ASAPDataset(args.data_path)
    asap_dataset.load_data()

    source_prompts = [p for p in range(1, 9) if p != args.test_prompt]
    train_df, val_df, test_df = asap_dataset.get_cross_prompt_split(
        source_prompts=source_prompts,
        target_prompt=args.test_prompt,
        val_size=0.1,
        random_seed=args.seed
    )

    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data_module = PANNDataModule(
        asap_dataset=asap_dataset,
        tokenizer=tokenizer,
        max_essay_length=512,
        max_prompt_length=128,
        batch_size=args.batch_size,
        num_workers=0
    )

    loaders = data_module.create_dataloaders(train_df, val_df, test_df)
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    print(f"\nInitializing PANN model...")
    pann_config = PANNConfig(
        bert_model=args.model_name,
        freeze_bert=args.freeze_bert
    )
    model = PromptAwareAES(pann_config)
    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")

    best_val_qwk = -1.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_qwk': [],
        'val_pearson': [],
        'val_spearman': [],
        'val_rmse': [],
        'val_mae': []
    }

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 70)

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            args.device, args.max_grad_norm
        )

        val_metrics = evaluate_epoch(
            model, val_loader, args.device,
            asap_dataset, args.test_prompt
        )

        history['train_loss'].append(train_loss)
        history['val_qwk'].append(val_metrics['qwk'])
        history['val_pearson'].append(val_metrics['pearson'])
        history['val_spearman'].append(val_metrics['spearman'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('QWK/val', val_metrics['qwk'], epoch)
        writer.add_scalar('Pearson/val', val_metrics['pearson'], epoch)
        writer.add_scalar('Spearman/val', val_metrics['spearman'], epoch)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val QWK: {val_metrics['qwk']:.4f} | "
              f"Pearson: {val_metrics['pearson']:.4f} | "
              f"Spearman: {val_metrics['spearman']:.4f}")
        print(f"Val RMSE: {val_metrics['rmse']:.4f} | "
              f"MAE: {val_metrics['mae']:.4f}")

        if val_metrics['qwk'] > best_val_qwk:
            best_val_qwk = val_metrics['qwk']
            patience_counter = 0

            print(f"✓ New best QWK: {best_val_qwk:.4f} - Saving model...")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_qwk': best_val_qwk,
                'config': pann_config
            }, output_dir / 'best_model.pt')

        else:
            patience_counter += 1
            print(f"✗ No improvement ({patience_counter}/{args.early_stopping_patience})")

            if patience_counter >= args.early_stopping_patience:
                print("\nEarly stopping triggered!")
                break

        print()

    print("\n" + "="*70)
    print("Loading best model for final evaluation...")
    print("="*70 + "\n")

    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Evaluating on test set...")
    test_metrics = evaluate_epoch(
        model, test_loader, args.device,
        asap_dataset, args.test_prompt
    )

    print_metrics(test_metrics, f"Final Test Results - Prompt {args.test_prompt}")

    results = {
        'test_prompt': args.test_prompt,
        'train_prompts': source_prompts,
        'model': args.model_name,
        'epochs_trained': epoch + 1,
        'best_val_qwk': best_val_qwk,
        'test_metrics': test_metrics,
        'history': history
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}")
    print(f"✓ Logs saved to {log_dir}")

    writer.close()

    print("\n" + "="*70)
    print("Training completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
