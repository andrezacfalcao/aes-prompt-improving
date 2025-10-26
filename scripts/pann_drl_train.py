#!/usr/bin/env python3
"""
PANN+DRL Training Script

Trains PANN with Disentangled Representation Learning (DRL):
1. CNAA pre-training (optional): Disentangle quality vs content
2. CST warmup: Train PANN on original data
3. CST fine-tuning: Train with counterfactual data

Usage:
    python scripts/pann_drl_train.py --test_prompt 2
    python scripts/pann_drl_train.py --test_prompt 2 --enable_cnaa --enable_cst
    python scripts/pann_drl_train.py --test_prompt 2 --quick_test
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import ASAPDataset, PANNDataModule
from src.evaluation.metrics import print_metrics
from src.models.pann import PromptAwareAES, PANNConfig
from src.models.drl import DRLAESAgent, DRLConfig, CNAAConfig, CSTConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PANN+DRL model")

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
        "--enable_cnaa",
        action="store_true",
        help="Enable CNAA pre-training"
    )
    parser.add_argument(
        "--enable_cst",
        action="store_true",
        help="Enable CST fine-tuning"
    )
    parser.add_argument(
        "--cnaa_epochs",
        type=int,
        default=5,
        help="CNAA pre-training epochs"
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=2,
        help="CST warmup epochs"
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=3,
        help="CST fine-tuning epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size"
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
        help="Log directory"
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
        help="Quick test mode (1 epoch each phase)"
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


def main():
    """Main training function."""
    args = parse_args()

    set_seed(args.seed)

    if args.quick_test:
        args.cnaa_epochs = 1
        args.warmup_epochs = 1
        args.finetune_epochs = 1
        args.batch_size = 16
        args.enable_cnaa = True
        args.enable_cst = True
        print("\n⚡ QUICK TEST MODE: 1 epoch per phase, batch_size=16")

    if not args.enable_cnaa and not args.enable_cst:
        print("\nℹ️  No DRL component specified. Defaulting to CST only.")
        args.enable_cst = True

    experiment_name = f"pann_drl_p{args.test_prompt}"
    if args.enable_cnaa and args.enable_cst:
        experiment_name += "_full"
    elif args.enable_cnaa:
        experiment_name += "_cnaa"
    elif args.enable_cst:
        experiment_name += "_cst"

    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("PANN+DRL Training - Cross-Prompt AES")
    print("="*70)
    print(f"Test prompt: {args.test_prompt}")
    print(f"Train prompts: {[p for p in range(1, 9) if p != args.test_prompt]}")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"CNAA enabled: {args.enable_cnaa}")
    print(f"CST enabled: {args.enable_cst}")
    if args.enable_cnaa:
        print(f"  CNAA epochs: {args.cnaa_epochs}")
    if args.enable_cst:
        print(f"  CST warmup epochs: {args.warmup_epochs}")
        print(f"  CST finetune epochs: {args.finetune_epochs}")
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

    train_data_dict = data_module.prepare_for_drl(train_df)
    train_data_dict['train_loader'] = train_loader

    print(f"\nInitializing PANN model...")
    pann_config = PANNConfig(bert_model=args.model_name)
    model = PromptAwareAES(pann_config)

    cnaa_config = CNAAConfig(
        pretrain_epochs=args.cnaa_epochs,
        pretrain_learning_rate=5e-5
    )

    cst_config = CSTConfig(
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        warmup_learning_rate=5e-5,
        finetune_learning_rate=3e-5
    )

    drl_config = DRLConfig(
        cnaa=cnaa_config,
        cst=cst_config,
        device=args.device
    )

    print("Initializing DRL agent...")
    drl_agent = DRLAESAgent(
        model=model,
        tokenizer=tokenizer,
        config=drl_config
    )

    print("\n" + "="*70)
    print("Starting DRL Training Pipeline")
    print("="*70 + "\n")

    history = {}

    if args.enable_cnaa:
        print("\n" + "="*70)
        print("PHASE 1: CNAA Pre-training")
        print("="*70)

        cnaa_history = drl_agent.pretrain_cnaa(train_data_dict)
        history['cnaa'] = cnaa_history

        cnaa_checkpoint_path = output_dir / "cnaa_checkpoint.pt"
        drl_agent.save(str(cnaa_checkpoint_path))
        print(f"\n✓ CNAA checkpoint saved to {cnaa_checkpoint_path}")

    if args.enable_cst:
        print("\n" + "="*70)
        print("PHASE 2: CST Warmup + Fine-tuning")
        print("="*70)

        warmup_history = drl_agent.warmup_training(train_loader, val_loader)
        history['warmup'] = warmup_history

        warmup_checkpoint_path = output_dir / "warmup_checkpoint.pt"
        drl_agent.save(str(warmup_checkpoint_path))
        print(f"\n✓ Warmup checkpoint saved to {warmup_checkpoint_path}")

        print("\n" + "-"*70)
        print("Generating counterfactual data...")
        print("-"*70)

        cf_data = drl_agent.generate_and_merge_counterfactuals(train_data_dict)

        import pickle
        with open(output_dir / "counterfactual_data.pkl", "wb") as f:
            pickle.dump(cf_data, f)
        print(f"✓ Counterfactual data saved ({len(cf_data)} samples)")
    
        print("\nℹ️  CST fine-tuning requires additional implementation")
        print("   (counterfactual data loader creation)")

    print("\n" + "="*70)
    print("Final Evaluation on Test Set")
    print("="*70)

    test_metrics = drl_agent.evaluate(
        test_loader,
        asap_dataset=asap_dataset,
        prompt_id=args.test_prompt
    )

    print_metrics(test_metrics, f"Test Results - Prompt {args.test_prompt}")

    results = {
        'test_prompt': args.test_prompt,
        'train_prompts': source_prompts,
        'model': args.model_name,
        'drl_config': {
            'cnaa_enabled': args.enable_cnaa,
            'cst_enabled': args.enable_cst,
            'cnaa_epochs': args.cnaa_epochs if args.enable_cnaa else 0,
            'warmup_epochs': args.warmup_epochs if args.enable_cst else 0,
            'finetune_epochs': args.finetune_epochs if args.enable_cst else 0,
        },
        'test_metrics': test_metrics,
        'history': history
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    final_checkpoint_path = output_dir / "final_model.pt"
    drl_agent.save(str(final_checkpoint_path))

    print(f"\n✓ Results saved to {output_dir}")
    print(f"✓ Final model saved to {final_checkpoint_path}")

    print("\n" + "="*70)
    print("Training completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
