"""
Batch script to run leave-one-out experiments for all 8 prompts
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run leave-one-out experiments across prompts"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained model to use"
    )

    parser.add_argument(
        "--experiment_prefix",
        type=str,
        default=None,
        help="Optional prefix for experiment/checkpoint directories"
    )

    parser.add_argument(
        "--prompts",
        type=int,
        nargs="*",
        default=list(range(1, 9)),
        help="List of target prompts to evaluate (default: 1-8)"
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default=str(Path(__file__).parent.parent),
        help="Project base directory"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Tokenizer max length"
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
        help="Enable mixed precision"
    )

    parser.add_argument(
        "--train_subset",
        type=int,
        default=None,
        help="Optional subset for training samples"
    )

    parser.add_argument(
        "--val_subset",
        type=int,
        default=None,
        help="Optional subset for validation samples"
    )

    parser.add_argument(
        "--test_subset",
        type=int,
        default=None,
        help="Optional subset for test samples"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers"
    )

    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable DataLoader pin_memory"
    )

    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Disable progress bars"
    )

    parser.add_argument(
        "--use_processed_splits",
        action="store_true",
        help="Reuse processed data splits instead of generating on the fly"
    )

    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed splits"
    )

    parser.add_argument(
        "--processed_format",
        type=str,
        choices=("csv", "parquet"),
        default=None,
        help="Format of processed splits"
    )

    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip prompts where results already exist"
    )

    parser.add_argument(
        "--output_summary",
        type=str,
        default=None,
        help="Optional custom path for summary JSON"
    )

    return parser.parse_args()


def sanitize_model_name(model_name: str) -> str:
    """Create a filesystem-friendly tag from a model name"""
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", model_name)
    sanitized = sanitized.strip("_").lower()
    return sanitized or "model"


def derive_experiment_prefix(args) -> str:
    """Derive experiment prefix (CLI flag overrides default)"""
    if args.experiment_prefix:
        return args.experiment_prefix
    return args.model_name.replace('/', '-')


def build_experiment_stub(model_name: str, experiment_prefix: str) -> str:
    """Combine prefix and sanitized model tag (matches train.py behaviour)"""
    model_tag = sanitize_model_name(model_name)
    if experiment_prefix:
        return f"{experiment_prefix}_{model_tag}"
    return model_tag


def find_prompt_dirs(
    checkpoints_dir: Path,
    prompt_id: int,
    experiment_stub: str,
    sanitized_model: str
) -> List[Path]:
    """Locate checkpoint directories for a given prompt"""
    if not checkpoints_dir.exists():
        return []

    expected = checkpoints_dir / f"{experiment_stub}_p{prompt_id}"
    if expected.exists():
        return [expected]

    prompt_dirs: List[Path] = []
    stub_lower = experiment_stub.lower()
    sanitized_lower = sanitized_model.lower()

    for candidate in checkpoints_dir.glob(f"*p{prompt_id}"):
        if not candidate.is_dir():
            continue
        name = candidate.name.lower()
        if stub_lower in name or sanitized_lower in name:
            prompt_dirs.append(candidate)

    return sorted(prompt_dirs)


def build_train_command(
    args,
    target_prompt: int,
    base_dir: Path
) -> List[str]:
    experiment_prefix = derive_experiment_prefix(args)
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--target_prompt", str(target_prompt),
        "--data_path", "data/raw/training_set_rel3.tsv",
        "--model_name", args.model_name,
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--num_epochs", str(args.num_epochs),
        "--learning_rate", str(args.learning_rate),
        "--max_length", str(args.max_length),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
    ]

    if args.fp16:
        cmd.append("--fp16")

    if args.pin_memory:
        cmd.append("--pin_memory")

    if args.disable_tqdm:
        cmd.append("--disable_tqdm")

    if args.use_processed_splits:
        cmd.append("--use_processed_splits")
        cmd.extend(["--processed_dir", str(Path(args.processed_dir).resolve())])
        if args.processed_format:
            cmd.extend(["--processed_format", args.processed_format])

    if args.train_subset:
        cmd.extend(["--train_subset", str(args.train_subset)])
    if args.val_subset:
        cmd.extend(["--val_subset", str(args.val_subset)])
    if args.test_subset:
        cmd.extend(["--test_subset", str(args.test_subset)])

    cmd.extend(["--experiment_prefix", experiment_prefix])
    cmd.append("--skip_if_exists")

    return cmd


def run_experiment(target_prompt: int, args, base_dir: Path):
    """Run training for a single target prompt"""
    print(f"\n{'='*100}")
    print(f"Starting experiment for model {args.model_name} | target prompt {target_prompt}")
    print(f"{'='*100}\n")

    cmd = build_train_command(args, target_prompt, base_dir)

    try:
        result = subprocess.run(
            cmd,
            cwd=base_dir,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ Completed experiment for prompt {target_prompt}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed experiment for prompt {target_prompt}")
        print(f"Error: {e}")
        return False

def collect_results(
    base_dir: Path,
    experiment_stub: str,
    sanitized_model: str,
    prompts: List[int],
    output_summary: Optional[Path]
):
    """Collect all results from checkpoints"""
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'model': experiment_stub,
        'experiments': {}
    }

    checkpoints_dir = base_dir / "checkpoints"

    for prompt_id in prompts:
        prompt_dirs = find_prompt_dirs(checkpoints_dir, prompt_id, experiment_stub, sanitized_model)

        if not prompt_dirs:
            print(f"Warning: Results directory not found for prompt {prompt_id}")
            continue

        prompt_dir = prompt_dirs[-1]
        results_file = prompt_dir / "results.json"

        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                results_summary['experiments'][f'prompt_{prompt_id}'] = {
                    'target_prompt': data['target_prompt'],
                    'source_prompts': data['source_prompts'],
                    'best_val_qwk': data['best_val_qwk'],
                    'test_qwk': data['test_metrics']['qwk'],
                    'test_pearson': data['test_metrics']['pearson'],
                    'test_spearman': data['test_metrics']['spearman'],
                    'test_rmse': data['test_metrics']['rmse'],
                    'test_mae': data['test_metrics']['mae'],
                    'total_epochs': data['total_epochs']
                }
        else:
            print(f"Warning: Results not found for prompt {prompt_id}")

    # Save summary
    summary_file = output_summary or (checkpoints_dir / f"{experiment_stub}_all_results_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults summary saved to: {summary_file}")
    return results_summary

def print_summary_table(results_summary: Dict, prompts: List[int]):
    """Print formatted summary table"""
    print("\n" + "="*100)
    print("CROSS-PROMPT LEAVE-ONE-OUT RESULTS SUMMARY")
    print("="*100)
    print(f"{'Target':<8} {'Val QWK':<10} {'Test QWK':<10} {'Pearson':<10} {'Spearman':<10} {'RMSE':<8} {'MAE':<8} {'Epochs':<8}")
    print("-"*100)

    all_test_qwk = []
    all_val_qwk = []

    for prompt_id in prompts:
        key = f'prompt_{prompt_id}'
        if key in results_summary['experiments']:
            exp = results_summary['experiments'][key]
            print(f"P{prompt_id:<7} "
                  f"{exp['best_val_qwk']:<10.4f} "
                  f"{exp['test_qwk']:<10.4f} "
                  f"{exp['test_pearson']:<10.4f} "
                  f"{exp['test_spearman']:<10.4f} "
                  f"{exp['test_rmse']:<8.4f} "
                  f"{exp['test_mae']:<8.4f} "
                  f"{exp['total_epochs']:<8}")

            all_test_qwk.append(exp['test_qwk'])
            all_val_qwk.append(exp['best_val_qwk'])

    if all_test_qwk:
        import numpy as np
        print("-"*100)
        print(f"{'Mean':<8} "
              f"{np.mean(all_val_qwk):<10.4f} "
              f"{np.mean(all_test_qwk):<10.4f}")
        print(f"{'Std':<8} "
              f"{np.std(all_val_qwk):<10.4f} "
              f"{np.std(all_test_qwk):<10.4f}")

    print("="*100 + "\n")

def main():
    args = parse_args()
    base_dir = Path(args.base_dir)

    model_tag = args.model_name.replace('/', '-').lower()

    prompts = sorted(set(args.prompts))
    prompts = [p for p in prompts if 1 <= p <= 8]
    if not prompts:
        raise ValueError("No valid prompts provided; expected values between 1 and 8")

    experiment_prefix = derive_experiment_prefix(args)
    experiment_stub = build_experiment_stub(args.model_name, experiment_prefix)
    sanitized_model = sanitize_model_name(args.model_name)

    print("="*100)
    print(f"LEAVE-ONE-OUT CROSS-PROMPT AES | MODEL: {args.model_name}")
    print("="*100)
    print(f"Base directory: {base_dir}")
    print(f"Prompts to evaluate: {prompts}")
    print(f"Experiment stub: {experiment_stub}")
    print()

    success_count = 0
    failed_prompts: List[int] = []

    for prompt_id in prompts:
        if args.skip_completed:
            prompt_dirs = find_prompt_dirs(
                base_dir / "checkpoints",
                prompt_id,
                experiment_stub,
                sanitized_model
            )
            existing = [d for d in prompt_dirs if (d / "results.json").exists()]
            if existing:
                print(f"✓ Prompt {prompt_id} already completed; skipping")
                success_count += 1
                continue

        success = run_experiment(prompt_id, args, base_dir)
        if success:
            success_count += 1
        else:
            failed_prompts.append(prompt_id)

    print("\n" + "="*100)
    print("COLLECTING RESULTS")
    print("="*100)

    summary_path = Path(args.output_summary) if args.output_summary else None
    results_summary = collect_results(
        base_dir,
        experiment_stub,
        sanitized_model,
        prompts,
        summary_path
    )
    print_summary_table(results_summary, prompts)

    print(f"\nExperiments completed: {success_count}/{len(prompts)}")
    if failed_prompts:
        print(f"Failed prompts: {failed_prompts}")
    else:
        print("All requested experiments completed successfully! ✓")

if __name__ == "__main__":
    main()
