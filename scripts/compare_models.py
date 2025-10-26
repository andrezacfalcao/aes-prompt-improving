#!/usr/bin/env python3
"""
Model Comparison Script

Compares different models across all prompts:
- Baseline (BERT/RoBERTa/DistilBERT/ELECTRA)
- PANN
- PANN+DRL (with CNAA and/or CST)

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --prompts 1 2 3
    python scripts/compare_models.py --models baseline pann pann_drl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare model results")

    parser.add_argument(
        "--prompts",
        type=int,
        nargs="+",
        default=list(range(1, 9)),
        help="Prompts to analyze (default: all 1-8)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["baseline", "pann", "pann_drl"],
        choices=["baseline", "pann", "pann_drl"],
        help="Models to compare"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis",
        help="Output directory for comparison results"
    )

    return parser.parse_args()


def find_checkpoint_dirs(checkpoint_dir: Path, model_type: str, prompt: int) -> List[Path]:
    """Find checkpoint directories for a given model type and prompt."""
    patterns = {
        "baseline": [
            f"bert-base-uncased_bert_base_uncased_p{prompt}",
            f"bert_p{prompt}",
            f"roberta_p{prompt}",
            f"distilbert_p{prompt}",
            f"electra_p{prompt}",
        ],
        "pann": [
            f"pann_p{prompt}",
        ],
        "pann_drl": [
            f"pann_drl_p{prompt}_full",
            f"pann_drl_p{prompt}_cnaa",
            f"pann_drl_p{prompt}_cst",
            f"pann_drl_p{prompt}",
        ]
    }

    matching_dirs = []

    for pattern in patterns.get(model_type, []):
        matching = list(checkpoint_dir.glob(pattern))
        matching_dirs.extend(matching)

    return matching_dirs


def load_results(results_path: Path) -> Dict:
    """Load results from JSON file."""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode {results_path}")
        return None


def extract_metrics(results: Dict, model_name: str, prompt: int) -> Dict:
    """Extract metrics from results dict."""
    if results is None:
        return None

    test_metrics = results.get('test_metrics', {})

    return {
        'model': model_name,
        'prompt': prompt,
        'qwk': test_metrics.get('qwk', None),
        'pearson': test_metrics.get('pearson', None),
        'spearman': test_metrics.get('spearman', None),
        'rmse': test_metrics.get('rmse', None),
        'mae': test_metrics.get('mae', None),
        'epochs_trained': results.get('epochs_trained', results.get('drl_config', {}).get('cnaa_epochs', None)),
        'checkpoint_path': results.get('checkpoint_path', 'N/A')
    }


def create_comparison_table(all_results: List[Dict]) -> pd.DataFrame:
    """Create comparison table from results."""
    df = pd.DataFrame(all_results)

    if df.empty:
        print("⚠️  No results found!")
        return df

    df = df.sort_values(['model', 'prompt'])

    return df


def compute_average_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average metrics across all prompts for each model."""
    if df.empty:
        return df

    avg_df = df.groupby('model').agg({
        'qwk': 'mean',
        'pearson': 'mean',
        'spearman': 'mean',
        'rmse': 'mean',
        'mae': 'mean'
    }).reset_index()

    avg_df = avg_df.round(4)

    return avg_df


def print_comparison_table(df: pd.DataFrame, title: str = "Model Comparison"):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print(f"{title:^100}")
    print("="*100)

    if df.empty:
        print("No data available")
        return

    display_cols = ['model', 'prompt', 'qwk', 'pearson', 'spearman', 'rmse', 'mae']
    display_df = df[display_cols]

    print(display_df.to_string(index=False))
    print("="*100 + "\n")


def create_pivot_table(df: pd.DataFrame, metric: str = 'qwk') -> pd.DataFrame:
    """Create pivot table with models as rows and prompts as columns."""
    if df.empty:
        return df

    pivot = df.pivot(index='model', columns='prompt', values=metric)
    pivot = pivot.round(4)

    return pivot


def main():
    """Main comparison function."""
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*100)
    print("Model Comparison Tool")
    print("="*100)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Prompts: {args.prompts}")
    print(f"Models: {args.models}")
    print("="*100 + "\n")

    all_results = []

    for model_type in args.models:
        print(f"\nSearching for {model_type} results...")

        for prompt in args.prompts:
            checkpoint_dirs = find_checkpoint_dirs(checkpoint_dir, model_type, prompt)

            if not checkpoint_dirs:
                print(f"  ✗ No checkpoint found for prompt {prompt}")
                continue

            for ckpt_dir in checkpoint_dirs:
                results_path = ckpt_dir / 'results.json'

                if not results_path.exists():
                    print(f"  ✗ No results.json in {ckpt_dir.name}")
                    continue

                results = load_results(results_path)

                if results is None:
                    continue

                model_name = ckpt_dir.name

                if model_type == "baseline":
                    if "bert-base-uncased" in model_name or "bert_base" in model_name:
                        model_name = "BERT"
                    elif "roberta" in model_name:
                        model_name = "RoBERTa"
                    elif "distilbert" in model_name:
                        model_name = "DistilBERT"
                    elif "electra" in model_name:
                        model_name = "ELECTRA"
                elif model_type == "pann":
                    model_name = "PANN"
                elif model_type == "pann_drl":
                    if "full" in ckpt_dir.name:
                        model_name = "PANN+DRL (Full)"
                    elif "cnaa" in ckpt_dir.name:
                        model_name = "PANN+DRL (CNAA)"
                    elif "cst" in ckpt_dir.name:
                        model_name = "PANN+DRL (CST)"
                    else:
                        model_name = "PANN+DRL"

                metrics = extract_metrics(results, model_name, prompt)

                if metrics:
                    all_results.append(metrics)
                    print(f"  ✓ Found results for {model_name} - Prompt {prompt} (QWK: {metrics['qwk']:.4f})")

    df = create_comparison_table(all_results)

    print_comparison_table(df, "Detailed Results")

    avg_df = compute_average_metrics(df)
    print_comparison_table(avg_df, "Average Metrics Across All Prompts")

    qwk_pivot = create_pivot_table(df, 'qwk')

    if not qwk_pivot.empty:
        print("\n" + "="*100)
        print("QWK by Model and Prompt")
        print("="*100)
        print(qwk_pivot.to_string())
        print("="*100 + "\n")

        qwk_pivot.to_csv(output_dir / 'qwk_comparison.csv')
        print(f"✓ QWK pivot table saved to {output_dir / 'qwk_comparison.csv'}")

    if not df.empty:
        df.to_csv(output_dir / 'full_comparison.csv', index=False)
        print(f"✓ Full comparison saved to {output_dir / 'full_comparison.csv'}")

        avg_df.to_csv(output_dir / 'average_metrics.csv', index=False)
        print(f"✓ Average metrics saved to {output_dir / 'average_metrics.csv'}")

    if not df.empty and len(df['model'].unique()) > 1:
        print("\n" + "="*100)
        print("Statistical Summary")
        print("="*100)

        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            print(f"\n{model}:")
            print(f"  Mean QWK: {model_df['qwk'].mean():.4f} ± {model_df['qwk'].std():.4f}")
            print(f"  Min QWK: {model_df['qwk'].min():.4f}")
            print(f"  Max QWK: {model_df['qwk'].max():.4f}")
            print(f"  N prompts: {len(model_df)}")

        print("="*100 + "\n")

    print("\n✓ Comparison complete!\n")


if __name__ == "__main__":
    main()
