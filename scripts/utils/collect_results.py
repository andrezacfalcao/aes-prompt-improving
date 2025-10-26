"""Utility to consolidate leave-one-out results across models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate cross-prompt results")

    parser.add_argument(
        "--results_root",
        type=str,
        default="checkpoints",
        help="Directory containing per-experiment folders"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="analysis/aggregate_results.json",
        help="Path to save consolidated JSON"
    )

    parser.add_argument(
        "--include_models",
        type=str,
        nargs="*",
        default=None,
        help="Subset of model identifiers to aggregate (default: all)"
    )

    return parser.parse_args()


def collect_results(results_root: Path, include_models: List[str] | None) -> Dict:
    summary: Dict[str, Dict] = {}

    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        if include_models and model_name not in include_models:
            continue

        results_files = list(model_dir.glob("results.json"))
        if not results_files:
            continue

        for result_file in results_files:
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            key = f"{model_name}"
            summary.setdefault(key, {})[f"prompt_{data['target_prompt']}"] = data

    return summary


def write_summary_table(summary: Dict[str, Dict], output_dir: Path) -> None:
    records = []
    for model_name, prompts in summary.items():
        for prompt_key, data in prompts.items():
            metrics = data.get("test_metrics", {})
            records.append({
                "model": model_name,
                "target_prompt": data.get("target_prompt"),
                "best_val_qwk": data.get("best_val_qwk"),
                "test_qwk": metrics.get("qwk"),
                "test_pearson": metrics.get("pearson"),
                "test_spearman": metrics.get("spearman"),
                "test_rmse": metrics.get("rmse"),
                "test_mae": metrics.get("mae"),
                "total_epochs": data.get("total_epochs")
            })

    if records:
        df = pd.DataFrame(records)
        csv_path = output_dir / "aggregate_results.csv"
        pivot_path = output_dir / "aggregate_qwk_pivot.csv"

        df.to_csv(csv_path, index=False)

        pivot = df.pivot_table(
            values="test_qwk",
            index="model",
            columns="target_prompt"
        )
        pivot.to_csv(pivot_path)

        print(f"Saved aggregate CSV to {csv_path}")
        print(f"Saved QWK pivot table to {pivot_path}")
    else:
        print("No records to summarize into CSV.")


def main() -> None:
    args = parse_args()

    results_root = Path(args.results_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = collect_results(results_root, args.include_models)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved aggregated results to {output_path}")

    write_summary_table(summary, output_path.parent)


if __name__ == "__main__":
    main()


