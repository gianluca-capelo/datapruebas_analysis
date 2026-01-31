"""
Concatenate regression results for a specific timestamp.
"""
import argparse
import os
from pathlib import Path

import pandas as pd

from src.config import REGRESSION_ANALYSIS_FOLDER, REGRESSION_RESULTS_DIR


def concat_regression_results(timestamp: str) -> pd.DataFrame:
    """Concatenate all summary.csv files for a given timestamp."""
    base_path = Path(REGRESSION_RESULTS_DIR) / timestamp

    if not base_path.exists():
        raise ValueError(f"Timestamp directory not found: {base_path}")

    dfs = []
    for target_dir in base_path.iterdir():
        if not target_dir.is_dir():
            continue
        for dataset_dir in target_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            summary_path = dataset_dir / "summary.csv"
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                df["target"] = target_dir.name
                df["dataset"] = dataset_dir.name
                dfs.append(df)

    if not dfs:
        raise ValueError(f"No summary.csv files found in {base_path}")

    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Concatenate regression results")
    parser.add_argument(
        "--timestamp",
        required=True,
        help="Timestamp folder (e.g., 2026-01-26_1901)",
    )
    parser.add_argument(
        "--output",
        help="Output path (default: data/regression_analysis/{timestamp}/combined_summary.csv)",
    )
    args = parser.parse_args()

    combined = concat_regression_results(args.timestamp)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(REGRESSION_ANALYSIS_FOLDER) / args.timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "combined_summary.csv"
    combined.to_csv(output_path, index=False)
    print(f"Saved combined results to {output_path}")


if __name__ == "__main__":
    main()
