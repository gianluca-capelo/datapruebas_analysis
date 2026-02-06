"""
Add dispersion metrics (SD, IQR) and permutation test p-values to regression results.
"""
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import iqr

from analysis.scripts.concat_regression_results import concat_regression_results
from analysis.scripts.utils import get_latest_regression_timestamp
from src.config import REGRESSION_ANALYSIS_FOLDER
from src.model.permutation_tests import permutation_test


def parse_array_string(array_str: str) -> np.ndarray:
    """Parse string representation of list to numpy array."""
    return np.array(ast.literal_eval(array_str))


def compute_dispersion_metrics(y_true_str: str) -> pd.Series:
    """Compute SD and IQR from y_true string."""
    arr = parse_array_string(y_true_str)
    return pd.Series({
        "sd_y_true": np.std(arr, ddof=1),
        "iqr_y_true": iqr(arr),
    })


def compute_p_value_mae(y_true_str: str, y_pred_str: str) -> float:
    """Compute permutation test p-value using MAE metric."""
    y_true = parse_array_string(y_true_str)
    y_pred = parse_array_string(y_pred_str)
    _, p_value = permutation_test(
        y_true, y_pred, n_permutations=1000, seed=42, metric="mae"
    )
    return p_value


def add_dispersion_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add sd_y_true, iqr_y_true, and p_value_mae columns to DataFrame."""
    df = df.copy()
    dispersion = df["y_true"].apply(compute_dispersion_metrics)
    df = df.join(dispersion)
    df["p_value_mae"] = df.apply(
        lambda row: compute_p_value_mae(row["y_true"], row["y_pred"]),
        axis=1,
    )
    return df


def add_metrics_to_results(timestamp: str) -> pd.DataFrame:
    """Load combined results and add dispersion metrics."""
    df = concat_regression_results(timestamp)
    return add_dispersion_columns(df)


def run_dispersion_analysis(timestamp: str | None = None, output: str | None = None) -> Path:
    """
    Run dispersion analysis on regression results.

    Args:
        timestamp: Timestamp folder (e.g., '2026-01-26_1901'). If None, uses latest.
        output: Output path. If None, saves to default location.

    Returns:
        Path to the saved CSV file.
    """
    # Resolve timestamp: use latest if not provided
    if timestamp is None:
        timestamp = get_latest_regression_timestamp()
        print(f"Auto-detected latest timestamp: {timestamp}")

    df = add_metrics_to_results(timestamp)

    if output:
        output_path = Path(output)
    else:
        output_dir = Path(REGRESSION_ANALYSIS_FOLDER) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "combined_summary_with_dispersion.csv"

    df.to_csv(output_path, index=False)
    print(f"Saved results with dispersion metrics to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Add dispersion metrics (SD and IQR) to regression results"
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Timestamp folder (e.g., 2026-01-26_1901). If not provided, uses latest.",
    )
    parser.add_argument(
        "--output",
        help="Output path (default: analysis/results/{timestamp}/combined_summary_with_dispersion.csv)",
    )
    args = parser.parse_args()

    run_dispersion_analysis(timestamp=args.timestamp, output=args.output)


if __name__ == "__main__":
    main()
