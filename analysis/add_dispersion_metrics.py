"""
Add dispersion metrics (SD and IQR) to regression results.
"""
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import iqr

from analysis.concat_regression_results import concat_regression_results
from src.config import REGRESSION_ANALYSIS_FOLDER, REGRESSION_RESULTS_DIR


def get_latest_regression_timestamp() -> str:
    """
    Find the latest regression results timestamp directory.

    Returns:
        str: The timestamp string (e.g., '2026-01-30_0007')

    Raises:
        FileNotFoundError: If no valid regression results directories exist
    """
    base_dir = Path(REGRESSION_RESULTS_DIR)

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Regression results directory does not exist: {base_dir}"
        )

    # Find directories that contain at least one summary.csv
    candidates = [
        d for d in base_dir.iterdir()
        if d.is_dir() and list(d.glob("*/*/summary.csv"))
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No valid regression results found in {base_dir}. "
            "A valid directory must contain at least one summary.csv file."
        )

    # Timestamps are YYYY-MM-DD_HHMM format, alphabetically sortable
    latest_dir = max(candidates, key=lambda d: d.name)
    return latest_dir.name


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


def add_dispersion_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add sd_y_true and iqr_y_true columns to DataFrame."""
    df = df.copy()
    return df.join(df["y_true"].apply(compute_dispersion_metrics))


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
