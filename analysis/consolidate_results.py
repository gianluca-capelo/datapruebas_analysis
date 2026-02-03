"""Consolidate regression results from multiple timestamps into a single CSV."""
import argparse
import sys
from pathlib import Path

import pandas as pd

from analysis.add_dispersion_metrics import add_metrics_to_results
from src.config import REGRESSION_ANALYSIS_FOLDER


def _get_default_output_path(timestamps: list[str]) -> Path:
    suffix = "_".join(timestamps)
    return Path(REGRESSION_ANALYSIS_FOLDER) / "consolidated" / f"combined_summary_with_dispersion_{suffix}.csv"


def _validate_no_duplicates(df: pd.DataFrame) -> None:
    """Raise ValueError if any (target, dataset, model) combination appears in multiple timestamps."""
    duplicate_mask = df.duplicated(subset=["target", "dataset", "model"], keep=False)
    if not duplicate_mask.any():
        return

    duplicates = df.loc[duplicate_mask, ["target", "dataset", "model", "timestamp"]]
    duplicate_summary = (
        duplicates
        .sort_values(["target", "dataset", "model", "timestamp"])
        .groupby(["target", "dataset", "model"])["timestamp"]
        .apply(list)
        .reset_index()
    )

    error_lines = ["Duplicate (target, dataset, model) combinations found across timestamps:\n"]
    for _, row in duplicate_summary.iterrows():
        error_lines.append(
            f"  - ({row['target']}, {row['dataset']}, {row['model']}) "
            f"in timestamps: {', '.join(row['timestamp'])}"
        )

    raise ValueError("\n".join(error_lines))


def consolidate_results(timestamps: list[str], output: str | None = None) -> Path:
    """Load, concatenate, and validate regression results from multiple timestamps.

    Args:
        timestamps: List of timestamp strings (e.g., ['2026-01-29_1027', '2026-01-30_0007'])
        output: Optional output path. If None, uses default consolidated path.

    Returns:
        Path to the saved CSV file.

    Raises:
        ValueError: If duplicate (target, dataset, model) combinations are found across timestamps.
    """
    dfs = []
    for timestamp in timestamps:
        df = add_metrics_to_results(timestamp)
        df["timestamp"] = timestamp
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    _validate_no_duplicates(combined)

    combined = combined.drop(columns=["timestamp"])

    output_path = Path(output) if output else _get_default_output_path(timestamps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved consolidated results ({len(combined)} rows) to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate regression results from multiple timestamps"
    )
    parser.add_argument(
        "timestamps",
        nargs="+",
        help="Timestamp folders to consolidate (e.g., 2026-01-29_1027 2026-01-30_0007)",
    )
    parser.add_argument(
        "--output",
        help="Output path (default: analysis/results/consolidated/combined_summary_with_dispersion_<timestamps>.csv)",
    )
    args = parser.parse_args()

    try:
        consolidate_results(args.timestamps, args.output)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
