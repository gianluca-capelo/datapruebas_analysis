"""Utility functions for analysis scripts."""

from pathlib import Path

from src.config import REGRESSION_RESULTS_DIR


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


# Regression runs reported in the thesis. The March run was split across two
# days: tmt_k_mean landed in the first folder, every other target in the second.
THESIS_RUN = "2026-03-07_1213"
THESIS_RUN_K_MEAN = "2026-03-06_2028"
