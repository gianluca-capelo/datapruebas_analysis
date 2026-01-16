"""
Load TMT analyses with different interpolation settings by date.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd

from src import config
from src.loader.load_last_split import get_run_configuration
from src.model.datasetbuilder.dataset_builder import DatasetBuilder


def _get_run_directory_by_date(date: str) -> Path:
    """Find run directory matching a date prefix (YYYY-MM-DD)."""
    base_dir = Path(config.HAND_ANALYSIS_FOLDER)
    candidates = [
        d for d in base_dir.iterdir()
        if d.is_dir()
        and d.name.startswith(date)
        and (d / "configuration.json").exists()
        and (d / "analysis.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No run directories found for date {date!r}")
    return max(candidates, key=lambda d: d.name)


def _load_valid_tmt_trials_by_date(date: str) -> pd.DataFrame:
    """Load valid TMT trials from analysis by date."""
    run_dir = _get_run_directory_by_date(date)
    df = pd.read_csv(run_dir / "analysis.csv", on_bad_lines='warn')
    builder = DatasetBuilder()
    return builder._get_valid_tmt_trials(df)


def load_interpolated_and_raw_analyses(
    interpolated_date: str,
    raw_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load two TMT analyses: one interpolated, one raw.

    Args:
        interpolated_date: Date of interpolated analysis (YYYY-MM-DD)
        raw_date: Date of raw analysis (YYYY-MM-DD)

    Returns:
        (df_interpolated, df_raw) - DataFrames with valid TMT trials

    Raises:
        ValueError: If interpolation settings don't match expected values
    """
    run_dir_interp = _get_run_directory_by_date(interpolated_date)
    run_dir_raw = _get_run_directory_by_date(raw_date)

    config_interp = get_run_configuration(run_dir_interp)
    config_raw = get_run_configuration(run_dir_raw)

    is_interp = config_interp.get("interpolate_trajectory", False)
    is_raw = config_raw.get("interpolate_trajectory", True)

    if not is_interp:
        raise ValueError(
            f"Analysis {interpolated_date} has interpolate_trajectory=False, expected True"
        )
    if is_raw:
        raise ValueError(
            f"Analysis {raw_date} has interpolate_trajectory=True, expected False"
        )

    df_interpolated = _load_valid_tmt_trials_by_date(interpolated_date)
    df_raw = _load_valid_tmt_trials_by_date(raw_date)
    return df_interpolated, df_raw


# Analysis dates
INTERPOLATED_DATE = "2026-01-16_09-59-35"  # interpolate_trajectory=True
RAW_DATE = "2026-01-16_18-29-32"           # interpolate_trajectory=False


if __name__ == "__main__":
    df_interp, df_raw = load_interpolated_and_raw_analyses(INTERPOLATED_DATE, RAW_DATE)

    print(f"Interpolated ({INTERPOLATED_DATE}):")
    print(f"  - Trials: {len(df_interp):,}")
    print(f"  - Subjects: {df_interp['subject_id'].nunique()}")

    print(f"\nRaw ({RAW_DATE}):")
    print(f"  - Trials: {len(df_raw):,}")
    print(f"  - Subjects: {df_raw['subject_id'].nunique()}")
