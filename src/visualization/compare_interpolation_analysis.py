"""
Load TMT analyses with different interpolation settings by date.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd

from src import config
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
    """
    df_interpolated = _load_valid_tmt_trials_by_date(interpolated_date)
    df_raw = _load_valid_tmt_trials_by_date(raw_date)
    return df_interpolated, df_raw
