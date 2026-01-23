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


def load_interpolated_tmt_trials_by_date(date: str) -> pd.DataFrame:
    """Load valid TMT trials from interpolated analysis by date.

    Raises:
        ValueError: If interpolate_trajectory is not True
    """
    run_dir = _get_run_directory_by_date(date)
    run_config = get_run_configuration(run_dir)
    is_interpolated = run_config.get("interpolate_trajectory", False)

    if not is_interpolated:
        raise ValueError(
            f"Analysis {date} has interpolate_trajectory=False, expected True"
        )

    df = pd.read_csv(run_dir / "analysis.csv", on_bad_lines='warn')
    builder = DatasetBuilder()
    return builder._get_valid_tmt_trials(df)


def load_raw_tmt_trials_by_date(date: str) -> pd.DataFrame:
    """Load valid TMT trials from raw (non-interpolated) analysis by date.

    Raises:
        ValueError: If interpolate_trajectory is not False
    """
    run_dir = _get_run_directory_by_date(date)
    run_config = get_run_configuration(run_dir)
    is_interpolated = run_config.get("interpolate_trajectory", False)

    if is_interpolated:
        raise ValueError(
            f"Analysis {date} has interpolate_trajectory=True, expected False"
        )

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
    df_interpolated = load_interpolated_tmt_trials_by_date(interpolated_date)
    df_raw = load_raw_tmt_trials_by_date(raw_date)

    # Keep only trials present in both analyses
    common_trials = set(zip(df_interpolated['subject_id'], df_interpolated['trial_id'])) & \
                   set(zip(df_raw['subject_id'], df_raw['trial_id']))

    df_interpolated = df_interpolated[
        df_interpolated.apply(lambda r: (r['subject_id'], r['trial_id']) in common_trials, axis=1)
    ].reset_index(drop=True)

    df_raw = df_raw[
        df_raw.apply(lambda r: (r['subject_id'], r['trial_id']) in common_trials, axis=1)
    ].reset_index(drop=True)

    return df_interpolated, df_raw


# Columns to exclude from metric comparison (metadata/identifiers)
EXCLUDE_COLS = {
    'subject_id', 'trial_id', 'trial_type', 'is_valid',
    'trial_order_of_appearance', 'invalid_cause', 'error_message',
    'recorded_at', 'start_date', 'mail',
    'dispositivo', 'mano', 'dispositivo-config', 'alcohol-drogas',
    'tratamiento', 'usoDelPad', 'comentarioFinal',
    'MouseOrPad-choice', 'hand-choice', 'hand_config-choice',
    'PadUsechoice', 'age', 'gender', 'education_level', 'nationality',
    'experiment_origin', 'device', 'hand', 'device_config',
    'alcohol_drugs', 'treatment', 'pad_usage', 'final_comment',
    'speed_threshold', 'px2mm', 'scale_factor',
}


def compare_metrics(
    df_interpolated: pd.DataFrame,
    df_raw: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare metrics between interpolated and raw analyses.

    Merges on (subject_id, trial_id) and computes differences for each metric.

    Args:
        df_interpolated: DataFrame with interpolated trials
        df_raw: DataFrame with raw trials

    Returns:
        DataFrame with comparison statistics per metric:
        - mean_interp, mean_raw: Mean values
        - mean_diff: Mean difference (interp - raw)
        - mean_abs_diff: Mean absolute difference
        - mean_pct_diff: Mean percentage difference
        - correlation: Pearson correlation between interp and raw
    """
    merge_keys = ['subject_id', 'trial_id']
    merged = pd.merge(
        df_interpolated,
        df_raw,
        on=merge_keys,
        suffixes=('_interp', '_raw'),
        how='inner'
    )

    # Find numeric metric columns
    metric_cols = []
    for col in df_interpolated.columns:
        if col in EXCLUDE_COLS or col in merge_keys:
            continue
        if pd.api.types.is_numeric_dtype(df_interpolated[col]):
            metric_cols.append(col)

    results = []
    for col in metric_cols:
        col_interp = f"{col}_interp"
        col_raw = f"{col}_raw"

        if col_interp not in merged.columns or col_raw not in merged.columns:
            continue

        interp_vals = merged[col_interp].dropna()
        raw_vals = merged[col_raw].dropna()

        # Get aligned values (both non-null)
        mask = merged[col_interp].notna() & merged[col_raw].notna()
        interp_aligned = merged.loc[mask, col_interp]
        raw_aligned = merged.loc[mask, col_raw]

        if len(interp_aligned) == 0:
            continue

        diff = interp_aligned - raw_aligned
        abs_diff = diff.abs()

        # Percentage difference (relative to raw, avoiding division by zero)
        with pd.option_context('mode.use_inf_as_na', True):
            pct_diff = ((diff / raw_aligned.replace(0, float('nan'))) * 100).dropna()

        # Correlation
        if len(interp_aligned) > 1:
            corr = interp_aligned.corr(raw_aligned)
        else:
            corr = float('nan')

        results.append({
            'metric': col,
            'n_trials': len(interp_aligned),
            'mean_interp': interp_aligned.mean(),
            'mean_raw': raw_aligned.mean(),
            'mean_diff': diff.mean(),
            'std_diff': diff.std(),
            'mean_abs_diff': abs_diff.mean(),
            'mean_pct_diff': pct_diff.mean() if len(pct_diff) > 0 else float('nan'),
            'correlation': corr,
        })

    return pd.DataFrame(results).sort_values('mean_abs_diff', ascending=False)


# Analysis dates
INTERPOLATED_DATE = "2026-01-16_19-20-02"  # interpolate_trajectory=True
RAW_DATE = "2026-01-16_18-57-15"           # interpolate_trajectory=False


if __name__ == "__main__":
    df_interp, df_raw = load_interpolated_and_raw_analyses(INTERPOLATED_DATE, RAW_DATE)

    print(f"Interpolated ({INTERPOLATED_DATE}):")
    print(f"  - Trials: {len(df_interp):,}")
    print(f"  - Subjects: {df_interp['subject_id'].nunique()}")

    print(f"\nRaw ({RAW_DATE}):")
    print(f"  - Trials: {len(df_raw):,}")
    print(f"  - Subjects: {df_raw['subject_id'].nunique()}")

    print("\n" + "=" * 80)
    print("METRIC COMPARISON (sorted by mean absolute difference)")
    print("=" * 80)

    comparison = compare_metrics(df_interp, df_raw)

    output_path = Path(config.DATA_DIR) / "interpolation_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    print(comparison.to_string(index=False))
