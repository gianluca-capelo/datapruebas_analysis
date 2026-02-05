"""Tests for add_dispersion_metrics using real regression results.

Reads the pre-generated consolidated CSV instead of recomputing permutation tests.
Run `python -m analysis.consolidate_results 2026-02-03_2051 2026-02-03_2053` first.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import REGRESSION_ANALYSIS_FOLDER, REGRESSION_RESULTS_DIR
from src.model.permutation_tests import compute_permutation_tests

TIMESTAMPS = ["2026-02-03_2051", "2026-02-03_2053"]
CSV_PATH = (
    Path(REGRESSION_ANALYSIS_FOLDER)
    / "consolidated"
    / f"combined_summary_with_dispersion_{'_'.join(TIMESTAMPS)}.csv"
)


@pytest.fixture(scope="module")
def consolidated_df():
    """Load the pre-generated consolidated CSV."""
    if not CSV_PATH.exists():
        pytest.fail(
            f"CSV not found at {CSV_PATH}. "
            f"Run: python -m analysis.consolidate_results {' '.join(TIMESTAMPS)}"
        )
    return pd.read_csv(CSV_PATH)


class TestPValueColumn:
    """Tests for the p_value_mae column."""

    def test_p_value_column_exists(self, consolidated_df):
        assert "p_value_mae" in consolidated_df.columns

    def test_p_value_in_valid_range(self, consolidated_df):
        assert (consolidated_df["p_value_mae"] > 0).all(), (
            f"p_value_mae has non-positive values: min={consolidated_df['p_value_mae'].min()}"
        )
        assert (consolidated_df["p_value_mae"] <= 1.0).all(), (
            f"p_value_mae exceeds 1.0: max={consolidated_df['p_value_mae'].max()}"
        )

    def test_p_value_no_nan(self, consolidated_df):
        assert consolidated_df["p_value_mae"].notna().all(), (
            f"Found {consolidated_df['p_value_mae'].isna().sum()} NaN values in p_value_mae"
        )

    def test_p_value_is_float(self, consolidated_df):
        assert consolidated_df["p_value_mae"].dtype == np.float64

    def test_dummy_regressor_not_significant(self, consolidated_df):
        dummy_rows = consolidated_df[consolidated_df["model"] == "DummyRegressor"]
        if len(dummy_rows) == 0:
            pytest.skip("No DummyRegressor rows found")
        assert (dummy_rows["p_value_mae"] > 0.3).all(), (
            f"DummyRegressor should not be significant, but got p_values: "
            f"{dummy_rows[['target', 'dataset', 'p_value_mae']].to_string()}"
        )

    def test_dispersion_columns_still_present(self, consolidated_df):
        assert "sd_y_true" in consolidated_df.columns
        assert "iqr_y_true" in consolidated_df.columns
        assert consolidated_df["sd_y_true"].notna().all()
        assert consolidated_df["iqr_y_true"].notna().all()

    def test_row_count_matches_sum_of_timestamps(self, consolidated_df):
        from analysis.concat_regression_results import concat_regression_results

        expected_rows = sum(
            len(concat_regression_results(ts)) for ts in TIMESTAMPS
        )
        assert len(consolidated_df) == expected_rows, (
            f"Row count mismatch: {len(consolidated_df)} vs {expected_rows}"
        )


class TestCrossValidation:
    """Cross-validate p_values against compute_permutation_tests()."""

    def test_cross_validation_with_compute_permutation_tests(self, consolidated_df):
        timestamp = "2026-02-03_2053"
        target = "ssrt"

        target_df = consolidated_df[consolidated_df["target"] == target]
        assert len(target_df) > 0, f"No rows for target={target}"

        results_dir = Path(REGRESSION_RESULTS_DIR) / timestamp / target
        reference_df = compute_permutation_tests(
            results_dir, task="regression", metric="mae"
        )

        merged = target_df.merge(
            reference_df.rename(columns={"Dataset": "dataset", "Model": "model"}),
            on=["dataset", "model"],
            suffixes=("_new", "_ref"),
        )

        assert len(merged) > 0, f"No matching rows for target={target}"
        np.testing.assert_allclose(
            merged["p_value_mae"].values,
            merged["p_value"].values,
            rtol=1e-10,
            err_msg="p_values from add_dispersion_metrics vs compute_permutation_tests differ",
        )
