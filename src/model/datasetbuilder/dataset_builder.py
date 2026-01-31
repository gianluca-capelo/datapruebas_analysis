"""
DatasetBuilder: Constructs ML datasets from multiple cognitive task analyses.
"""

import logging
from typing import Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from src.loader.load_last_split import load_last_analysis
from src.loader.sst_analysis_loader import get_latest_sst_analysis
from src.loader.cdt_analysis_loader import get_latest_cdt_analysis
from src.loader.gonogo_analysis_loader import get_latest_gonogo_analysis


class DatasetBuilder:
    """
    Builds X, y datasets from cognitive task data.

    Supported datasets:
        - 'tmt_ssrt':   TMT features → SSRT target (Stop Signal Task)
        - 'tmt_k6':     TMT features → K_6 capacity target (Change Detection Task)
        - 'tmt_k4':     TMT features → K_4 capacity target (Change Detection Task, set size 4)
        - 'tmt_k_mean': TMT features → K_mean target (average of K_4 and K_6)
        - 'tmt_k6_v2':  TMT features → K_6 target with QC filter (0 <= K_6 <= 4.5)
        - 'tmt_dprime': TMT features → d' sensitivity target (Go/No-Go Task)
        - 'tmt_age':    TMT features → age target (subject age prediction)
    """
    
    # Columns to exclude from features (metadata/identifiers)
    EXCLUDE_COLS = {
        'subject_id', 'trial_id', 'trial_type', 'is_valid',
        'trial_order_of_appearance', 'invalid_cause', 'error_message',
        'recorded_at', 'start_date', 'mail',
        # Config/demographic columns (not trial features)
        'dispositivo', 'mano', 'dispositivo-config', 'alcohol-drogas',
        'tratamiento', 'usoDelPad', 'comentarioFinal',
        'MouseOrPad-choice', 'hand-choice', 'hand_config-choice',
        'PadUsechoice', 'age', 'gender', 'education_level', 'nationality',
        'experiment_origin', 'device', 'hand', 'device_config',
        'alcohol_drugs', 'treatment', 'pad_usage', 'final_comment',
        'speed_threshold', 'px2mm',
    }
        
    def get_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray, list, str]:
        """
        Get X, y, feature_names, and target_name for a dataset.
        
        Args:
            name: Dataset name (e.g., 'tmt_ssrt', 'tmt_k', 'tmt_dprime')
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature column names
            target_name: Name of the target column
        """
        if name == 'tmt_ssrt':
            return self._build_tmt_ssrt()
        elif name == 'tmt_k6':
            return self._build_tmt_k6()
        elif name == 'tmt_k6_v2':
            return self._build_tmt_k6_v2()
        elif name == 'tmt_k4':
            return self._build_tmt_k4()
        elif name == 'tmt_k_mean':
            return self._build_tmt_k_mean()
        elif name == 'tmt_dprime':
            return self._build_tmt_dprime()
        elif name == 'tmt_age':
            return self._build_tmt_age()
        else:
            available = ['tmt_ssrt', 'tmt_k6', 'tmt_k4', 'tmt_k_mean', 'tmt_k6_v2', 'tmt_dprime', 'tmt_age']
            raise ValueError(
                f"Unknown dataset: {name}. Available: {available}"
            )
    
    def _load_valid_tmt_trials(self) -> pd.DataFrame:
        """
        Load TMT data and filter to valid trials.

        Returns:
            DataFrame with valid trials from subjects with both trial types.
        """
        tmt_df, _ = load_last_analysis()
        return self._get_valid_tmt_trials(tmt_df)

    def _load_tmt_aggregated(self) -> pd.DataFrame:
        """
        Load and aggregate TMT data to subject level.

        Returns:
            DataFrame with one row per subject and aggregated TMT features.
        """
        tmt_valid = self._load_valid_tmt_trials()
        return self._aggregate_tmt(tmt_valid)

    def _build_generic_dataset(self, loader_func, target_col: str, loader_name: str,
                                min_val: float = None, max_val: float = None) -> Tuple[np.ndarray, np.ndarray, list, str]:
        """
        Generic helper to build TMT features vs Any Target.

        Args:
            loader_func: Function to load target task data
            target_col: Name of target column
            loader_name: Name of target task (for logging)
            min_val: Optional minimum value filter for target
            max_val: Optional maximum value filter for target
        """
        tmt_agg = self._load_tmt_aggregated()

        # Load Target Task data
        task_result = loader_func()
        if task_result is None:
            raise RuntimeError(f"No {loader_name} analysis found. Run {loader_name} analysis first.")

        # Asumimos que el loader devuelve (df, metadata) o similar, tomamos el df [0]
        # y filtramos subject_id y el target
        task_df = task_result[0]

        if target_col not in task_df.columns:
             raise ValueError(f"Target column '{target_col}' not found in {loader_name} data. Available: {list(task_df.columns)}")

        task_subset = task_df[['subject_id', target_col]]

        # Merge on subject_id
        merged = pd.merge(tmt_agg, task_subset, on='subject_id', how='inner')

        if len(merged) == 0:
            raise RuntimeError(f"No matching subjects between TMT and {loader_name} data.")

        # Apply target value filter if specified
        if min_val is not None or max_val is not None:
            n_before = len(merged)
            if min_val is not None:
                merged = merged[merged[target_col] >= min_val]
            if max_val is not None:
                merged = merged[merged[target_col] <= max_val]
            n_filtered = n_before - len(merged)
            if n_filtered > 0:
                logger.info(f"Filtered {n_filtered} subjects with {target_col} outside [{min_val}, {max_val}]")

        # Extract X and y
        feature_cols = [c for c in merged.columns if c not in ['subject_id', target_col]]
        X = merged[feature_cols].values
        y = merged[target_col].values

        return X, y, feature_cols, target_col
    
    
    def _build_tmt_ssrt(self) -> Tuple[np.ndarray, np.ndarray, list, str]:
        """
        Build dataset with TMT features and SSRT (Stop Signal) as target.
        """
        target_name = 'ssrt'
        return self._build_generic_dataset(
            loader_func=get_latest_sst_analysis,
            target_col=target_name,
            loader_name="SST"
        )

    def _build_tmt_k6(self) -> Tuple[np.ndarray, np.ndarray, list, str]:
        """
        Build dataset with TMT features and K_6 (CDT Capacity) as target.
        """
        target_name = 'K_6'
        return self._build_generic_dataset(
            loader_func=get_latest_cdt_analysis,
            target_col=target_name,
            loader_name="CDT"
        )

    def _build_tmt_k4(self) -> Tuple[np.ndarray, np.ndarray, list, str]:
        """
        Build dataset with TMT features and K_4 (CDT Capacity for set size 4) as target.
        """
        target_name = 'K_4'
        return self._build_generic_dataset(
            loader_func=get_latest_cdt_analysis,
            target_col=target_name,
            loader_name="CDT"
        )

    def _build_tmt_k_mean(self) -> Tuple[np.ndarray, np.ndarray, list, str]:
        """
        Build dataset with TMT features and K_mean (average of K_4 and K_6) as target.
        """
        target_name = 'K_mean'
        return self._build_generic_dataset(
            loader_func=get_latest_cdt_analysis,
            target_col=target_name,
            loader_name="CDT"
        )

    def _build_tmt_k6_v2(self) -> Tuple[np.ndarray, np.ndarray, list, str]:
        """
        Build dataset with TMT features and K_6 as target, filtering extreme values.

        Filters: 0 <= K_6 <= 4.5 (based on literature typical range for Cowan's K)
        """
        target_name = 'K_6'
        return self._build_generic_dataset(
            loader_func=get_latest_cdt_analysis,
            target_col=target_name,
            loader_name="CDT",
            min_val=0,
            max_val=4.5
        )

    def _build_tmt_dprime(self) -> Tuple[np.ndarray, np.ndarray, list, str]:
        """
        Build dataset with TMT features and d' (Go/No-Go Sensitivity) as target.
        """
        target_name = 'sensibilidad'
        return self._build_generic_dataset(
            loader_func=get_latest_gonogo_analysis,
            target_col=target_name,
            loader_name="Go/No-Go"
        )

    def _build_tmt_age(self) -> Tuple[np.ndarray, np.ndarray, list, str]:
        """
        Build dataset with TMT features and age as target.

        Filters subjects with missing or invalid age (outside 18-100 range).
        """
        # Load valid TMT trials (before aggregation)
        tmt_valid = self._load_valid_tmt_trials()

        # Extract age per subject (before aggregation loses it)
        age_per_subject = tmt_valid.groupby('subject_id')['age'].first()

        # Aggregate TMT features
        tmt_agg = self._aggregate_tmt(tmt_valid)

        # Merge age back
        merged = tmt_agg.merge(
            age_per_subject.reset_index(),
            on='subject_id',
            how='inner'
        )

        # Filter subjects with missing age
        missing_age = merged['age'].isna().sum()
        if missing_age > 0:
            logger.warning(f"Filtering {missing_age} subjects with missing age data")
            merged = merged[merged['age'].notna()]

        # Filter subjects with invalid age (outside 18-100 range)
        invalid_age_mask = (merged['age'] < 18) | (merged['age'] > 100)
        if invalid_age_mask.any():
            n_invalid = invalid_age_mask.sum()
            logger.warning(f"Filtering {n_invalid} subjects with invalid age (outside [18, 100])")
            merged = merged[~invalid_age_mask]

        target_col = 'age'
        feature_cols = [c for c in merged.columns if c not in ['subject_id', target_col]]
        X = merged[feature_cols].values
        y = merged[target_col].values

        return X, y, feature_cols, target_col

    def _get_valid_tmt_trials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter TMT data to valid trials with proper coverage.

        Filters to valid trials only and excludes subjects without
        at least one valid trial of each type (PART_A and PART_B).

        Args:
            df: TMT DataFrame with trial-level data

        Returns:
            DataFrame with valid trials from subjects with both trial types
        """
        # Filter valid trials only (handle both bool and string 'True')
        df_valid = df[df['is_valid'].astype(str) == 'True'].copy()

        # Filter subjects by trial type coverage (must have >=1 PART_A and >=1 PART_B)
        df_valid = self._filter_subjects_by_trial_coverage(df_valid)

        return df_valid

    def _aggregate_tmt(self, df_valid: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate TMT trial-level data to subject-level.

        Pivots by trial_type (PART_A, PART_B) and computes mean per subject.

        Args:
            df_valid: TMT DataFrame with valid trials (already filtered)

        Returns:
            DataFrame with one row per subject and columns like 'rt_PART_A', 'rt_PART_B'
        """
        # Auto-detect numeric feature columns (exclude metadata)
        feature_cols = []
        for col in df_valid.columns:
            if col in self.EXCLUDE_COLS:
                continue
            # Try to convert to numeric
            numeric_col = pd.to_numeric(df_valid[col], errors='coerce')
            # Keep if at least 50% of values are numeric
            if numeric_col.notna().mean() > 0.5:
                df_valid[col] = numeric_col
                feature_cols.append(col)

        # Pivot by trial_type and aggregate with mean
        agg = df_valid.pivot_table(
            index='subject_id',
            columns='trial_type',
            values=feature_cols,
            aggfunc='mean'
        )

        # Flatten column names: (rt, PART_A) → rt_PART_A
        agg.columns = [f"{col}_{trial_type}" for col, trial_type in agg.columns]

        return agg.reset_index()

    def _filter_subjects_by_trial_coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out subjects that don't have at least one trial of each type (PART_A and PART_B).

        Args:
            df: DataFrame with valid trials only

        Returns:
            DataFrame with only subjects that have both trial types
        """
        required_types = ['PART_A', 'PART_B']

        # Count trials per subject per type
        trial_type_counts = df.groupby('subject_id')['trial_type'].value_counts().unstack(fill_value=0)

        # Ensure both columns exist (edge case: dataset might have only one type)
        for trial_type in required_types:
            if trial_type not in trial_type_counts.columns:
                trial_type_counts[trial_type] = 0

        # Identify valid and excluded subjects
        valid_mask = (trial_type_counts['PART_A'] >= 1) & (trial_type_counts['PART_B'] >= 1)
        valid_subjects = trial_type_counts[valid_mask].index.tolist()
        excluded_subjects = trial_type_counts[~valid_mask].index.tolist()

        # Log exclusions
        if excluded_subjects:
            logger.warning(
                f"Excluding {len(excluded_subjects)} subjects without both PART_A and PART_B trials: "
                f"{excluded_subjects[:5]}{'...' if len(excluded_subjects) > 5 else ''}"
            )
            for subj in excluded_subjects[:5]:
                counts = trial_type_counts.loc[subj]
                logger.debug(f"  Subject {subj}: PART_A={counts['PART_A']}, PART_B={counts['PART_B']}")

        # Validate at least some subjects remain
        assert len(valid_subjects) > 0, "No subjects remain after trial type coverage filter"

        return df[df['subject_id'].isin(valid_subjects)]

    def get_exclusion_report(self, df: pd.DataFrame) -> dict:
        """
        Generate a report of subjects that would be excluded due to missing trial types.

        Args:
            df: TMT DataFrame with trial-level data (before aggregation)

        Returns:
            dict with keys:
                - 'total_subjects': int
                - 'valid_subjects': int
                - 'excluded_subjects': list of subject_ids
                - 'exclusion_reasons': dict mapping subject_id to reason
        """
        df_valid = df[df['is_valid'].astype(str) == 'True'].copy()
        trial_type_counts = df_valid.groupby('subject_id')['trial_type'].value_counts().unstack(fill_value=0)

        for trial_type in ['PART_A', 'PART_B']:
            if trial_type not in trial_type_counts.columns:
                trial_type_counts[trial_type] = 0

        exclusion_reasons = {}
        for subj in trial_type_counts.index:
            part_a = trial_type_counts.loc[subj, 'PART_A']
            part_b = trial_type_counts.loc[subj, 'PART_B']
            if part_a == 0 and part_b == 0:
                exclusion_reasons[subj] = "No valid trials (PART_A=0, PART_B=0)"
            elif part_a == 0:
                exclusion_reasons[subj] = f"Missing PART_A (PART_B={part_b})"
            elif part_b == 0:
                exclusion_reasons[subj] = f"Missing PART_B (PART_A={part_a})"

        valid_mask = (trial_type_counts['PART_A'] >= 1) & (trial_type_counts['PART_B'] >= 1)

        return {
            'total_subjects': len(trial_type_counts),
            'valid_subjects': int(valid_mask.sum()),
            'excluded_subjects': list(exclusion_reasons.keys()),
            'exclusion_reasons': exclusion_reasons
        }
    

 
    


