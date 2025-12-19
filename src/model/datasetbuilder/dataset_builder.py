"""
DatasetBuilder: Constructs ML datasets from multiple cognitive task analyses.
"""

from typing import Tuple
import numpy as np
import pandas as pd

from src.loader.load_last_split import load_last_analysis
from src.loader.sst_analysis_loader import get_latest_sst_analysis


class DatasetBuilder:
    """
    Builds X, y datasets from cognitive task data.
    
    Supported datasets:
        - 'tmt_ssrt': TMT features → SSRT target
    """
    
    TMT_FEATURE_COLS = [
        'rt', 'mean_speed', 'std_speed', 'peak_speed',
        'mean_acceleration', 'std_acceleration', 'peak_acceleration',
        'hesitation_time', 'travel_time', 'search_time',
        'hesitation_distance', 'travel_distance', 'search_distance',
        'total_hesitations', 'average_duration', 'max_duration',
        'distance_difference_from_ideal', 'area_difference_from_ideal',
        'intra_target_time', 'inter_target_time'
    ]
    
    def get_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Get X, y, feature_names for a dataset.
        
        Args:
            name: Dataset name (e.g., 'tmt_ssrt')
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature column names
        """
        match name:
            case 'tmt_ssrt':
                return self._build_tmt_ssrt()
            case _:
                raise ValueError(f"Unknown dataset: {name}. "
                               f"Available: ['tmt_ssrt']")
    
    def _build_tmt_ssrt(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Build dataset with TMT features and SSRT as target.
        
        Returns:
            X: TMT features aggregated by subject
            y: SSRT values from SST analysis
            feature_names: List of TMT feature names
        """
        # Load TMT data
        tmt_df, _ = load_last_analysis()
        tmt_agg = self._aggregate_tmt(tmt_df)
        
        # Load SST data
        sst_result = get_latest_sst_analysis()
        if sst_result is None:
            raise RuntimeError("No SST analysis found. Run SST analysis first.")
        sst_df = sst_result[0][['subject_id', 'ssrt']]
        
        # Merge on subject_id
        merged = pd.merge(tmt_agg, sst_df, on='subject_id', how='inner')
        
        if len(merged) == 0:
            raise RuntimeError("No matching subjects between TMT and SST data.")
        
        # Extract X and y
        feature_cols = [c for c in merged.columns if c not in ['subject_id', 'ssrt']]
        X = merged[feature_cols].values
        y = merged['ssrt'].values
        
        return X, y, feature_cols
    
    def _aggregate_tmt(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate TMT trial-level data to subject-level.
        
        Pivots by trial_type (PART_A, PART_B) and computes mean per subject.
        
        Args:
            df: TMT DataFrame with trial-level data
            
        Returns:
            DataFrame with one row per subject and columns like 'rt_PART_A', 'rt_PART_B'
        """
        # Filter valid trials only (handle both bool and string 'True')
        df_valid = df[df['is_valid'].astype(str) == 'True'].copy()
        
        # Get available feature columns
        feature_cols = [c for c in self.TMT_FEATURE_COLS if c in df_valid.columns]
        
        # Convert feature columns to numeric, coercing errors to NaN
        for col in feature_cols:
            df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce')
        
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

