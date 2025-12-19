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
            name: Dataset name (e.g., 'tmt_ssrt')
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature column names
            target_name: Name of the target column
        """
        if name == 'tmt_ssrt':
            return self._build_tmt_ssrt()
        else:
            raise ValueError(
                f"Unknown dataset: {name}. Available: ['tmt_ssrt']"
            )
    
    def _build_tmt_ssrt(self) -> Tuple[np.ndarray, np.ndarray, list, str]:
        """
        Build dataset with TMT features and SSRT as target.
        
        Returns:
            X: TMT features aggregated by subject
            y: SSRT values from SST analysis
            feature_names: List of TMT feature names
            target_name: 'ssrt'
        """
        target_name = 'ssrt'
        
        # Load TMT data
        tmt_df, _ = load_last_analysis()
        tmt_agg = self._aggregate_tmt(tmt_df)
        
        # Load SST data
        sst_result = get_latest_sst_analysis()
        if sst_result is None:
            raise RuntimeError("No SST analysis found. Run SST analysis first.")
        sst_df = sst_result[0][['subject_id', target_name]]
        
        # Merge on subject_id
        merged = pd.merge(tmt_agg, sst_df, on='subject_id', how='inner')
        
        if len(merged) == 0:
            raise RuntimeError("No matching subjects between TMT and SST data.")
        
        # Extract X and y
        feature_cols = [c for c in merged.columns if c not in ['subject_id', target_name]]
        X = merged[feature_cols].values
        y = merged[target_name].values
        
        return X, y, feature_cols, target_name
    
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

