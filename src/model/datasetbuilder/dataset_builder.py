"""
DatasetBuilder: Constructs ML datasets from multiple cognitive task analyses.
"""

from typing import Tuple
import numpy as np
import pandas as pd

from src.loader.load_last_split import load_last_analysis
from src.loader.sst_analysis_loader import get_latest_sst_analysis
from src.loader.cdt_analysis_loader import get_latest_cdt_analysis
from src.loader.gonogo_analysis_loader import get_latest_gonogo_analysis


class DatasetBuilder:
    """
    Builds X, y datasets from cognitive task data.
    
    Supported datasets:
        - 'tmt_ssrt':   TMT features → SSRT target (Stop Signal Task)
        - 'tmt_k':      TMT features → K capacity target (Change Detection Task)
        - 'tmt_dprime': TMT features → d' sensitivity target (Go/No-Go Task)
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
        elif name == 'tmt_k':
            return self._build_tmt_k()
        elif name == 'tmt_dprime':
            return self._build_tmt_dprime()
        else:
            available = ['tmt_ssrt', 'tmt_k', 'tmt_dprime']
            raise ValueError(
                f"Unknown dataset: {name}. Available: {available}"
            )
    
    def _build_generic_dataset(self, loader_func, target_col: str, loader_name: str) -> Tuple[np.ndarray, np.ndarray, list, str]:
        """
        Generic helper to build TMT features vs Any Target.
        """
        # Load TMT data
        tmt_df, _ = load_last_analysis()
        tmt_agg = self._aggregate_tmt(tmt_df)
        
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

    def _build_tmt_k(self) -> Tuple[np.ndarray, np.ndarray, list, str]:
        """
        Build dataset with TMT features and K (CDT Capacity) as target.
        """
        target_name = 'K_6'
        return self._build_generic_dataset(
            loader_func=get_latest_cdt_analysis,
            target_col=target_name,
            loader_name="CDT"
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
    

 
    


