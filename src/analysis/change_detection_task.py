"""
Change Detection Task (CDT) analysis.

CDT is a visual working memory task where participants must remember
colored squares and detect changes. The main metric is Cowan's K,
which estimates working memory capacity.
"""

import logging

import numpy as np
import pandas as pd




class ChangeDetectionTask:
    """
    Analyzer for Change Detection Task data.
    
    Computes working memory capacity (Cowan's K) and performance metrics.
    """

    # Strings to detect experiment boundaries
    EXPERIMENT_START = "<h1 style = 'margin: 30px;'> A continuación comenzará"
    EXPERIMENT_END = "<h3>El experimento finalizó"

    CDT_MIN_ACCURACY = 0.60
    MAX_OMISSION_RATE = 0.20
    
    def run(self, subjects_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Run CDT analysis on all subjects.
        
        Args:
            subjects_data: Dictionary mapping subject_id to their DataFrame.
            
        Returns:
            DataFrame with metrics for each subject.
        """
        all_metrics = []
        failed_subjects = []
        
        for subject_id, subject_df in subjects_data.items():
            print(f"Processing subject: {subject_id}")
            
            try:
                metrics = self._analyze_subject(subject_id, subject_df)
                if metrics is not None:
                    all_metrics.append(metrics)
                else:
                    failed_subjects.append(subject_id)
            except Exception as e:
                logging.exception(f"Error processing subject {subject_id}: {e}")
                failed_subjects.append(subject_id)
        
        print(f"Skipped {len(failed_subjects)} subjects")
        
        if not all_metrics:
            return pd.DataFrame()
        
        return pd.DataFrame(all_metrics)
    
    def _analyze_subject(self, subject_id: str, df: pd.DataFrame) -> dict | None:
        """
        Analyze a single subject's CDT data.
        
        Args:
            subject_id: Subject identifier.
            df: DataFrame with subject's trial data.
            
        Returns:
            Dictionary with metrics or None if analysis fails.
        """
        # Try to find experiment boundaries
        try:
            idx_range = self._calculate_range_of_interest(df)
        except Exception as e:
            logging.warning(f"Could not find experiment boundaries for {subject_id}: {e}")
            # Fall back to using test_part column if available
            if "test_part" not in df.columns:
                logging.error(f"No test_part column found for {subject_id}")
                return None
            idx_range = df[df["test_part"] == "test"].index
        
        # Ensure rt is numeric
        df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
        
        # Get test trials only
        if "test_part" in df.columns:
            test_trials = df[df["test_part"] == "test"].copy()
        else:
            test_trials = df.iloc[list(idx_range)].copy()
        
        if len(test_trials) == 0:
            logging.warning(f"No test trials found for {subject_id}")
            return None
        
        # Ensure stimamt is numeric for comparison
        test_trials["stimamt"] = pd.to_numeric(test_trials["stimamt"], errors="coerce")
        
        # Separate by set size
        trials_4 = test_trials[test_trials["stimamt"] == 4]
        trials_6 = test_trials[test_trials["stimamt"] == 6]
        
        # Count correct trials by set size
        correct_4 = self._count_correct(trials_4)
        correct_6 = self._count_correct(trials_6)
        
        # Count trials without response
        no_response_4 = trials_4["response"].isna().sum()
        no_response_6 = trials_6["response"].isna().sum()
        trials_no_contestados = no_response_4 + no_response_6

        
        # Total test trials per set size (should be ~60 each)
        n_trials_4 = len(trials_4)
        n_trials_6 = len(trials_6)
        n_trials_total = n_trials_4 + n_trials_6
        
        if (trials_no_contestados/n_trials_total) > self.MAX_OMISSION_RATE:
            logging.warning(f" OMISSION_RATE > {self.MAX_OMISSION_RATE} for {subject_id}")
            return None


        if n_trials_total == 0:
            logging.warning(f"No valid trials for {subject_id}")
            return None
        
        # Calculate accuracy per set size (excluding non-responses)
        responded_4 = n_trials_4 - no_response_4
        responded_6 = n_trials_6 - no_response_6
        
        if responded_4 > 0:
            accuracy_4 = correct_4 / responded_4
        else:
            accuracy_4 = 0
            
        if responded_6 > 0:
            accuracy_6 = correct_6 / responded_6
        else:
            accuracy_6 = 0

        if accuracy_4 < 0 or accuracy_6 < 0:
            logging.warning(f"Accuracy 4 or 6 < 0 for {subject_id}")
            return None

        
        # Calculate Cowan's K: K = N * (2 * accuracy - 1)
        K_4 = 4 * (2 * accuracy_4 - 1)
        K_6 = 6 * (2 * accuracy_6 - 1)
        
        # Overall accuracy
        total_correct = correct_4 + correct_6
        total_responded = responded_4 + responded_6
        accuracy = total_correct / total_responded if total_responded > 0 else 0

        if accuracy < 0:
            logging.warning(f"Accuracy < 0 for {subject_id}")
            return None
        elif accuracy < self.CDT_MIN_ACCURACY:
            logging.warning(f"Accuracy < 0.6 for {subject_id}")
            return None
            

        
        # Response times (only for responded trials)
        rt_values = test_trials["rt"].dropna()
        media_rt = rt_values.mean() if len(rt_values) > 0 else np.nan
        mediana_rt = rt_values.median() if len(rt_values) > 0 else np.nan
        
        # Metadata
        view_dist_cm = self._get_view_distance(df)
        
        return {
            "subject_id": subject_id,
            "media_rt": round(media_rt, 2) if not np.isnan(media_rt) else np.nan,
            "mediana_rt": round(mediana_rt, 2) if not np.isnan(mediana_rt) else np.nan,
            "trials_no_contestados": int(trials_no_contestados),
            "accuracy": round(accuracy, 2),
            "accuracy_4": round(accuracy_4, 2),
            "accuracy_6": round(accuracy_6, 2),
            "K_4": round(K_4, 2),
            "K_6": round(K_6, 2),
            "n_trials_4": n_trials_4,
            "n_trials_6": n_trials_6,
            "view_dist_cm": view_dist_cm,
        }
    
    def _count_correct(self, trials: pd.DataFrame) -> int:
        """Count correct trials, handling different data formats."""
        if len(trials) == 0:
            return 0
        
        correct_col = trials["correct"]
        
        # Handle different formats: True/False, "true"/"false", 1/0
        count = 0
        for val in correct_col:
            if val is True or val == "true" or val == "True" or val == 1:
                count += 1
        
        return count
    
    def _calculate_range_of_interest(self, df: pd.DataFrame) -> range:
        """
        Find the range of rows containing actual experiment data.
        
        Uses stimulus column to find start and end markers.
        """
        if "stimulus" not in df.columns:
            raise ValueError("No stimulus column found")
        
        # Find start
        start_mask = df["stimulus"].str.contains(self.EXPERIMENT_START, na=False)
        if not start_mask.any():
            raise ValueError("Experiment start marker not found")
        first_row = start_mask.idxmax()
        
        # Find end
        end_mask = df["stimulus"].str.contains(self.EXPERIMENT_END, na=False)
        if not end_mask.any():
            # Use all rows after start
            return range(first_row + 1, len(df))
        last_row = end_mask.idxmax()
        
        return range(first_row + 1, last_row)
    
    def _get_view_distance(self, df: pd.DataFrame) -> float | None:
        """Extract viewing distance from virtual chinrest data."""
        if "view_dist_mm" not in df.columns:
            return None
        
        view_dist = pd.to_numeric(df["view_dist_mm"], errors="coerce")
        valid_values = view_dist[view_dist > 0]
        
        if len(valid_values) == 0:
            return None
        
        return round(float(valid_values.iloc[0]) / 10, 2)  # Convert mm to cm

