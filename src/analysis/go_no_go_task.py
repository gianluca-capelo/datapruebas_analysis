"""
Go/No-Go Task Analysis.

This module provides the GoNoGoTask class for analyzing Go/No-Go experiments,
a task measuring inhibitory control.

The analysis logic is preserved exactly from the original implementation in
analisis-neuropruebas/neuropsychdata/neuropsychdata/go_no_go.py to ensure
consistent results.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _calculate_range_of_interest(subject_df: pd.DataFrame, start_string: str, end_string: str) -> range:
    """
    Calculate the range of rows of interest for the subject.
    
    Args:
        subject_df: Subject's DataFrame.
        start_string: String marking experiment start.
        end_string: String marking experiment end.
        
    Returns:
        Range of row indices for experiment trials.
    """
    first_row = _find_target_row_index(subject_df, start_string)
    last_row = _find_target_row_index(subject_df, end_string)
    return range(first_row + 1, last_row)


def _find_target_row_index(subject_df: pd.DataFrame, target_string: str) -> int:
    """
    Find the index of the row where the 'stimulus' column starts with the target string.
    
    Args:
        subject_df: Subject's DataFrame.
        target_string: String to search for.
        
    Returns:
        Row index where stimulus starts with target_string.
        
    Raises:
        Exception: If no row found starting with target_string.
    """
    target_row_index = subject_df["stimulus"].str.startswith(target_string, na=False).idxmax()
    
    if not subject_df["stimulus"].iloc[target_row_index].startswith(target_string):
        raise Exception(f"No row found starting with '{target_string}' for subject.")
    
    return target_row_index


def _get_first_value_or_nan(df: pd.DataFrame, column: str):
    """Get the first value of a column or NaN if not available."""
    if column in df.columns and not df[column].empty:
        return df[column].iat[0]
    else:
        return np.nan


@dataclass
class GoNoGoTask:
    """
    Analyze Go/No-Go experiments.
    
    Go/No-Go is an inhibitory control task where:
    - Blue stimulus (Go): Press spacebar
    - Orange stimulus (No-Go): Do NOT press
    
    Metrics calculated:
    - HR (Hit Rate): Correct responses to Go trials
    - FA (False Alarm): Incorrect responses to No-Go trials
    - c: Response criterion (standardized)
    - sensibilidad: Discrimination ability (standardized)
    - eficiencia: Acc(Go) - Acc(NoGo)
    """
    
    # Experiment boundary markers (preserved from original)
    EXPERIMENT_START = "<h1> A continuación comenzará"
    EXPERIMENT_END = "<h3>El experimento finalizó. En"
    MIN_RT = 150        # Mínimo fisiológico (ms)
    MIN_ACCURACY = 0.60 # Precisión mínima (60%)
    MAX_FA_RATE = 0.90  # Para detectar inversión de comportamiento
    
    def run(self, subjects_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Run Go/No-Go analysis for multiple subjects.
        
        Args:
            subjects_data: Dictionary mapping subject_id to their DataFrame.
            
        Returns:
            DataFrame with metrics for all subjects.
            
        Note:
            Population-level metrics (c, sensibilidad) are calculated AFTER
            processing all subjects, as they require population mean/std.
        """
        results = []
        failed_subjects = []
        
        for subject_id, subject_df in subjects_data.items():
            logging.debug("Processing subject: %s", subject_id)
            
            result = self._analyze_subject(subject_id, subject_df, failed_subjects)
            if result is not None:
                results.append(result)
        
        if not results:
            logging.warning("No subjects were successfully analyzed")
            return pd.DataFrame()
        
        # Create DataFrame with basic metrics
        df = pd.DataFrame(results)
        
        # Calculate population-level metrics (require all subjects first)
        df = self._calculate_c(df)
        df = self._calculate_hr_fa_sin_estandarizar(df)
        df = self._calculate_sensibilidad(df)
        
        self._log_failed_subjects(failed_subjects)
        
        return df
    
    def _analyze_subject(
        self, 
        subject_id: str, 
        subject_df: pd.DataFrame,
        failed_subjects: list
    ) -> dict | None:
        """
        Analyze a single subject's Go/No-Go data.
        
        Returns None if subject should be skipped.
        """
        # Check for valid stimulus column
        if subject_df["stimulus"].isna().all():
            logging.error(
                "Skipping subject %s: stimulus column is all NaN", subject_id
            )
            failed_subjects.append((subject_id, "stimulus_column_all_nan"))
            return None
        
        # Calculate range of interest (experiment boundaries)
        try:
            idx_gonogo = _calculate_range_of_interest(
                subject_df, self.EXPERIMENT_START, self.EXPERIMENT_END
            )
        except Exception as e:
            logging.error(
                "Skipping subject %s: could not find experiment boundaries: %s", 
                subject_id, e
            )
            failed_subjects.append((subject_id, "error_calculating_range_of_interest"))
            return None
        
        total_trials = len(idx_gonogo)
        logging.debug("Total trials: %d", total_trials)
        
        # Ensure RT is numeric
        subject_df["rt"] = pd.to_numeric(subject_df["rt"], errors='coerce')

        # NUEVO: Filtrar respuestas anticipadas (< 150ms) antes de calcular medias
        # Solo en los trials de interés
        rt_validos = subject_df["rt"].iloc[idx_gonogo]
        rt_validos = rt_validos[rt_validos >= self.MIN_RT] 
        
        if len(rt_validos) == 0:
             logging.warning("Subject %s excluded: No valid RTs > 150ms", subject_id)
             failed_subjects.append((subject_id, "all_rts_premature"))
             return None

        media_rt = rt_validos.mean()
        mediana_rt = rt_validos.median()
        
        # CORRECCIÓN DE LÓGICA:
        # Si la columna stimulus no tiene "blue", lo inferimos de 'correct_response'.
        # Lógica: ' ' (espacio) = Go (Blue). NaN/Vacío = No-Go (Orange).
        stimulus_slice = subject_df["stimulus"].iloc[idx_gonogo]
        if stimulus_slice.str.count("blue").sum() == 0:
            if "correct_response" not in subject_df.columns:
                logging.warning("No 'correct_response' column for subject %s", subject_id)
                return None
            
            # Usamos lógica vectorizada: Si es espacio (' ') es blue, sino orange
            # Esto arregla el error de conteo 45 vs 105
            inferred_stimulus = subject_df["correct_response"].iloc[idx_gonogo].apply(
                lambda x: "blue" if str(x) == " " else "orange"
            )
            subject_df.loc[idx_gonogo, "stimulus"] = inferred_stimulus
        
        # Count correct responses
        if "true" in subject_df["correct"].values:
            cant_resp_correctas = subject_df["correct"][idx_gonogo].str.count("true").sum()
        else:
            cant_resp_correctas = subject_df["correct"][idx_gonogo].sum()
        
        # Accuracy
        accuracy = cant_resp_correctas / total_trials

        # NUEVO: Exclusión por baja performance
        if accuracy < self.MIN_ACCURACY:
            logging.warning("Subject %s excluded: Low accuracy (%.2f)", subject_id, accuracy)
            failed_subjects.append((subject_id, "low_accuracy"))
            return None

        
        # Validate accuracy against jsPsych (if available)
        if "accuracy" in subject_df.columns:
            jspsych_accuracy = subject_df["accuracy"].iloc[0]
            if not np.isnan(jspsych_accuracy):
                if not abs(jspsych_accuracy - accuracy * 100) <= 2:
                    logging.warning(
                        "Accuracy mismatch for %s: calculated=%.2f%%, jspsych=%.2f%%",
                        subject_id, accuracy * 100, jspsych_accuracy
                    )
                    failed_subjects.append((subject_id, "accuracy_mismatch_jspsych"))
        
        # Count Go/No-Go trials
        cant_go_trials = subject_df["stimulus"][idx_gonogo].str.count("blue").sum()
        cant_nogo_trials = subject_df["stimulus"][idx_gonogo].str.count("orange").sum()
        
        # Convert correct column to boolean (preserve original logic)
        subject_df["correct"] = subject_df["correct"].replace("true", True)
        subject_df["correct"] = subject_df["correct"].replace("false", False)
        
        # Calculate TP, TN, FP, FN (Signal Detection Theory)
        # TP = Hit (correctly respond to Go)
        # TN = Correct Rejection (correctly withhold to No-Go)
        # FP = False Alarm (incorrectly respond to No-Go)
        # FN = Miss (incorrectly withhold to Go)
        TP = subject_df[
            (subject_df["stimulus"][idx_gonogo].str.contains("blue", na=False))
            & (subject_df["correct"] == True)
        ]
        TN = subject_df[
            (subject_df["stimulus"][idx_gonogo].str.contains("orange", na=False))
            & (subject_df["correct"] == True)
        ]
        FP = subject_df[
            (subject_df["stimulus"][idx_gonogo].str.contains("orange", na=False))
            & (subject_df["correct"] == False)
        ]
        FN = subject_df[
            (subject_df["stimulus"][idx_gonogo].str.contains("blue", na=False))
            & (subject_df["correct"] == False)
        ]
        
        logging.debug("TP=%d, TN=%d, FP=%d, FN=%d", len(TP), len(TN), len(FP), len(FN))
        
        # Secondary accuracy check
        accuracy_test = (len(TP) + len(TN)) / total_trials
        if accuracy_test != accuracy:
            logging.warning(
                "Secondary accuracy check failed for %s: %.2f vs %.2f",
                subject_id, accuracy, accuracy_test
            )
            failed_subjects.append((subject_id, "accuracy_mismatch_test"))
        
        # Calculate False Alarm rate and Hit Rate
        FA = len(FP) / cant_nogo_trials if cant_nogo_trials > 0 else 0
        HR = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) > 0 else 0
        
        # NUEVO: Exclusión si el sujeto invirtió la tarea (Falsas alarmas > Aciertos)
        # Esto indica d' negativo (o que presionó en Naranja y no en Azul)
        if FA > HR:
             logging.warning("Subject %s excluded: Inverted performance (FA > HR)", subject_id)
             failed_subjects.append((subject_id, "inverted_performance_fa_gt_hr"))
             return None
        
        # Calculate efficiency metrics
        acc_go = len(TP) / cant_go_trials if cant_go_trials > 0 else 0
        acc_no_go = len(TN) / cant_nogo_trials if cant_nogo_trials > 0 else 0
        eficiencia = acc_go - acc_no_go
        
        # Get metadata
        recorded_at = _get_first_value_or_nan(subject_df, "recorded_at")
        user_agent = _get_first_value_or_nan(subject_df, "user_agent")
        device = _get_first_value_or_nan(subject_df, "device")
        
        browser = None
        if "browser" in subject_df.columns and "browser_version" in subject_df.columns:
            browser = f"{subject_df['browser'].iloc[0]} {subject_df['browser_version'].iloc[0]}"
        
        platform = None
        if "platform" in subject_df.columns and "platform_version" in subject_df.columns:
            platform = f"{subject_df['platform'].iloc[0]} {subject_df['platform_version'].iloc[0]}"
        
        return {
            "subject_id": subject_id,
            "media_rt": round(media_rt, 2),
            "mediana_rt": round(mediana_rt, 2),
            "accuracy": round(accuracy, 2),
            "FA": round(FA, 2),
            "HR": round(HR, 2),
            "eficiencia": round(eficiencia, 2),
            "recorded_at": recorded_at,
            "user_agent": user_agent,
            "device": device,
            "browser": browser,
            "platform": platform,
            "n_go_trials": int(cant_go_trials),
            "n_nogo_trials": int(cant_nogo_trials),
        }
    
    def _calculate_c(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate response criterion 'c' (requires population statistics).
        
        c = -0.5 * ((HR - HR_mean) / HR_std + (FA - FA_mean) / FA_std)
        """
        HR_mean = df["HR"].mean()
        HR_std = df["HR"].std()
        FA_mean = df["FA"].mean()
        FA_std = df["FA"].std()
        
        c_values = []
        for _, row in df.iterrows():
            HR = row["HR"]
            FA = row["FA"]
            
            # Avoid division by zero
            if HR_std == 0 or FA_std == 0:
                c = np.nan
            else:
                c = -0.5 * (((HR - HR_mean) / HR_std) + ((FA - FA_mean) / FA_std))
            
            c_values.append(round(c, 2) if not np.isnan(c) else np.nan)
        
        df["c"] = c_values
        return df
    
    def _calculate_hr_fa_sin_estandarizar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate HR+FA without standardization.
        
        HR_FA_sin_estandarizar = -0.5 * (HR + FA)
        """
        hr_fa_values = []
        for _, row in df.iterrows():
            HR = row["HR"]
            FA = row["FA"]
            hr_fa = -0.5 * (HR + FA)
            hr_fa_values.append(round(hr_fa, 2))
        
        df["HR_FA_sin_estandarizar"] = hr_fa_values
        return df
    
    def _calculate_sensibilidad(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sensitivity/discrimination (requires population statistics).
        
        sensibilidad = ((HR - HR_mean) / HR_std) - ((FA - FA_mean) / FA_std)
        """
        HR_mean = df["HR"].mean()
        HR_std = df["HR"].std()
        FA_mean = df["FA"].mean()
        FA_std = df["FA"].std()
        
        sensibilidad_values = []
        for _, row in df.iterrows():
            HR = row["HR"]
            FA = row["FA"]
            
            # Avoid division by zero
            if HR_std == 0 or FA_std == 0:
                sensibilidad = np.nan
            else:
                sensibilidad = ((HR - HR_mean) / HR_std) - ((FA - FA_mean) / FA_std)
            
            sensibilidad_values.append(round(sensibilidad, 2) if not np.isnan(sensibilidad) else np.nan)
        
        df["sensibilidad"] = sensibilidad_values
        return df
    
    def _log_failed_subjects(self, failed_subjects: list) -> None:
        """Log summary of failed subjects."""
        if failed_subjects:
            logging.warning("Summary of failed subjects (%d total):", len(failed_subjects))
            for subject, reason in failed_subjects:
                if reason == "stimulus_column_all_nan":
                    logging.warning("  %s: stimulus column all NaN", subject)
                elif reason == "error_calculating_range_of_interest":
                    logging.warning("  %s: could not find experiment boundaries", subject)
                elif reason == "accuracy_mismatch_jspsych":
                    logging.warning("  %s: accuracy mismatch with jsPsych", subject)
                elif reason == "accuracy_mismatch_test":
                    logging.warning("  %s: accuracy mismatch in secondary check", subject)

