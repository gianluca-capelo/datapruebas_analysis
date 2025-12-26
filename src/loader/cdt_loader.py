"""
CDT (Change Detection Task) data loader.

Provides functions to load CDT experiment data from CSV files.
"""

import glob
import os

import pandas as pd

from src import config


def load_cdt_subjects(folder_path: str) -> dict[str, pd.DataFrame]:
    """
    Load CDT subject data from CSV files in a folder.
    
    Args:
        folder_path: Path to folder containing CSV files (one per subject).
        
    Returns:
        Dictionary mapping subject_id to their DataFrame.
        
    Note:
        This function does NOT modify the original files.
        Subject ID is extracted from the filename (without .csv extension).
    """
    subjects_dict = {}
    
    for filepath in glob.glob(os.path.join(folder_path, "*.csv")):
        filename = os.path.basename(filepath)
        subject_id = filename.replace(".csv", "")
        
        df = pd.read_csv(filepath, on_bad_lines="skip")
        
        # Ensure correct data types for CDT columns
        if "rt" in df.columns:
            df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
        if "stimamt" in df.columns:
            df["stimamt"] = pd.to_numeric(df["stimamt"], errors="coerce")
        
        subjects_dict[subject_id] = df
    
    return subjects_dict


def load_cdt_experiment(origin: str) -> dict[str, pd.DataFrame]:
    """
    Load CDT experiment data by origin name.
    
    Args:
        origin: Either "datapruebas" or "neuropruebas".
        
    Returns:
        Dictionary mapping subject_id to their DataFrame.
        
    Raises:
        ValueError: If origin is not recognized.
    """
    if origin == "datapruebas":
        folder_path = config.CDT_DATAPRUEBAS_PATH
    elif origin == "neuropruebas":
        folder_path = config.CDT_NEUROPRUEBAS_PATH
    else:
        raise ValueError(f"Unknown origin: {origin}. Must be 'datapruebas' or 'neuropruebas'.")
    
    return load_cdt_subjects(folder_path)


def load_all_cdt_experiments() -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load CDT data from both datapruebas and neuropruebas.
    
    Returns:
        Dictionary with keys 'datapruebas' and 'neuropruebas',
        each containing a dict mapping subject_id to DataFrame.
    """
    return {
        "datapruebas": load_cdt_experiment("datapruebas"),
        "neuropruebas": load_cdt_experiment("neuropruebas"),
    }

