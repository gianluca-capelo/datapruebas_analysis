"""
Go/No-Go Analysis Loader.

Provides functions to run Go/No-Go analysis pipeline, save results with metadata,
following the same pattern as the SST and CDT analysis loaders.
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config
from src.analysis import GoNoGoTask
from src.loader.gonogo_loader import load_gonogo_experiment


def load_gonogo_analysis(save_results: bool = True) -> tuple[pd.DataFrame, Path | None]:
    """
    Run the Go/No-Go analysis pipeline for both datapruebas and neuropruebas.
    
    Args:
        save_results: If True, save results to disk with timestamp and metadata.
        
    Returns:
        Tuple of (metrics_df, save_path). save_path is None if save_results=False.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # 1) Generate a timestamped run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = _create_run_folder(timestamp) if save_results else None
    
    if run_dir:
        logging.info("Created run directory %s", run_dir)
    
    # 2) Execute analysis for both sources
    metrics_df = compute_gonogo_metrics()
    
    # 3) Save results if requested
    save_path = None
    if save_results and run_dir:
        save_path = _save_results(metrics_df, run_dir, timestamp)
    
    return metrics_df, save_path


def compute_gonogo_metrics() -> pd.DataFrame:
    """
    Compute Go/No-Go metrics for both datapruebas and neuropruebas.
    
    Returns:
        Combined DataFrame with metrics from both sources.
        
    Note:
        Population-level metrics (c, sensibilidad) are calculated on the
        combined dataset to maintain consistency with the original analysis.
    """
    # Load data from both sources
    logging.info("Loading Go/No-Go data from datapruebas...")
    datapruebas_data = load_gonogo_experiment("datapruebas")
    logging.info("Loaded %d subjects from datapruebas", len(datapruebas_data))
    
    logging.info("Loading Go/No-Go data from neuropruebas...")
    neuropruebas_data = load_gonogo_experiment("neuropruebas")
    logging.info("Loaded %d subjects from neuropruebas", len(neuropruebas_data))
    
    # Combine all subjects for analysis (population metrics need all subjects)
    all_subjects = {}
    
    # Add datapruebas subjects with origin tracking
    datapruebas_subjects = {}
    for subject_id, df in datapruebas_data.items():
        datapruebas_subjects[subject_id] = df
        all_subjects[subject_id] = df
    
    # Add neuropruebas subjects with origin tracking
    neuropruebas_subjects = {}
    for subject_id, df in neuropruebas_data.items():
        neuropruebas_subjects[subject_id] = df
        all_subjects[subject_id] = df
    
    logging.info("Running Go/No-Go analysis for all %d subjects...", len(all_subjects))
    
    # Run analysis on combined data (for correct population-level metrics)
    analyzer = GoNoGoTask()
    metrics_df = analyzer.run(all_subjects)
    
    # Add origin column
    def get_origin(subject_id):
        if subject_id in datapruebas_subjects:
            return "datapruebas"
        elif subject_id in neuropruebas_subjects:
            return "neuropruebas"
        return "unknown"
    
    metrics_df["origin"] = metrics_df["subject_id"].apply(get_origin)
    
    n_datapruebas = len(metrics_df[metrics_df["origin"] == "datapruebas"])
    n_neuropruebas = len(metrics_df[metrics_df["origin"] == "neuropruebas"])
    
    logging.info(
        "Total subjects analyzed: %d (datapruebas=%d, neuropruebas=%d)",
        len(metrics_df), n_datapruebas, n_neuropruebas
    )
    
    return metrics_df


def compute_gonogo_metrics_for_origin(origin: str) -> pd.DataFrame:
    """
    Compute Go/No-Go metrics for a single origin.
    
    WARNING: This calculates population metrics (c, sensibilidad) using only
    subjects from this origin, which may differ from the combined analysis.
    Use compute_gonogo_metrics() for consistent population-level metrics.
    
    Args:
        origin: Either "datapruebas" or "neuropruebas".
        
    Returns:
        DataFrame with Go/No-Go metrics for all subjects from the origin.
    """
    logging.info("Loading Go/No-Go data from %s...", origin)
    subjects_data = load_gonogo_experiment(origin)
    logging.info("Loaded %d subjects from %s", len(subjects_data), origin)
    
    logging.info("Running Go/No-Go analysis for %s...", origin)
    analyzer = GoNoGoTask()
    metrics_df = analyzer.run(subjects_data)
    
    # Add origin column
    metrics_df["origin"] = origin
    
    logging.info("Analyzed %d subjects from %s", len(metrics_df), origin)
    
    return metrics_df


def _create_run_folder(timestamp: str) -> Path:
    """
    Create a timestamped directory under Go/No-Go analysis folder.
    """
    root_dir = Path(config.GONOGO_ANALYSIS_FOLDER)
    run_dir = root_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _git_info() -> dict:
    """
    Retrieve current Git branch, commit SHA, message, and dirty state.
    """
    meta: dict = {}
    try:
        meta["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        meta["git_msg"] = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        meta["git_dirty"] = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD", "--"]
        ) != 0
    except subprocess.CalledProcessError as e:
        logging.warning("Failed to retrieve Git metadata: %s", e)
        meta["git_error"] = str(e)
    return meta


def _save_results(
    df: pd.DataFrame,
    run_dir: Path,
    timestamp: str,
) -> Path:
    """
    Save the metrics DataFrame and configuration metadata to the run directory.
    """
    # Build configuration
    run_config = {
        "timestamp": timestamp,
        "task": "GoNoGo",
        "n_subjects_total": len(df),
        "n_subjects_datapruebas": len(df[df["origin"] == "datapruebas"]),
        "n_subjects_neuropruebas": len(df[df["origin"] == "neuropruebas"]),
        "gonogo_datapruebas_path": config.GONOGO_DATAPRUEBAS_PATH,
        "gonogo_neuropruebas_path": config.GONOGO_NEUROPRUEBAS_PATH,
    }
    run_config.update(_git_info())
    
    # Write configuration JSON
    config_path = run_dir / "configuration.json"
    try:
        with config_path.open("w") as f:
            json.dump(run_config, f, indent=2)
        logging.info("Saved configuration to %s", config_path)
    except IOError as e:
        logging.error("Failed to write configuration file: %s", e)
    
    # Write results CSV
    data_path = run_dir / "gonogo_analysis.csv"
    try:
        df.to_csv(data_path, index=False)
        logging.info("Saved analysis results to %s", data_path)
    except IOError as e:
        logging.error("Failed to write analysis CSV: %s", e)
    
    # Also save separate files by origin for convenience
    for origin in ["datapruebas", "neuropruebas"]:
        origin_df = df[df["origin"] == origin]
        origin_path = run_dir / f"gonogo_analysis_{origin}.csv"
        origin_df.to_csv(origin_path, index=False)
    
    return data_path


def get_latest_gonogo_analysis() -> tuple[pd.DataFrame, dict] | None:
    """
    Load the most recent Go/No-Go analysis results.
    
    Returns:
        Tuple of (metrics_df, config_dict) or None if no analysis found.
    """
    results_dir = Path(config.GONOGO_ANALYSIS_FOLDER)
    
    if not results_dir.exists():
        logging.warning("Go/No-Go analysis directory does not exist: %s", results_dir)
        return None
    
    # Find all timestamped directories
    run_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name,
        reverse=True
    )
    
    if not run_dirs:
        logging.warning("No Go/No-Go analysis runs found in %s", results_dir)
        return None
    
    latest_dir = run_dirs[0]
    logging.info("Loading latest Go/No-Go analysis from %s", latest_dir)
    
    # Load data
    data_path = latest_dir / "gonogo_analysis.csv"
    config_path = latest_dir / "configuration.json"
    
    if not data_path.exists():
        logging.error("Analysis CSV not found: %s", data_path)
        return None
    
    df = pd.read_csv(data_path)
    
    config_dict = {}
    if config_path.exists():
        with config_path.open() as f:
            config_dict = json.load(f)
    
    return df, config_dict


def run_gonogo_analysis():
    """
    CLI entry point for running Go/No-Go analysis.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Go/No-Go analysis.")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to disk (useful for testing)."
    )
    args = parser.parse_args()
    
    metrics_df, save_path = load_gonogo_analysis(save_results=not args.no_save)
    
    print("\n" + "=" * 60)
    print("Go/No-Go Analysis Complete")
    print("=" * 60)
    print(f"Total subjects: {len(metrics_df)}")
    print(f"  - Datapruebas: {len(metrics_df[metrics_df['origin'] == 'datapruebas'])}")
    print(f"  - Neuropruebas: {len(metrics_df[metrics_df['origin'] == 'neuropruebas'])}")
    
    if "HR" in metrics_df.columns:
        print(f"\nMean HR: {metrics_df['HR'].mean():.2f} (SD={metrics_df['HR'].std():.2f})")
        print(f"Mean FA: {metrics_df['FA'].mean():.2f} (SD={metrics_df['FA'].std():.2f})")
        print(f"Mean accuracy: {metrics_df['accuracy'].mean():.2f} (SD={metrics_df['accuracy'].std():.2f})")
    
    if save_path:
        print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    run_gonogo_analysis()

