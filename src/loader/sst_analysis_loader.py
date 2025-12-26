"""
SST Analysis Loader.

Provides functions to run SST analysis pipeline, save results with metadata,
following the same pattern as the TMT analysis_loader.
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config
from src.analysis import StopSignalTask
from src.loader.sst_loader import load_sst_experiment


def load_sst_analysis(save_results: bool = True) -> tuple[pd.DataFrame, Path | None]:
    """
    Run the SST analysis pipeline for both datapruebas and neuropruebas.
    
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
    metrics_df = compute_sst_metrics()
    
    # 3) Save results if requested
    save_path = None
    if save_results and run_dir:
        save_path = _save_results(metrics_df, run_dir, timestamp)
    
    return metrics_df, save_path


def compute_sst_metrics() -> pd.DataFrame:
    """
    Compute SST metrics for both datapruebas and neuropruebas.
    
    Returns:
        Combined DataFrame with metrics from both sources.
    """
    datapruebas_metrics = compute_sst_metrics_for_origin("datapruebas")
    neuropruebas_metrics = compute_sst_metrics_for_origin("neuropruebas")
    
    # Combine results
    metrics = pd.concat([datapruebas_metrics, neuropruebas_metrics], ignore_index=True)
    
    logging.info(
        "Total subjects analyzed: %d (datapruebas=%d, neuropruebas=%d)",
        len(metrics), len(datapruebas_metrics), len(neuropruebas_metrics)
    )
    
    return metrics


def compute_sst_metrics_for_origin(origin: str) -> pd.DataFrame:
    """
    Compute SST metrics for a single origin.
    
    Args:
        origin: Either "datapruebas" or "neuropruebas".
        
    Returns:
        DataFrame with SST metrics for all subjects from the origin.
    """
    logging.info("Loading SST data from %s...", origin)
    subjects_data = load_sst_experiment(origin)
    logging.info("Loaded %d subjects from %s", len(subjects_data), origin)
    
    logging.info("Running SST analysis for %s...", origin)
    analyzer = StopSignalTask()
    metrics_df = analyzer.run(subjects_data)
    
    # Add origin column
    metrics_df["origin"] = origin
    
    # Rename Subject to subject_id for consistency with TMT
    metrics_df = metrics_df.rename(columns={"Subject": "subject_id"})
    
    logging.info("Analyzed %d subjects from %s", len(metrics_df), origin)
    
    return metrics_df


def _create_run_folder(timestamp: str) -> Path:
    """
    Create a timestamped directory under SST analysis folder.
    """
    root_dir = Path(config.SST_ANALYSIS_FOLDER)
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
        "task": "SST",
        "n_subjects_total": len(df),
        "n_subjects_datapruebas": len(df[df["origin"] == "datapruebas"]),
        "n_subjects_neuropruebas": len(df[df["origin"] == "neuropruebas"]),
        "sst_datapruebas_path": config.SST_DATAPRUEBAS_PATH,
        "sst_neuropruebas_path": config.SST_NEUROPRUEBAS_PATH,
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
    data_path = run_dir / "sst_analysis.csv"
    try:
        df.to_csv(data_path, index=False)
        logging.info("Saved analysis results to %s", data_path)
    except IOError as e:
        logging.error("Failed to write analysis CSV: %s", e)
    
    # Also save separate files by origin for convenience
    for origin in ["datapruebas", "neuropruebas"]:
        origin_df = df[df["origin"] == origin]
        origin_path = run_dir / f"sst_analysis_{origin}.csv"
        origin_df.to_csv(origin_path, index=False)
    
    return data_path


def get_latest_sst_analysis() -> tuple[pd.DataFrame, dict] | None:
    """
    Load the most recent SST analysis results.
    
    Returns:
        Tuple of (metrics_df, config_dict) or None if no analysis found.
    """
    results_dir = Path(config.SST_ANALYSIS_FOLDER)
    
    if not results_dir.exists():
        logging.warning("SST analysis directory does not exist: %s", results_dir)
        return None
    
    # Find all timestamped directories
    run_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name,
        reverse=True
    )
    
    if not run_dirs:
        logging.warning("No SST analysis runs found in %s", results_dir)
        return None
    
    latest_dir = run_dirs[0]
    logging.info("Loading latest SST analysis from %s", latest_dir)
    
    # Load data
    data_path = latest_dir / "sst_analysis.csv"
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


def run_sst_analysis():
    """
    CLI entry point for running SST analysis.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SST (Stop Signal Task) analysis.")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to disk (useful for testing)."
    )
    args = parser.parse_args()
    
    metrics_df, save_path = load_sst_analysis(save_results=not args.no_save)
    
    print("\n" + "=" * 60)
    print("SST Analysis Complete")
    print("=" * 60)
    print(f"Total subjects: {len(metrics_df)}")
    print(f"  - Datapruebas: {len(metrics_df[metrics_df['origin'] == 'datapruebas'])}")
    print(f"  - Neuropruebas: {len(metrics_df[metrics_df['origin'] == 'neuropruebas'])}")
    print(f"\nMean SSRT: {metrics_df['ssrt'].mean():.1f} ms (SD={metrics_df['ssrt'].std():.1f})")
    
    if save_path:
        print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    run_sst_analysis()

