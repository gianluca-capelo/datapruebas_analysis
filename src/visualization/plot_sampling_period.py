#!/usr/bin/env python
"""
Visualize the distribution of sampling period across valid TMT trials.

Usage:
    python -m src.visualization.plot_refresh_rate
    python -m src.visualization.plot_refresh_rate --show
    python -m src.visualization.plot_refresh_rate --bins 50
    python -m src.visualization.plot_refresh_rate --date 2026-01-16_18-57-15
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.loader.load_last_split import get_last_run_directory, get_run_configuration
from src.model.datasetbuilder.dataset_builder import DatasetBuilder


def get_run_directory_by_date(date: str) -> Path:
    """Find run directory matching a date prefix."""
    base_dir = Path(config.HAND_ANALYSIS_FOLDER)
    candidates = [
        d for d in base_dir.iterdir()
        if d.is_dir()
        and d.name.startswith(date)
        and (d / "configuration.json").exists()
        and (d / "analysis.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No run directories found for date {date!r}")
    return max(candidates, key=lambda d: d.name)


# Options: "mean" or "median"
SAMPLING_PERIOD_STAT = "median"

COLUMN_NAME = f"{SAMPLING_PERIOD_STAT}_sampling_period"
NON_CUT_COLUMN_NAME = f"non_cut_{SAMPLING_PERIOD_STAT}_sampling_period"


def setup_style():
    """Configure seaborn for modern aesthetics."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'figure.facecolor': '#fafafa',
        'axes.facecolor': '#fafafa',
        'grid.alpha': 0.4,
    })


def load_valid_tmt_data(run_dir: Path = None) -> pd.DataFrame:
    """
    Load valid TMT trials using DatasetBuilder.

    Args:
        run_dir: Specific run directory to load from. If None, uses latest.

    Returns:
        DataFrame with valid TMT trials filtered by trial type coverage.
    """
    builder = DatasetBuilder()
    if run_dir is None:
        return builder._load_valid_tmt_trials()

    df = pd.read_csv(run_dir / "analysis.csv", on_bad_lines='warn')
    return builder._get_valid_tmt_trials(df)


def plot_refresh_rate_histogram(df: pd.DataFrame, ax: plt.Axes, column_name: str, bins: int = 30):
    """
    Plot histogram of sampling period.

    Args:
        df: DataFrame with valid TMT trials containing the column.
        ax: Matplotlib axes to plot on.
        column_name: Name of the column to plot.
        bins: Number of histogram bins.
    """
    data = df[column_name].dropna()

    ax.hist(data, bins=bins, color='#3498db', edgecolor='white', linewidth=0.8, alpha=0.8)

    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()

    ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='#2ecc71', linestyle=':', linewidth=2, label=f'Median: {median_val:.2f}')

    ax.set_xlabel('Sampling Period (ms)')
    ax.set_ylabel('Number of Trials')
    ax.set_title(f'Distribution of {column_name}\n(n={len(data):,} valid trials, std={std_val:.2f})')

    # Add info about which stat is being used
    ax.text(0.02, 0.98, f'Stat: {SAMPLING_PERIOD_STAT}', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontstyle='italic', color='gray')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_figure(df: pd.DataFrame, column_name: str, bins: int, output_path: str = None) -> plt.Figure:
    """
    Create the histogram figure.

    Args:
        df: DataFrame with valid TMT trials.
        column_name: Name of the column to plot.
        bins: Number of histogram bins.
        output_path: Path to save figure (optional).

    Returns:
        Matplotlib figure.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_refresh_rate_histogram(df, ax, column_name=column_name, bins=bins)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#fafafa')
        print(f"Saved: {output_path}")

    return fig


def print_stats(df: pd.DataFrame, column_name: str):
    """Print statistics for a column."""
    data = df[column_name].dropna()
    print(f"\n{'='*50}")
    print(f"STATISTICS FOR {column_name}")
    print(f"{'='*50}")
    print(f"Count:  {len(data):,}")
    print(f"Mean:   {data.mean():.4f}")
    print(f"Median: {data.median():.4f}")
    print(f"Std:    {data.std():.4f}")
    print(f"Min:    {data.min():.4f}")
    print(f"Max:    {data.max():.4f}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize sampling period distribution")
    parser.add_argument('--output-dir', type=str, default=config.FIGURES_DIR,
                        help=f'Output directory (default: {config.FIGURES_DIR})')
    parser.add_argument('--show', action='store_true',
                        help='Show the plot interactively')
    parser.add_argument('--bins', type=int, default=30,
                        help='Number of histogram bins (default: 30)')
    parser.add_argument('--date', type=str, default=None,
                        help='Analysis date/timestamp (e.g., 2026-01-16_18-57-15). If not specified, uses latest.')

    args = parser.parse_args()

    # Get run directory
    if args.date:
        run_dir = get_run_directory_by_date(args.date)
    else:
        run_dir = get_last_run_directory()

    print(f"Using analysis: {run_dir.name}")
    run_config = get_run_configuration(run_dir)
    print(f"interpolate_trajectory: {run_config.get('interpolate_trajectory', False)}")

    print("Loading valid TMT trials...")
    df = load_valid_tmt_data(run_dir)
    print(f"Loaded {len(df):,} valid trials")

    os.makedirs(args.output_dir, exist_ok=True)
    is_interpolated = run_config.get("interpolate_trajectory", False)
    suffix = "_interpolated" if is_interpolated else "_raw"

    # Plot both columns
    for col_name in [COLUMN_NAME, NON_CUT_COLUMN_NAME]:
        if col_name not in df.columns:
            print(f"\nError: Column '{col_name}' not found in data.")
            continue

        print_stats(df, col_name)

        col_suffix = "non_cut_" if "non_cut" in col_name else ""
        output_path = os.path.join(args.output_dir, f'{col_suffix}{SAMPLING_PERIOD_STAT}_sampling_period_histogram{suffix}.png')

        fig = create_figure(df, column_name=col_name, bins=args.bins, output_path=output_path)

        if args.show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
