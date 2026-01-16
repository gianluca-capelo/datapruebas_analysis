#!/usr/bin/env python
"""
Visualize the distribution of mean_refresh_rate across valid TMT trials.

Usage:
    python -m src.visualization.plot_refresh_rate
    python -m src.visualization.plot_refresh_rate --show
    python -m src.visualization.plot_refresh_rate --bins 50
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.loader.load_last_split import get_last_run_directory, get_run_configuration
from src.model.datasetbuilder.dataset_builder import DatasetBuilder


COLUMN_NAME = "mean_refresh_rate"


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


def load_valid_tmt_data() -> pd.DataFrame:
    """
    Load valid TMT trials using DatasetBuilder.

    Returns:
        DataFrame with valid TMT trials filtered by trial type coverage.
    """
    builder = DatasetBuilder()
    return builder._load_valid_tmt_trials()


def plot_refresh_rate_histogram(df: pd.DataFrame, ax: plt.Axes, bins: int = 30):
    """
    Plot histogram of mean_refresh_rate.

    Args:
        df: DataFrame with valid TMT trials containing mean_refresh_rate column.
        ax: Matplotlib axes to plot on.
        bins: Number of histogram bins.
    """
    data = df[COLUMN_NAME].dropna()

    ax.hist(data, bins=bins, color='#3498db', edgecolor='white', linewidth=0.8, alpha=0.8)

    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()

    ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='#2ecc71', linestyle=':', linewidth=2, label=f'Median: {median_val:.2f}')

    ax.set_xlabel('Mean Refresh Rate')
    ax.set_ylabel('Number of Trials')
    ax.set_title(f'Distribution of {COLUMN_NAME}\n(n={len(data):,} valid trials, std={std_val:.2f})')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_figure(df: pd.DataFrame, bins: int, output_path: str = None) -> plt.Figure:
    """
    Create the histogram figure.

    Args:
        df: DataFrame with valid TMT trials.
        bins: Number of histogram bins.
        output_path: Path to save figure (optional).

    Returns:
        Matplotlib figure.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_refresh_rate_histogram(df, ax, bins=bins)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#fafafa')
        print(f"Saved: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize mean_refresh_rate distribution")
    parser.add_argument('--output-dir', type=str, default=config.FIGURES_DIR,
                        help=f'Output directory (default: {config.FIGURES_DIR})')
    parser.add_argument('--show', action='store_true',
                        help='Show the plot interactively')
    parser.add_argument('--bins', type=int, default=30,
                        help='Number of histogram bins (default: 30)')

    args = parser.parse_args()

    print("Loading valid TMT trials...")
    df = load_valid_tmt_data()
    print(f"Loaded {len(df):,} valid trials")

    if COLUMN_NAME not in df.columns:
        print(f"\nError: Column '{COLUMN_NAME}' not found in data.")
        print(f"Available columns: {sorted(df.columns.tolist())}")
        return

    data = df[COLUMN_NAME].dropna()
    print(f"\n{'='*50}")
    print(f"STATISTICS FOR {COLUMN_NAME}")
    print(f"{'='*50}")
    print(f"Count:  {len(data):,}")
    print(f"Mean:   {data.mean():.4f}")
    print(f"Median: {data.median():.4f}")
    print(f"Std:    {data.std():.4f}")
    print(f"Min:    {data.min():.4f}")
    print(f"Max:    {data.max():.4f}")
    print(f"{'='*50}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = get_last_run_directory()
    run_config = get_run_configuration(run_dir)
    is_interpolated = run_config.get("interpolate_trajectory", False)
    suffix = "_interpolated" if is_interpolated else "_raw"
    output_path = os.path.join(args.output_dir, f'refresh_rate_histogram{suffix}.png')

    fig = create_figure(df, bins=args.bins, output_path=output_path)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
