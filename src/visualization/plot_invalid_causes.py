#!/usr/bin/env python
"""
Visualize the distribution of invalid_cause in the analysis data.

Usage:
    python -m src.visualization.plot_invalid_causes
    python -m src.visualization.plot_invalid_causes --analysis-path data/hand_analysis/2025-12-22_18-52-45/analysis.csv
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.loader.load_last_split import load_last_analysis


# Modern color palette
PALETTE = {
    'valid': '#2ecc71',           # Green
    'CUT_CRITERIA_ERROR': '#e74c3c',  # Red
    'INVALID_LENGTH': '#f39c12',      # Orange
    'NON_MONOTONIC_TIME': '#9b59b6',  # Purple
    'OTHER': '#95a5a6'                # Gray
}


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


def load_data(analysis_path: str = None) -> pd.DataFrame:
    """Load analysis data from CSV or last analysis."""
    if analysis_path and os.path.exists(analysis_path):
        df = pd.read_csv(analysis_path)
    else:
        train_df, _ = load_last_analysis()
        df = train_df
    
    # Fill NaN invalid_cause with 'Valid' for clarity
    df['invalid_cause_display'] = df['invalid_cause'].fillna('Valid')
    
    return df


def plot_overall_distribution(df: pd.DataFrame, ax: plt.Axes):
    """Plot overall distribution of validity status."""
    # Count by invalid_cause_display
    counts = df['invalid_cause_display'].value_counts()
    
    # Order: Valid first, then by count
    order = ['Valid'] + [x for x in counts.index if x != 'Valid']
    counts = counts.reindex(order)
    
    # Create colors
    colors = [PALETTE.get(x, PALETTE['OTHER']) for x in counts.index]
    
    # Plot horizontal bars
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, counts.values):
        pct = val / len(df) * 100
        ax.text(bar.get_width() + len(df) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:,} ({pct:.1f}%)', va='center', fontsize=10, fontweight='medium')
    
    ax.set_xlabel('Number of Trials')
    ax.set_title('Distribution of Trial Validity', pad=15)
    ax.set_xlim(0, counts.max() * 1.25)
    ax.invert_yaxis()
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_by_origin(df: pd.DataFrame, ax: plt.Axes):
    """Plot distribution grouped by experiment origin."""
    if 'experiment_origin' not in df.columns:
        ax.text(0.5, 0.5, 'No experiment_origin column', 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Prepare data
    grouped = df.groupby(['experiment_origin', 'invalid_cause_display']).size().unstack(fill_value=0)
    
    # Ensure consistent order
    all_causes = ['Valid', 'CUT_CRITERIA_ERROR', 'INVALID_LENGTH', 'NON_MONOTONIC_TIME']
    existing_causes = [c for c in all_causes if c in grouped.columns]
    grouped = grouped[existing_causes]
    
    # Calculate percentages
    grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100
    
    # Colors
    colors = [PALETTE.get(c, PALETTE['OTHER']) for c in existing_causes]
    
    # Stacked bar chart
    grouped_pct.plot(kind='barh', stacked=True, ax=ax, color=colors, 
                     edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Percentage (%)')
    ax.set_title('Validity by Data Origin', pad=15)
    ax.set_xlim(0, 100)
    ax.legend(title='Status', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    # Add total count annotations
    for i, (origin, row) in enumerate(grouped.iterrows()):
        total = row.sum()
        ax.text(101, i, f'n={total:,}', va='center', fontsize=9, color='#666')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_invalid_only(df: pd.DataFrame, ax: plt.Axes):
    """Plot pie chart of invalid causes (excluding valid trials)."""
    invalid_df = df[df['invalid_cause_display'] != 'Valid']
    
    if len(invalid_df) == 0:
        ax.text(0.5, 0.5, 'No invalid trials', ha='center', va='center', transform=ax.transAxes)
        return
    
    counts = invalid_df['invalid_cause_display'].value_counts()
    colors = [PALETTE.get(c, PALETTE['OTHER']) for c in counts.index]
    
    # Pie chart with modern styling
    wedges, texts, autotexts = ax.pie(
        counts.values, 
        labels=counts.index,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
        colors=colors,
        explode=[0.02] * len(counts),
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Style the text
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    ax.set_title(f'Invalid Trial Causes\n(n={len(invalid_df):,})', pad=10)


def plot_validity_rate_by_trial(df: pd.DataFrame, ax: plt.Axes):
    """Plot validity rate by trial_id."""
    if 'trial_id' not in df.columns:
        ax.text(0.5, 0.5, 'No trial_id column', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Calculate validity rate per trial_id
    validity_rate = df.groupby('trial_id').apply(
        lambda x: (x['invalid_cause_display'] == 'Valid').mean() * 100
    ).sort_values(ascending=True)
    
    # Limit to top 20 trial types for readability
    if len(validity_rate) > 20:
        validity_rate = validity_rate.tail(20)
    
    # Color based on validity rate
    colors = ['#2ecc71' if v >= 80 else '#f39c12' if v >= 50 else '#e74c3c' for v in validity_rate.values]
    
    bars = ax.barh(validity_rate.index, validity_rate.values, color=colors, 
                   edgecolor='white', linewidth=1)
    
    # Add percentage labels
    for bar, val in zip(bars, validity_rate.values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)
    
    ax.set_xlabel('Validity Rate (%)')
    ax.set_title('Validity Rate by Trial Type', pad=15)
    ax.set_xlim(0, 105)
    ax.axvline(x=80, color='#2ecc71', linestyle='--', alpha=0.5, label='80% threshold')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_dashboard(df: pd.DataFrame, output_path: str = None):
    """Create a comprehensive dashboard of invalid causes."""
    setup_style()
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.4)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot all visualizations
    plot_overall_distribution(df, ax1)
    plot_by_origin(df, ax2)
    plot_invalid_only(df, ax3)
    plot_validity_rate_by_trial(df, ax4)
    
    # Main title
    total = len(df)
    valid = (df['invalid_cause_display'] == 'Valid').sum()
    fig.suptitle(f'Trial Validity Analysis Dashboard\n'
                 f'Total: {total:,} trials | Valid: {valid:,} ({valid/total*100:.1f}%)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#fafafa')
        print(f"Saved: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize invalid_cause distribution")
    parser.add_argument('--analysis-path', type=str, default=None,
                        help='Path to analysis CSV (default: load from last analysis)')
    parser.add_argument('--output-dir', type=str, default=config.FIGURES_DIR,
                        help=f'Output directory (default: {config.FIGURES_DIR})')
    parser.add_argument('--show', action='store_true',
                        help='Show the plot interactively')
    
    args = parser.parse_args()
    
    print("Loading analysis data...")
    df = load_data(args.analysis_path)
    print(f"Loaded {len(df):,} trials")
    
    # Print summary
    print("\n" + "="*50)
    print("VALIDITY SUMMARY")
    print("="*50)
    print(df['invalid_cause_display'].value_counts().to_string())
    print("="*50 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'invalid_causes_dashboard.png')
    
    # Create dashboard
    fig = create_dashboard(df, output_path)
    
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

