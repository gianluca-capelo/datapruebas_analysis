#!/usr/bin/env python
"""
CLI script to visualize TMT trial segmentation plots.

Usage examples:
    # Plot a single trial for a specific subject
    python -m src.visualization.plot_trials_cli --mode single --subject 14 --trial 9

    # Plot all valid trials for a given trial_id
    python -m src.visualization.plot_trials_cli --mode all --trial 9
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from neurotask.tmt.tmt_analyzer import TMTAnalyzer

from src import config
from src.hand_analysis.loader.load_last_split import load_last_analysis
from src.hand_analysis.mapper.psychopy_mapper import PsychopyTMTMapper
from src.visualization.segmentation_plotting import plot_segmentation


def load_experiment():
    """Load the TMT experiment using TMTAnalyzer."""
    analyzer = TMTAnalyzer(
        mapper=PsychopyTMTMapper(),
        dataset_path=config.PYXATIONS_PATIENTS_DATA_DIR,
        output_path=config.DATA_DIR
    )
    return analyzer.get_experiment()


def get_valid_analysis(min_targets: int = 10, trial_id: int = None):
    """
    Load and filter valid analysis data.
    
    Args:
        min_targets: Minimum number of correct target touches to include.
        trial_id: If provided, filter to only this trial_id.
    
    Returns:
        DataFrame with valid analysis data.
    """
    train_set, _ = load_last_analysis()
    valid_analysis = train_set[train_set['is_valid'] == True].copy()
    valid_analysis = valid_analysis[
        valid_analysis['non_cut_correct_targets_touches'] > min_targets
    ].copy()
    
    if trial_id is not None:
        valid_analysis = valid_analysis[valid_analysis['trial_id'] == trial_id]
    
    return valid_analysis


def plot_trial(subject, trial_id: int, subject_analysis, output_path: str):
    """
    Plot and save a segmentation figure for a specific trial.
    
    Args:
        subject: The subject object from the experiment.
        trial_id: The trial ID to plot.
        subject_analysis: DataFrame with analysis data for this subject.
        output_path: Path to save the figure.
    """
    for trial in subject.testing_trials:
        is_valid = subject_analysis[subject_analysis['trial_id'] == trial.id]
        if is_valid.empty:
            continue

        if trial.id != trial_id:
            continue

        speed_threshold = subject_analysis['speed_threshold'].values[0]
        fig = plot_segmentation(
            trial, 
            subject.target_radius, 
            speed_threshold, 
            cmap_name="Set1"
        )

        ax = fig.axes[0]
        ax.set_xlim(500, 1500)

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")
        return True
    
    return False


def format_subject_id(subject_id: int) -> str:
    """Format subject ID with leading zeros (e.g., 14 -> '0014')."""
    return f"{subject_id:04d}"


def plot_single_trial(experiment, valid_analysis, subject_id: int, trial_id: int, output_dir: str):
    """
    Plot a single trial for a specific subject.
    
    Args:
        experiment: The loaded TMT experiment.
        valid_analysis: DataFrame with valid analysis data.
        subject_id: The subject ID.
        trial_id: The trial ID.
        output_dir: Directory to save the figure.
    """
    subject_key = format_subject_id(subject_id)
    
    if subject_key not in experiment.subjects:
        print(f"Error: Subject {subject_id} (key: {subject_key}) not found in experiment.")
        return False
    
    subject = experiment.subjects[subject_key]
    subject_analysis = valid_analysis[valid_analysis['subject_id'] == subject_id]
    
    if subject_analysis.empty:
        print(f"Error: No valid analysis data for subject {subject_id} with trial_id {trial_id}")
        return False
    
    output_path = os.path.join(
        output_dir, 
        f"segmentation_plot_subject_{subject_id}_trial_{trial_id}.png"
    )
    
    success = plot_trial(subject, trial_id, subject_analysis, output_path)
    if not success:
        print(f"Warning: Could not plot trial {trial_id} for subject {subject_id}")
    
    return success


def plot_all_trials(experiment, valid_analysis, trial_id: int, output_dir: str):
    """
    Plot all valid trials for a given trial_id.
    
    Args:
        experiment: The loaded TMT experiment.
        valid_analysis: DataFrame with valid analysis data.
        trial_id: The trial ID to plot.
        output_dir: Directory to save the figures.
    """
    plotted_count = 0
    
    for idx, row in valid_analysis.iterrows():
        subject_id = row['subject_id']
        subject_key = format_subject_id(subject_id)
        
        if subject_key not in experiment.subjects:
            print(f"Warning: Subject {subject_id} not found in experiment, skipping.")
            continue
        
        try:
            subject = experiment.subjects[subject_key]
            subject_analysis = valid_analysis[valid_analysis['subject_id'] == subject_id]
            
            output_path = os.path.join(
                output_dir,
                f"segmentation_plot_subject_{subject_id}_trial_{trial_id}.png"
            )
            
            success = plot_trial(subject, trial_id, subject_analysis, output_path)
            if success:
                plotted_count += 1
        except Exception as e:
            print(f"Error plotting subject {subject_id}: {e}")
            continue
    
    print(f"\nTotal plots saved: {plotted_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot TMT trial segmentation figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Plot a single trial:
    python -m src.visualization.plot_trials_cli --mode single --subject 14 --trial 9

  Plot all valid trials:
    python -m src.visualization.plot_trials_cli --mode all --trial 9
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["single", "all"],
        required=True,
        help="'single' to plot one subject's trial, 'all' to plot all valid trials"
    )
    parser.add_argument(
        "--subject",
        type=int,
        help="Subject ID (required for mode='single')"
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=9,
        help="Trial ID to plot (default: 9)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.FIGURES_DIR,
        help=f"Output directory for figures (default: {config.FIGURES_DIR})"
    )
    parser.add_argument(
        "--min-targets",
        type=int,
        default=10,
        help="Minimum correct target touches to include (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "single" and args.subject is None:
        parser.error("--subject is required when --mode is 'single'")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading experiment...")
    experiment = load_experiment()
    
    print("Loading valid analysis data...")
    valid_analysis = get_valid_analysis(
        min_targets=args.min_targets,
        trial_id=args.trial
    )
    
    print(f"Found {len(valid_analysis)} valid trials with trial_id={args.trial}")
    
    if args.mode == "single":
        print(f"\nPlotting trial {args.trial} for subject {args.subject}...")
        plot_single_trial(
            experiment, 
            valid_analysis, 
            args.subject, 
            args.trial, 
            args.output_dir
        )
    else:
        print(f"\nPlotting all valid trials with trial_id={args.trial}...")
        plot_all_trials(
            experiment, 
            valid_analysis, 
            args.trial, 
            args.output_dir
        )


if __name__ == "__main__":
    main()

