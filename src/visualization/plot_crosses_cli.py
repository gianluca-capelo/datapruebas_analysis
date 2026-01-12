#!/usr/bin/env python
"""
CLI script to visualize TMT trial crosses.

Usage examples:
    # Plot crosses for a single trial
    python -m src.visualization.plot_crosses_cli --subject "UUID" --trial DATAPRUEBAS_0
    
    # Plot with custom time threshold
    python -m src.visualization.plot_crosses_cli --subject "UUID" --trial DATAPRUEBAS_0 --time-threshold 1000
"""

import argparse
import os

import matplotlib.pyplot as plt

from src import config
from src.loader import load_experiment
from src.visualization.segmentation_plotting import plot_crosses_segmentation


def main():
    parser = argparse.ArgumentParser(description='Plot trial with crosses highlighted')
    parser.add_argument('--subject', required=True, help='Subject ID')
    parser.add_argument('--trial', required=True, help='Trial ID')
    parser.add_argument('--origin', default='datapruebas', 
                        choices=['datapruebas', 'neuropruebas'],
                        help='Data origin (default: datapruebas)')
    parser.add_argument('--time-threshold', type=float, default=500.0,
                        help='Time threshold in ms for cross detection (default: 500.0)')
    parser.add_argument('--output-dir', default=config.FIGURES_DIR,
                        help='Output directory for figures')
    parser.add_argument('--cmap', default='tab10',
                        help='Colormap name (default: tab10)')
    args = parser.parse_args()
    
    print(f"Loading {args.origin}...")
    experiment = load_experiment(args.origin)
    
    if args.subject not in experiment.subjects:
        print(f"Error: Subject {args.subject} not found")
        print(f"Available subjects: {list(experiment.subjects.keys())[:5]}...")
        return
    
    subject = experiment.subjects[args.subject]
    trial = next((t for t in subject.testing_trials if t.id == args.trial), None)
    
    if not trial:
        print(f"Error: Trial {args.trial} not found for subject {args.subject}")
        available = [t.id for t in subject.testing_trials]
        print(f"Available trials: {available}")
        return
    
    print(f"Plotting crosses for trial {args.trial}...")
    fig = plot_crosses_segmentation(
        trial, 
        subject.target_radius, 
        time_threshold=args.time_threshold,
        cmap_name=args.cmap
    )
    
    if fig is None:
        print("Error: Could not create plot")
        return
    
    # Save figure
    os.makedirs(args.output_dir, exist_ok=True)
    safe_subject_id = args.subject[:8].replace("-", "_") if "-" in args.subject else args.subject[:8]
    safe_trial_id = args.trial.replace("-", "_") if "-" in args.trial else args.trial
    filename = f"crosses_plot_{safe_subject_id}_{safe_trial_id}.png"
    output_path = os.path.join(args.output_dir, filename)
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

