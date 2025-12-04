#!/usr/bin/env python
"""
Plot trials highlighting points with anomalous cursor speeds.

Uses get_cursor_trail_from_start() for consistency with other analysis.

Usage:
    python -m src.visualization.plot_speed_anomalies --subject "UUID" --trial "TRIAL_ID"
    python -m src.visualization.plot_speed_anomalies --subject "UUID" --trial "TRIAL_ID" --threshold 5
"""

import argparse
import math
import os

import matplotlib.pyplot as plt

from src import config
from src.mapper.datapruebas.datapruebas_mapper import DatapruebasTMTMapper
from src.mapper.neuropruebas.neuropruebas_mapper import NeuropruebasTMTMapper


def load_experiment(origin):
    """Load experiment using the appropriate mapper."""
    if origin == "datapruebas":
        dataset_path = os.path.join(
            config.DATA_DIR,
            "raw/tmt/datapruebas/subjects",
            config.EXPERIMENT_FILE_NAME
        )
        mapper = DatapruebasTMTMapper()
    else:
        dataset_path = os.path.join(
            config.DATA_DIR,
            "raw/tmt/neuropruebas/subjects"
        )
        mapper = NeuropruebasTMTMapper()
    return mapper.map(dataset_path)


def calculate_speeds(cursor_trail):
    """Calculate speed between consecutive points.
    
    Args:
        cursor_trail: List of CursorInfo objects
    
    Returns:
        List of speeds (px/ms) for each point (first point is 0)
    """
    speeds = [0]  # First point has no speed
    
    for i in range(1, len(cursor_trail)):
        curr = cursor_trail[i]
        prev = cursor_trail[i-1]
        
        dt = curr.time - prev.time
        if dt > 0:
            dx = curr.position.x - prev.position.x
            dy = curr.position.y - prev.position.y
            distance = math.sqrt(dx**2 + dy**2)
            speeds.append(distance / dt)
        else:
            speeds.append(0)
    
    return speeds


def plot_trial_with_speed_anomalies(trial, target_radius, threshold):
    """Plot trial highlighting points with anomalous speeds.
    
    Uses get_cursor_trail_from_start() for consistency with other analysis.
    
    Args:
        trial: TMTTrial object
        target_radius: Radius of targets in pixels
        threshold: Speed threshold in px/ms
    
    Returns:
        matplotlib figure
    """
    cursor_trail = trial.get_cursor_trail_from_start()
    if not cursor_trail or len(cursor_trail) < 2:
        return None
    
    # Calculate speeds
    speeds = calculate_speeds(cursor_trail)
    
    # Separate normal and anomalous points
    normal_x, normal_y = [], []
    anomaly_x, anomaly_y, anomaly_speeds = [], [], []
    
    for i, point in enumerate(cursor_trail):
        if speeds[i] > threshold:
            anomaly_x.append(point.position.x)
            anomaly_y.append(point.position.y)
            anomaly_speeds.append(speeds[i])
        else:
            normal_x.append(point.position.x)
            normal_y.append(point.position.y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw trajectory line first (background)
    all_x = [p.position.x for p in cursor_trail]
    all_y = [p.position.y for p in cursor_trail]
    ax.plot(all_x, all_y, 'b-', alpha=0.2, linewidth=0.5, zorder=1)
    
    # Draw targets
    for target in trial.stimuli:
        circle = plt.Circle(
            (target.position.x, target.position.y),
            target_radius,
            color='steelblue',
            alpha=0.3,
            zorder=2
        )
        ax.add_patch(circle)
        ax.text(
            target.position.x, target.position.y,
            target.content,
            fontsize=8, ha='center', va='center',
            zorder=6
        )
    
    # Plot normal points (small, gray)
    ax.scatter(normal_x, normal_y, c='gray', s=10, alpha=0.5,
               label=f'Normal ({len(normal_x)} pts)', zorder=3)
    
    # Plot anomalous points (large, red X)
    ax.scatter(anomaly_x, anomaly_y, c='red', s=100, marker='X',
               label=f'Anomaly >{threshold} px/ms ({len(anomaly_x)} pts)', zorder=5)
    
    # Mark start position
    if trial.start:
        ax.scatter(
            trial.start.position.x, trial.start.position.y,
            c='green', s=150, marker='*',
            label='First click', zorder=7
        )
    
    ax.legend(loc='upper right')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(
        f'Trial {trial.id} - Speed Anomalies\n'
        f'Threshold: {threshold} px/ms | '
        f'{len(anomaly_x)} anomalies / {len(cursor_trail)} points'
    )
    
    # Set limits with margin
    margin = 50
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Invert Y axis (screen coordinates)
    ax.set_aspect('equal', adjustable='box')
    
    return fig, len(anomaly_x), len(cursor_trail)


def main():
    parser = argparse.ArgumentParser(description='Plot trial with speed anomalies highlighted')
    parser.add_argument('--subject', required=True, help='Subject ID')
    parser.add_argument('--trial', required=True, help='Trial ID')
    parser.add_argument('--origin', default='datapruebas', 
                        choices=['datapruebas', 'neuropruebas'],
                        help='Data origin (default: datapruebas)')
    parser.add_argument('--threshold', type=float, default=8.0,
                        help='Speed threshold in px/ms (default: 8.0)')
    parser.add_argument('--output-dir', default=config.FIGURES_DIR,
                        help='Output directory for figures')
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
    
    # Create plot
    result = plot_trial_with_speed_anomalies(trial, subject.target_radius, args.threshold)
    
    if result is None:
        print("Error: Not enough data to plot")
        return
    
    fig, anomaly_count, total_points = result
    
    print(f"Found {anomaly_count} anomalous points out of {total_points} "
          f"({anomaly_count/total_points*100:.1f}%)")
    
    # Save figure
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"speed_anomalies_{args.subject[:8]}_{args.trial}.png"
    output_path = os.path.join(args.output_dir, filename)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

