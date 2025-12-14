#!/usr/bin/env python
"""
Plot trials highlighting points with anomalous cursor speeds.

Uses get_cursor_trail_from_start() for consistency with other analysis.

Usage:
    # Spatial plot (X/Y position)
    python -m src.visualization.plot_speed_anomalies --subject "UUID" --trial "TRIAL_ID"
    python -m src.visualization.plot_speed_anomalies --subject "UUID" --trial "TRIAL_ID" --threshold 5
    
    # Temporal plot (time vs distance)
    python -m src.visualization.plot_speed_anomalies --subject "UUID" --trial "TRIAL_ID" --mode temporal
"""

import argparse
import math
import os

import matplotlib.pyplot as plt

from neurotask.tmt.metrics.speed_metrics import calculate_speeds
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


def plot_trial_with_speed_anomalies(trial, target_radius, threshold, canvas_size=750):
    """Plot trial highlighting points with anomalous speeds.
    
    Uses get_cursor_trail_from_start() for consistency with other analysis.
    
    Args:
        trial: TMTTrial object
        target_radius: Radius of targets in pixels
        threshold: Speed threshold in px/ms
        canvas_size: Canvas size in pixels (default 750)
    
    Returns:
        matplotlib figure
    """
    cursor_trail = trial.get_cursor_trail_from_start()
    if not cursor_trail or len(cursor_trail) < 2:
        return None
    
    # Calculate speeds using neurotask (prepend 0 to align with cursor_trail indices)
    speeds = [0] + calculate_speeds(cursor_trail, raise_on_threshold=False)
    
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
    
    # Set limits based on canvas_size (use full canvas for proper visualization)
    margin = 20
    ax.set_xlim(-margin, canvas_size + margin)
    ax.set_ylim(canvas_size + margin, -margin)  # Invert Y axis (screen coordinates)
    ax.set_aspect('equal', adjustable='box')
    
    return fig, len(anomaly_x), len(cursor_trail)


def plot_distance_over_time(trial, threshold):
    """Plot distance and speed over time, highlighting speed anomalies.
    
    Creates three subplots:
    - Top: Cumulative distance vs time (line plot)
    - Middle: Instantaneous distance vs time (bar plot)
    - Bottom: Speed vs time (line plot with threshold)
    
    Args:
        trial: TMTTrial object
        threshold: Speed threshold in px/ms
    
    Returns:
        tuple: (fig, anomaly_count, total_points)
    """
    cursor_trail = trial.get_cursor_trail_from_start()
    if not cursor_trail or len(cursor_trail) < 2:
        return None
    
    # Extract times
    times = [p.time for p in cursor_trail]
    
    # Calculate speeds using neurotask (prepend 0 to align with cursor_trail indices)
    speeds = [0] + calculate_speeds(cursor_trail, raise_on_threshold=False)
    
    # Calculate distances (neurotask doesn't expose this directly)
    distances = [0]
    cumulative = [0]
    for i in range(1, len(cursor_trail)):
        curr = cursor_trail[i]
        prev = cursor_trail[i-1]
        dx = curr.position.x - prev.position.x
        dy = curr.position.y - prev.position.y
        dist = math.sqrt(dx**2 + dy**2)
        distances.append(dist)
        cumulative.append(cumulative[-1] + dist)
    
    # Identify anomaly indices
    anomaly_idx = [i for i, s in enumerate(speeds) if s > threshold]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Cumulative distance
    ax1.plot(times, cumulative, 'b-', linewidth=1.5, label='Distancia acumulada')
    if anomaly_idx:
        ax1.scatter(
            [times[i] for i in anomaly_idx],
            [cumulative[i] for i in anomaly_idx],
            c='red', s=100, marker='X', zorder=5,
            label=f'Anomalía (>{threshold} px/ms)'
        )
    ax1.set_ylabel('Distancia acumulada (px)')
    ax1.set_title(
        f'Trial {trial.id} - Métricas Temporales\n'
        f'{len(anomaly_idx)} anomalías / {len(cursor_trail)} puntos | Threshold: {threshold} px/ms'
    )
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Instantaneous distance (per sample)
    bar_colors = ['red' if i in anomaly_idx else 'steelblue' for i in range(len(times))]
    ax2.bar(times, distances, width=15, color=bar_colors, alpha=0.7)
    ax2.set_ylabel('Distancia instantánea (px)')
    ax2.grid(True, alpha=0.3)
    
    # Add legend for bar chart
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Normal'),
        Patch(facecolor='red', alpha=0.7, label=f'Anomalía (>{threshold} px/ms)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # Plot 3: Speed vs time
    ax3.plot(times, speeds, 'g-', linewidth=1, alpha=0.7, label='Velocidad')
    ax3.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold} px/ms)')
    if anomaly_idx:
        ax3.scatter(
            [times[i] for i in anomaly_idx],
            [speeds[i] for i in anomaly_idx],
            c='red', s=100, marker='X', zorder=5,
            label=f'Anomalías ({len(anomaly_idx)})'
        )
    ax3.set_xlabel('Tiempo (ms)')
    ax3.set_ylabel('Velocidad (px/ms)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Set y-axis limit for speed plot to show threshold clearly
    max_normal_speed = max([s for s in speeds if s <= threshold], default=threshold)
    y_max = max(threshold * 2, max_normal_speed * 1.5)
    if anomaly_idx:
        # If there are anomalies, show them but cap the y-axis for readability
        y_max = min(max(speeds) * 1.1, threshold * 5)
    ax3.set_ylim(0, y_max)
    
    plt.tight_layout()
    
    return fig, len(anomaly_idx), len(cursor_trail)


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
    parser.add_argument('--mode', default='spatial',
                        choices=['spatial', 'temporal'],
                        help='Plot mode: spatial (X/Y position) or temporal (time vs distance)')
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
    
    # Create plot based on mode
    if args.mode == 'spatial':
        result = plot_trial_with_speed_anomalies(trial, subject.target_radius, args.threshold, subject.canvas_size)
        mode_suffix = 'spatial'
    else:  # temporal
        result = plot_distance_over_time(trial, args.threshold)
        mode_suffix = 'temporal'
    
    if result is None:
        print("Error: Not enough data to plot")
        return
    
    fig, anomaly_count, total_points = result
    
    print(f"Found {anomaly_count} anomalous points out of {total_points} "
          f"({anomaly_count/total_points*100:.1f}%)")
    
    # Save figure
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"speed_anomalies_{args.subject[:8]}_{args.trial}_{mode_suffix}.png"
    output_path = os.path.join(args.output_dir, filename)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

