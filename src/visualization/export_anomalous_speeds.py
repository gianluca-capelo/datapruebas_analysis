#!/usr/bin/env python
"""
Script to export trials with anomalous cursor speeds to CSV.

Usage:
    python -m src.visualization.export_anomalous_speeds
    python -m src.visualization.export_anomalous_speeds --threshold 5
    python -m src.visualization.export_anomalous_speeds --output anomalies.csv
"""

import argparse
import csv
import math
import os
from src import config
from src.mapper.datapruebas.datapruebas_mapper import DatapruebasTMTMapper
from src.mapper.neuropruebas.neuropruebas_mapper import NeuropruebasTMTMapper
from neurotask.tmt.metrics.targets_touched import count_correctly_touched_targets


def load_datapruebas():
    """Load the datapruebas experiment."""
    dataset_path = os.path.join(
        config.DATA_DIR,
        "raw/tmt/datapruebas/subjects",
        config.EXPERIMENT_FILE_NAME
    )
    mapper = DatapruebasTMTMapper()
    return mapper.map(dataset_path)


def load_neuropruebas():
    """Load the neuropruebas experiment."""
    dataset_path = os.path.join(
        config.DATA_DIR,
        "raw/tmt/neuropruebas/subjects"
    )
    mapper = NeuropruebasTMTMapper()
    return mapper.map(dataset_path)


def find_anomalous_speeds(experiment, origin, threshold):
    """Find all cursor movements exceeding the speed threshold.
    
    Uses get_cursor_trail_from_start() to only analyze data from the first click.
    
    Args:
        experiment: TMTExperiment object
        origin: String identifying data source ("datapruebas" or "neuropruebas")
        threshold: Speed threshold in px/ms
    
    Returns:
        List of dicts with anomaly details
    """
    anomalies = []
    
    for subject_id, subject in experiment.subjects.items():
        target_radius = subject.target_radius
        
        # Extraer px2mm de session_data
        session_data = subject.session_data
        px2mm = session_data.get('px2mm') if session_data else None
        if px2mm is None:
            px2mm = float('nan')
        
        for trial in subject.testing_trials:
            cursor_trail = trial.get_cursor_trail_from_start()
            if not cursor_trail or len(cursor_trail) < 2:
                continue
            
            # Calcular targets correctos para este trial
            try:
                correct_targets = count_correctly_touched_targets(trial, target_radius)
            except Exception:
                correct_targets = float('nan')
            
            for i in range(1, len(cursor_trail)):
                curr = cursor_trail[i]
                prev = cursor_trail[i-1]
                
                dt = curr.time - prev.time
                if dt <= 0:
                    continue
                
                dx = curr.position.x - prev.position.x
                dy = curr.position.y - prev.position.y
                distance = math.sqrt(dx**2 + dy**2)
                speed = distance / dt
                
                if speed > threshold:
                    anomalies.append({
                        'subject_id': subject_id,
                        'trial_id': trial.id,
                        'speed_px_ms': round(speed, 2),
                        'speed_px_s': round(speed * 1000, 0),
                        'origin': origin,
                        'correct_targets': correct_targets,
                        'px2mm': px2mm
                    })
    
    return anomalies


def main():
    parser = argparse.ArgumentParser(description='Export anomalous speed cases to CSV')
    parser.add_argument('--threshold', type=float, default=8.0,
                        help='Speed threshold in px/ms (default: 8.0)')
    parser.add_argument('--output', type=str, default='anomalous_speeds.csv',
                        help='Output CSV filename (default: anomalous_speeds.csv)')
    args = parser.parse_args()
    
    all_anomalies = []
    
    # Datapruebas
    print(f"Loading datapruebas...")
    try:
        dp_experiment = load_datapruebas()
        dp_anomalies = find_anomalous_speeds(dp_experiment, "datapruebas", args.threshold)
        all_anomalies.extend(dp_anomalies)
        print(f"  ✓ Found {len(dp_anomalies)} anomalies in datapruebas")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Neuropruebas
    print(f"\nLoading neuropruebas...")
    try:
        np_experiment = load_neuropruebas()
        np_anomalies = find_anomalous_speeds(np_experiment, "neuropruebas", args.threshold)
        all_anomalies.extend(np_anomalies)
        print(f"  ✓ Found {len(np_anomalies)} anomalies in neuropruebas")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Export to CSV
    output_path = os.path.join(config.DATA_DIR, args.output)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id', 'trial_id', 'speed_px_ms', 'speed_px_s', 'origin', 'correct_targets', 'px2mm'])
        writer.writeheader()
        writer.writerows(all_anomalies)
    
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Threshold: {args.threshold} px/ms ({args.threshold * 1000} px/s)")
    print(f"Total anomalies found: {len(all_anomalies)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()

