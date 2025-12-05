#!/usr/bin/env python
"""
Script to analyze the anomalous speeds CSV and generate summary reports.

Generates two CSV outputs:
1. anomalies_by_origin.csv - Unique subjects count per data origin
2. anomalies_by_subject.csv - Per-subject anomaly details

Usage:
    python -m src.visualization.analyze_anomalies
    python -m src.visualization.analyze_anomalies --input custom_anomalies.csv
"""

import argparse
import os

import pandas as pd

from src import config


def analyze_anomalies(input_path):
    """Analyze the anomalies CSV and return summary dataframes.
    
    Args:
        input_path: Path to the anomalous_speeds.csv file
    
    Returns:
        tuple: (by_origin_df, by_subject_df)
    """
    # Read input CSV
    df = pd.read_csv(input_path)
    
    print(f"Loaded {len(df)} anomalies from {input_path}")
    print(f"Columns: {list(df.columns)}")
    
    # Report 1: Unique subjects by origin
    by_origin = df.groupby('origin')['subject_id'].nunique().reset_index()
    by_origin.columns = ['origin', 'unique_subjects']
    by_origin = by_origin.sort_values('unique_subjects', ascending=False)
    
    # Report 2: Details by subject
    by_subject = df.groupby(['subject_id', 'origin']).agg(
        total_anomalies=('speed_px_ms', 'count'),
        trials_affected=('trial_id', 'nunique'),
        max_speed_px_ms=('speed_px_ms', 'max'),
        mean_speed_px_ms=('speed_px_ms', 'mean')
    ).reset_index()
    
    # Round mean speed
    by_subject['mean_speed_px_ms'] = by_subject['mean_speed_px_ms'].round(2)
    
    # Sort by total anomalies descending
    by_subject = by_subject.sort_values('total_anomalies', ascending=False)
    
    return by_origin, by_subject


def main():
    parser = argparse.ArgumentParser(description='Analyze anomalous speeds CSV')
    parser.add_argument('--input', type=str, default='anomalous_speeds.csv',
                        help='Input CSV filename (default: anomalous_speeds.csv)')
    args = parser.parse_args()
    
    input_path = os.path.join(config.DATA_DIR, args.input)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Analyze
    by_origin, by_subject = analyze_anomalies(input_path)
    
    # Print summaries
    print("\n" + "="*60)
    print("REPORT 1: UNIQUE SUBJECTS BY ORIGIN")
    print("="*60)
    print(by_origin.to_string(index=False))
    print(f"\nTotal unique subjects: {by_origin['unique_subjects'].sum()}")
    
    print("\n" + "="*60)
    print("REPORT 2: ANOMALIES BY SUBJECT (top 10)")
    print("="*60)
    print(by_subject.head(10).to_string(index=False))
    
    # Save CSVs
    output_origin = os.path.join(config.DATA_DIR, 'anomalies_by_origin.csv')
    output_subject = os.path.join(config.DATA_DIR, 'anomalies_by_subject.csv')
    
    by_origin.to_csv(output_origin, index=False)
    by_subject.to_csv(output_subject, index=False)
    
    print("\n" + "="*60)
    print("OUTPUT FILES")
    print("="*60)
    print(f"✓ {output_origin}")
    print(f"✓ {output_subject}")


if __name__ == "__main__":
    main()

