#!/usr/bin/env python
"""
Script to explore time units in cursor_trail data.

Analyzes timestamps to determine if they are in milliseconds or seconds,
and estimates the sampling rate.

Usage:
    python -m src.visualization.explore_time_units
"""

import statistics
from src.loader import load_datapruebas, load_neuropruebas


def analyze_trial_times(trial):
    """Analyze time values from a single trial.
    
    Returns dict with time statistics or None if trial has insufficient data.
    """
    if not trial.cursor_trail or len(trial.cursor_trail) < 2:
        return None
    
    times = [p.time for p in trial.cursor_trail]
    
    # Intervalos entre muestras consecutivas
    intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
    
    return {
        "first_time": times[0],
        "last_time": times[-1],
        "duration": times[-1] - times[0],
        "rt": trial.rt,
        "num_samples": len(times),
        "mean_interval": statistics.mean(intervals) if intervals else 0,
        "min_interval": min(intervals) if intervals else 0,
        "max_interval": max(intervals) if intervals else 0,
    }


def collect_time_stats(experiment, origin_name):
    """Collect time statistics from all trials in an experiment."""
    all_stats = []
    
    for subject_id, subject in experiment.subjects.items():
        for trial in subject.testing_trials:
            stats = analyze_trial_times(trial)
            if stats:
                stats["origin"] = origin_name
                all_stats.append(stats)
    
    return all_stats


def print_summary(all_stats, origin_name):
    """Print summary of time statistics for one origin."""
    if not all_stats:
        print(f"  No data")
        return
    
    durations = [s["duration"] for s in all_stats]
    rts = [s["rt"] for s in all_stats if s["rt"]]
    intervals = [s["mean_interval"] for s in all_stats]
    first_times = [s["first_time"] for s in all_stats]
    
    print(f"\n{origin_name.upper()}:")
    print(f"  Trials analyzed: {len(all_stats)}")
    
    print(f"\n  First timestamp (start of trial):")
    print(f"    min={min(first_times)}, max={max(first_times)}, mean={statistics.mean(first_times):.1f}")
    
    print(f"\n  Trial duration (last_time - first_time):")
    print(f"    min={min(durations)}, max={max(durations)}, mean={statistics.mean(durations):.1f}")
    
    if rts:
        print(f"\n  RT field:")
        print(f"    min={min(rts)}, max={max(rts)}, mean={statistics.mean(rts):.1f}")
    
    print(f"\n  Mean interval between samples:")
    print(f"    min={min(intervals):.1f}, max={max(intervals):.1f}, mean={statistics.mean(intervals):.1f}")


def main():
    all_stats = []
    dp_stats = []
    np_stats = []
    
    # Datapruebas
    print("Loading datapruebas...")
    try:
        dp_experiment = load_datapruebas()
        dp_stats = collect_time_stats(dp_experiment, "datapruebas")
        all_stats.extend(dp_stats)
        print(f"  ✓ {len(dp_experiment.subjects)} subjects loaded")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Neuropruebas
    print("\nLoading neuropruebas...")
    try:
        np_experiment = load_neuropruebas()
        np_stats = collect_time_stats(np_experiment, "neuropruebas")
        all_stats.extend(np_stats)
        print(f"  ✓ {len(np_experiment.subjects)} subjects loaded")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TIME VALUES ANALYSIS")
    print("="*60)
    
    if dp_stats:
        print_summary(dp_stats, "datapruebas")
    if np_stats:
        print_summary(np_stats, "neuropruebas")
    
    # Inference
    print("\n" + "-"*60)
    print("INFERENCE:")
    if all_stats:
        mean_duration = statistics.mean([s["duration"] for s in all_stats])
        mean_interval = statistics.mean([s["mean_interval"] for s in all_stats])
        
        if mean_duration > 1000:
            print(f"  → Durations are large ({mean_duration:.0f}), likely MILLISECONDS")
        else:
            print(f"  → Durations are small ({mean_duration:.1f}), likely SECONDS")
        
        if mean_interval > 1:
            print(f"  → Sample interval ~{mean_interval:.1f}ms suggests ~{1000/mean_interval:.0f} Hz sampling rate")
        else:
            print(f"  → Sample interval ~{mean_interval:.3f}s suggests ~{1/mean_interval:.0f} Hz sampling rate")


if __name__ == "__main__":
    main()
