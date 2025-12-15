#!/usr/bin/env python
"""
Script to explore canvas_size values across datapruebas and neuropruebas experiments.

Usage:
    python -m src.visualization.explore_canvas_size
"""

from src.loader import load_datapruebas, load_neuropruebas


def collect_canvas_sizes(experiment):
    """Collect canvas_size values from all subjects in an experiment.
    
    Returns a dict mapping canvas_size -> list of subject_ids
    """
    canvas_sizes = {}
    
    for subject_id, subject in experiment.subjects.items():
        size = subject.canvas_size
        if size not in canvas_sizes:
            canvas_sizes[size] = []
        canvas_sizes[size].append(subject_id)
    
    return canvas_sizes


def main():
    results = {}
    
    # Datapruebas
    print("Loading datapruebas...")
    try:
        dp_experiment = load_datapruebas()
        dp_sizes = collect_canvas_sizes(dp_experiment)
        results["datapruebas"] = dp_sizes
        print(f"  ✓ {len(dp_experiment.subjects)} subjects loaded")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["datapruebas"] = {}
    
    # Neuropruebas
    print("\nLoading neuropruebas...")
    try:
        np_experiment = load_neuropruebas()
        np_sizes = collect_canvas_sizes(np_experiment)
        results["neuropruebas"] = np_sizes
        print(f"  ✓ {len(np_experiment.subjects)} subjects loaded")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["neuropruebas"] = {}
    
    # Summary
    print("\n" + "="*50)
    print("CANVAS_SIZE VALUES SUMMARY")
    print("="*50)
    
    all_sizes = set()
    
    for origin, sizes in results.items():
        print(f"\n{origin.upper()}:")
        if not sizes:
            print("  No data")
            continue
            
        for size, subjects in sorted(sizes.items(), key=lambda x: (x[0] is None, x[0])):
            all_sizes.add(size)
            print(f"  canvas_size={size}: {len(subjects)} subjects")
    
    print("\n" + "-"*50)
    unique_values = sorted(s for s in all_sizes if s is not None)
    print(f"UNIQUE VALUES ACROSS ALL DATA: {unique_values}")
    if None in all_sizes:
        print("⚠️  WARNING: Some subjects have canvas_size=None")


if __name__ == "__main__":
    main()
