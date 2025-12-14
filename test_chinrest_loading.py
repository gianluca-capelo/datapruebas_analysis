"""
Quick test to verify px2mm loading from virtual-chinrest trials.

Usage:
    python test_chinrest_loading.py
"""
import csv
import os
from src import config
from src.loader import load_datapruebas, load_neuropruebas


def find_px2mm_values(experiment, origin):
    """Find all px2mm values in the experiment data.
    
    Args:
        experiment: TMTExperiment object
        origin: String identifying data source ("datapruebas" or "neuropruebas")
    
    Returns:
        List of dicts with px2mm details
    """
    entries = []
    
    for subject_id, subject in experiment.subjects.items():
        session_data = subject.session_data
        if session_data and session_data.get('px2mm') is not None:
            entries.append({
                'subject_id': subject_id,
                'origin': origin,
                'px2mm': session_data.get('px2mm')
            })
    
    return entries


def print_summary(name, experiment, px2mm_entries):
    print(f"\n{'='*50}")
    print(f"{name.upper()} SUMMARY")
    print(f"{'='*50}")
    print(f"Total subjects: {len(experiment.subjects)}")
    print(f"Subjects with px2mm: {len(px2mm_entries)}")
    
    if px2mm_entries:
        print("\nSample px2mm values:")
        for item in px2mm_entries[:10]:
            print(f"  Subject: {item['subject_id'][:8]}... | px2mm: {item['px2mm']}")
    else:
        print("\n⚠️  No px2mm values found in the data.")


def main():
    all_entries = []
    
    # Datapruebas
    print("Loading datapruebas...")
    try:
        dp_experiment = load_datapruebas()
        print(f"  ✓ Loaded {len(dp_experiment.subjects)} subjects")
        dp_px2mm = find_px2mm_values(dp_experiment, "datapruebas")
        all_entries.extend(dp_px2mm)
        print_summary("datapruebas", dp_experiment, dp_px2mm)
    except Exception as e:
        print(f"  ✗ Error loading datapruebas: {e}")

    # Neuropruebas
    print("\nLoading neuropruebas...")
    try:
        np_experiment = load_neuropruebas()
        print(f"  ✓ Loaded {len(np_experiment.subjects)} subjects")
        np_px2mm = find_px2mm_values(np_experiment, "neuropruebas")
        all_entries.extend(np_px2mm)
        print_summary("neuropruebas", np_experiment, np_px2mm)
    except Exception as e:
        print(f"  ✗ Error loading neuropruebas: {e}")

    # Save to CSV
    if all_entries:
        output_path = os.path.join(config.DATA_DIR, "px2mm_by_subject.csv")
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['subject_id', 'origin', 'px2mm'])
            writer.writeheader()
            writer.writerows(all_entries)
        print(f"\n{'='*50}")
        print(f"CSV saved to: {output_path}")
        print(f"Total entries: {len(all_entries)}")


if __name__ == "__main__":
    main()
