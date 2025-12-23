"""
Quick test to verify px2mm and scale_factor loading from virtual-chinrest trials.

Usage:
    python test_chinrest_loading.py
"""
import csv
import os
from src import config
from src.loader import load_datapruebas, load_neuropruebas


def find_chinrest_values(experiment, origin):
    """Find all px2mm and scale_factor values in the experiment data.
    
    Args:
        experiment: TMTExperiment object
        origin: String identifying data source ("datapruebas" or "neuropruebas")
    
    Returns:
        List of dicts with chinrest details (px2mm and scale_factor)
    """
    entries = []
    
    for subject_id, subject in experiment.subjects.items():
        session_data = subject.session_data
        if session_data:
            px2mm = session_data.get('px2mm')
            scale_factor = session_data.get('scale_factor')
            if px2mm is not None or scale_factor is not None:
                entries.append({
                    'subject_id': subject_id,
                    'origin': origin,
                    'px2mm': px2mm,
                    'scale_factor': scale_factor
                })
    
    return entries


def print_summary(name, experiment, chinrest_entries):
    print(f"\n{'='*50}")
    print(f"{name.upper()} SUMMARY")
    print(f"{'='*50}")
    print(f"Total subjects: {len(experiment.subjects)}")
    
    px2mm_count = sum(1 for e in chinrest_entries if e['px2mm'] is not None)
    scale_factor_count = sum(1 for e in chinrest_entries if e['scale_factor'] is not None)
    
    print(f"Subjects with px2mm: {px2mm_count}")
    print(f"Subjects with scale_factor: {scale_factor_count}")
    
    if chinrest_entries:
        print("\nSample chinrest values:")
        for item in chinrest_entries[:10]:
            print(f"  Subject: {item['subject_id'][:8]}... | px2mm: {item['px2mm']} | scale_factor: {item['scale_factor']}")
    else:
        print("\n⚠️  No chinrest values found in the data.")


def main():
    all_entries = []
    
    # Datapruebas
    print("Loading datapruebas...")
    try:
        dp_experiment = load_datapruebas()
        print(f"  ✓ Loaded {len(dp_experiment.subjects)} subjects")
        dp_chinrest = find_chinrest_values(dp_experiment, "datapruebas")
        all_entries.extend(dp_chinrest)
        print_summary("datapruebas", dp_experiment, dp_chinrest)
    except Exception as e:
        print(f"  ✗ Error loading datapruebas: {e}")

    # Neuropruebas
    print("\nLoading neuropruebas...")
    try:
        np_experiment = load_neuropruebas()
        print(f"  ✓ Loaded {len(np_experiment.subjects)} subjects")
        np_chinrest = find_chinrest_values(np_experiment, "neuropruebas")
        all_entries.extend(np_chinrest)
        print_summary("neuropruebas", np_experiment, np_chinrest)
    except Exception as e:
        print(f"  ✗ Error loading neuropruebas: {e}")

    # Save to CSV
    if all_entries:
        output_path = os.path.join(config.DATA_DIR, "chinrest_by_subject.csv")
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['subject_id', 'origin', 'px2mm', 'scale_factor'])
            writer.writeheader()
            writer.writerows(all_entries)
        print(f"\n{'='*50}")
        print(f"CSV saved to: {output_path}")
        print(f"Total entries: {len(all_entries)}")


if __name__ == "__main__":
    main()
