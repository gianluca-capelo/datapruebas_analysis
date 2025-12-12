"""
Quick test to verify px2mm loading from datapruebas virtual-chinrest trials.

Usage:
    python test_chinrest_loading.py
"""
import os
from src import config
from src.mapper.datapruebas.datapruebas_mapper import DatapruebasTMTMapper


def load_datapruebas():
    """Load the datapruebas experiment."""
    dataset_path = os.path.join(
        config.DATA_DIR,
        "raw/tmt/datapruebas/subjects",
        config.EXPERIMENT_FILE_NAME
    )
    mapper = DatapruebasTMTMapper()
    return mapper.map(dataset_path)


def find_px2mm_values(experiment):
    """Find all px2mm values in the experiment data.
    
    Args:
        experiment: TMTExperiment object
    
    Returns:
        List of dicts with px2mm details
    """
    entries = []
    
    for subject_id, subject in experiment.subjects.items():
        session_data = subject.session_data
        if session_data and session_data.get('px2mm') is not None:
            entries.append({
                'subject_id': subject_id,
                'px2mm': session_data.get('px2mm')
            })
    
    return entries


def main():
    print("Loading datapruebas...")
    try:
        experiment = load_datapruebas()
        print(f"  ✓ Loaded {len(experiment.subjects)} subjects")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return
    
    px2mm_entries = find_px2mm_values(experiment)
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Total subjects: {len(experiment.subjects)}")
    print(f"Subjects with px2mm: {len(px2mm_entries)}")
    
    if px2mm_entries:
        print("\nSample px2mm values:")
        for item in px2mm_entries[:10]:
            print(f"  Subject: {item['subject_id'][:8]}... | px2mm: {item['px2mm']}")
    else:
        print("\n⚠️  No px2mm values found in the data.")


if __name__ == "__main__":
    main()
