"""Export the per-model subset sizes (N) to a CSV.

Each online regression model is trained on the intersection of participants
who have valid cTMT features AND the target task (cTMT ∩ task), after QC.
Those intersection sizes are not reported in the thesis/paper (only per-task N
are). This script computes them via DatasetBuilder (the authoritative source of
the inner-merge + QC) and writes one row per subset/target.

Usage:
    python -m analysis.scripts.compute_model_subset_sizes
    python -m analysis.scripts.compute_model_subset_sizes --out path/to.csv
"""
import os
import argparse

import pandas as pd

from src.config import BASE_DIR
from src.model.datasetbuilder.dataset_builder import DatasetBuilder
from src.loader import (
    get_latest_sst_analysis,
    get_latest_cdt_analysis,
    get_latest_gonogo_analysis,
)

DEFAULT_OUT = os.path.join(BASE_DIR, "analysis", "results", "model_subset_sizes.csv")

# Online-study regression datasets (editable). loader=None means TMT-only (no
# external task intersection, e.g. age).
DATASETS = [
    {"dataset": "tmt_age",      "subset": "cTMT",       "loader": None},
    {"dataset": "tmt_ssrt",     "subset": "cTMT ∩ SST", "loader": get_latest_sst_analysis},
    {"dataset": "tmt_k_mean",   "subset": "cTMT ∩ CDT", "loader": get_latest_cdt_analysis},
    {"dataset": "tmt_accuracy", "subset": "cTMT ∩ GNG", "loader": get_latest_gonogo_analysis},
    {"dataset": "tmt_c",        "subset": "cTMT ∩ GNG", "loader": get_latest_gonogo_analysis},
]


def _task_n(loader, cache):
    """Unique-subject count for a task loader, memoized by loader identity."""
    if loader is None:
        return None
    if loader not in cache:
        df = loader()[0]
        cache[loader] = df["subject_id"].nunique()
    return cache[loader]


def main(out_path=DEFAULT_OUT):
    builder = DatasetBuilder()

    # cTMT subject set (valid trials aggregated to one row per subject)
    n_tmt = len(builder._load_tmt_aggregated())

    task_cache = {}
    rows = []
    for cfg in DATASETS:
        X, _y, _feats, target = builder.get_dataset(cfg["dataset"])
        n_model = X.shape[0]
        n_task = _task_n(cfg["loader"], task_cache)
        rows.append({
            "dataset": cfg["dataset"],
            "target": target,
            "subset": cfg["subset"],
            "n_tmt": n_tmt,
            "n_task": n_task if n_task is not None else n_tmt,
            "n_model": n_model,
        })

    df = pd.DataFrame(rows, columns=["dataset", "target", "subset", "n_tmt", "n_task", "n_model"])
    print(df.to_string(index=False))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export per-model subset sizes (intersection N) to CSV")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output CSV path (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    main(args.out)
