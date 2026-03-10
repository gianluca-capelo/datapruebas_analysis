"""Generate SHAP summary figures for multiple dataset/model combinations.

Usage:
    python -m analysis.scripts.generate_shap_figures
"""
import os
from pathlib import Path

from tqdm import tqdm

from src.config import REGRESSION_RESULTS_DIR, CLASSIFICATION_RESULTS_DIR, BASE_DIR
from src.model.shap.analyze_shap_results import run_analysis
from src.model.run_models import retrieve_dataset

ANALYSIS_FIGURES_DIR = os.path.join(BASE_DIR, "analysis", "figures")

COMBINATIONS = [
    {"dataset": "tmt_age",        "model": "SVR",                   "task": "regression", "timestamp": "2026-03-07_1213"},
    {"dataset": "tmt_k_mean",     "model": "XGBRegressor",          "task": "regression", "timestamp": "2026-03-06_2028"},
    {"dataset": "tmt_accuracy",   "model": "SVR",                   "task": "regression", "timestamp": "2026-03-07_1213"},
    {"dataset": "tmt_c",          "model": "Lasso",                 "task": "regression", "timestamp": "2026-03-07_1213"},
]


def _resolve_folds_path(dataset, task, timestamp):
    """Resolve the folds.csv path for a given combination and verify it exists."""
    results_dir = CLASSIFICATION_RESULTS_DIR if task == "classification" else REGRESSION_RESULTS_DIR
    _, _, _, target_name = retrieve_dataset(dataset)
    folds_path = Path(results_dir) / timestamp / target_name / dataset / "folds.csv"
    if not folds_path.exists():
        raise FileNotFoundError(f"folds.csv not found: {folds_path}")
    return folds_path


def main():
    os.makedirs(ANALYSIS_FIGURES_DIR, exist_ok=True)

    # Validate all combinations before starting (fail fast)
    for combo in COMBINATIONS:
        _resolve_folds_path(combo["dataset"], combo["task"], combo["timestamp"])

    succeeded = []
    failed = []

    for combo in tqdm(COMBINATIONS, desc="Generating SHAP figures"):
        dataset = combo["dataset"]
        model = combo["model"]
        task = combo["task"]
        timestamp = combo["timestamp"]
        is_classification = task == "classification"
        save_filename = f"shap_summary_{dataset}_{model}.png"

        tqdm.write(f"\n--- {dataset} / {model} ({task}, {timestamp}) ---")
        try:
            run_analysis(
                dataset_name=dataset,
                is_classification=is_classification,
                model=model,
                timestamp=timestamp,
                save_filename=save_filename,
                figures_dir=ANALYSIS_FIGURES_DIR,
            )
            succeeded.append(f"{dataset}/{model}")
        except Exception as e:
            tqdm.write(f"FAILED: {e}")
            failed.append(f"{dataset}/{model}: {e}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Done. {len(succeeded)}/{len(COMBINATIONS)} figures generated.")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
