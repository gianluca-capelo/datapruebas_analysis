# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neuropsychological data analysis pipeline for processing cognitive task data from multiple experiments. Processes raw experimental data, computes metrics, and runs machine learning models to predict cognitive variables.

## Experiments Analyzed

| Task | Full Name | Description | Key Metrics |
|------|-----------|-------------|-------------|
| **TMT** | Trail Making Test | Processing speed and cognitive flexibility | Execution time, errors, trajectory metrics |
| **SST** | Stop Signal Task | Inhibitory control and reaction time | SSRT, SSD, Go RT |
| **CDT** | Change Detection Task | Visual working memory | Cowan's K (K_4, K_6), accuracy |
| **Go/No-Go** | Go/No-Go Task | Simple inhibitory control | Hit Rate, False Alarm, c, sensitivity (d') |

## Running Analysis

```bash
# Activate virtual environment first
source venv/bin/activate

# Run all analyses sequentially (TMT, SST, CDT, Go/No-Go)
python -m src.runner.run_all_analysis

# Individual analyses
python -m src.loader.analysis_loader        # TMT
python -m src.loader.sst_analysis_loader    # SST
python -m src.loader.cdt_analysis_loader    # CDT
python -m src.loader.gonogo_analysis_loader # Go/No-Go

# Machine Learning Pipeline
python -m src.model.run_models --task regression
python -m src.model.run_models --task classification
```

### Testing

```bash
python -m pytest src/model/datasetbuilder/test_dataset_builder.py
```

### Visualization CLI Tools

```bash
python -m src.visualization.plot_trials_cli          # Plot trial trajectories
python -m src.visualization.segmentation_plotting    # Trajectory segmentation viz
python -m src.visualization.compare_interpolation    # Compare interpolation methods
```

## Architecture

### Data Flow

```
Raw Data (data/raw/)
    ↓
Loaders (src/loader/)
    ↓
Analysis (src/analysis/)
    ↓
Processed Results (data/*_analysis/<timestamp>/)
    ↓
DatasetBuilder (src/model/datasetbuilder/)
    ↓
ML Pipeline (src/model/run_models.py)
    ↓
Results (results/{regression|classification}/<timestamp>/)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/loader/` | Data loading and preprocessing. Each task has `*_loader.py` (data) and `*_analysis_loader.py` (pipeline) |
| `src/analysis/` | Core analysis algorithms: `stop_signal_task.py`, `change_detection_task.py`, `go_no_go_task.py` |
| `src/model/` | ML pipeline: `run_models.py` (LOOCV), `datasetbuilder/` (X,y construction), `permutation_tests.py` |
| `src/mapper/` | Format converters: `datapruebas/` (JSON), `neuropruebas/` (CSV) |
| `src/runner/` | Entry points: `run_all_analysis.py`, `run_hand_analysis.py` |
| `src/visualization/` | Plotting utilities for trajectories, segmentation, analysis results |

### Configuration (`src/config.py`)

**TMT Parameters:**
- `CONSECUTIVE_POINTS=5`: Trajectory smoothing window
- `CORRECT_THRESHOLD=10`: Pixels for correct target selection
- `CUT_CRITERIA="MINIMUM_TARGETS"`: Incomplete trial handling
- `INTERPOLATE_TRAJECTORY=False`: Trajectory interpolation toggle

**ML Settings:**
- Seeds: `MODEL_OUTER_SEED=47`, `MODEL_INNER_SEED=66`
- `INNER_CV_SPLITS=10`, `MAX_SELECTED_FEATURES=20`
- `PERFORM_FEATURE_SELECTION=True`, `TUNE_HYPERPARAMETERS=True`
- `DATASETS=['tmt_ssrt']`: Active dataset configurations

### Data Sources

Two formats merged after processing:
- **datapruebas**: JSON format (`data/raw/*/datapruebas/subjects/*.json`)
- **neuropruebas**: CSV format (`data/raw/*/neuropruebas/subjects/*.csv`)

## ML Pipeline Details

### Available Datasets

Defined in `src/model/datasetbuilder/dataset_builder.py`:

| Dataset | Features | Target | Description |
|---------|----------|--------|-------------|
| `tmt_ssrt` | TMT (~136 features) | `ssrt` | TMT → Stop Signal Reaction Time |
| `tmt_k` | TMT (~136 features) | `K` | TMT → CDT Capacity |
| `tmt_dprime` | TMT (~136 features) | `dprimer/sensibilidad` | TMT → Go/No-Go sensitivity |

### Cross-Validation Strategy

- **Outer CV**: Leave-One-Out (LOOCV) - each subject used once as test
- **Inner CV**: 10-fold for hyperparameter tuning

### Pipeline: `SimpleImputer → StandardScaler → [SelectKBest] → Model`

### Output Structure

```
results/{regression|classification}/<timestamp>/<target>/<dataset>/
├── config.json       # Experiment configuration
├── folds.csv         # Per-fold predictions (LOOCV)
└── summary.csv       # Aggregated metrics per model
```

## Analysis Output Structure

```
data/
├── hand_analysis/<timestamp>/     # TMT: analysis.csv, configuration.json
├── sst_analysis/<timestamp>/      # SST: sst_analysis.csv, configuration.json
├── cdt_analysis/<timestamp>/      # CDT: cdt_analysis.csv, configuration.json
└── gonogo_analysis/<timestamp>/   # Go/No-Go: gonogo_analysis.csv, configuration.json
```

### Loading Latest Analysis Programmatically

```python
# TMT
from src.loader.analysis_loader import load_analysis
df, path = load_analysis(random_state=78, eval_size=None, split=False, old_split_config_date=None)

# SST, CDT, Go/No-Go
from src.loader import get_latest_sst_analysis, get_latest_cdt_analysis, get_latest_gonogo_analysis
df, config = get_latest_sst_analysis()
df, config = get_latest_cdt_analysis()
df, config = get_latest_gonogo_analysis()
```

## Adding New Datasets

1. **Edit `src/model/datasetbuilder/dataset_builder.py`**:
   ```python
   def get_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray, list, str]:
       match name:
           case 'new_dataset':
               return self._build_new()

   def _build_new(self) -> Tuple[np.ndarray, np.ndarray, list, str]:
       # Load data, process, return (X, y, feature_names, target_name)
       ...
   ```

2. **Add to `src/config.py`**:
   ```python
   DATASETS = ['tmt_ssrt', 'new_dataset']
   ```

## Quality Control Thresholds

**SST** (`src/analysis/stop_signal_task.py`):
- `MIN_RT_THRESHOLD=150` ms, `MIN_GO_ACCURACY=0.60`
- `MIN_PRESP=0.10`, `MAX_PRESP=0.90`

**CDT** (`src/analysis/change_detection_task.py`):
- `CDT_MIN_ACCURACY=0.60`, `MAX_OMISSION_RATE=0.20`

## Feature Exclusion

`DatasetBuilder.EXCLUDE_COLS` defines metadata columns excluded from ML features:
- Subject identifiers: `subject_id`, `trial_id`, `mail`
- Trial metadata: `trial_type`, `is_valid`, `invalid_cause`
- Timestamps: `recorded_at`, `start_date`
- Demographic/config: `age`, `gender`, `education_level`, `device`, etc.

## Dependencies

Core (see `requirements.txt`):
- `pandas==2.0.3`, `numpy==1.24.4`
- `scikit-learn==1.7.1`, `xgboost==3.0.4`
- `matplotlib==3.9.2`, `seaborn==0.13.2`, `shap==0.48.0`
- `neurotask==0.0.0` - Not in pip (installed separately)
