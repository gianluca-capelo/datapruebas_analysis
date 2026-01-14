# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neuropsychological data analysis pipeline for processing and analyzing cognitive task data from multiple experiments. The project processes raw experimental data, computes metrics, and runs machine learning models to predict cognitive variables.

## Experiments Analyzed

| Task | Full Name | Description | Key Metrics |
|------|-----------|-------------|-------------|
| **TMT** | Trail Making Test | Processing speed and cognitive flexibility | Execution time, errors, trajectory metrics |
| **SST** | Stop Signal Task | Inhibitory control and reaction time | SSRT, SSD, Go RT |
| **CDT** | Change Detection Task | Visual working memory | Cowan's K (K_4, K_6), accuracy |
| **Go/No-Go** | Go/No-Go Task | Simple inhibitory control | Hit Rate, False Alarm, c, sensitivity (d') |

## Running Analysis

### Quick Start - Run All Analysis

```bash
# Activate virtual environment first
source venv/bin/activate

# Run all analyses sequentially (TMT, SST, CDT, Go/No-Go)
python -m src.runner.run_all_analysis
```

### Individual Analysis Commands

```bash
# TMT (Trail Making Test)
python -m src.loader.analysis_loader

# SST (Stop Signal Task)
python -m src.loader.sst_analysis_loader

# CDT (Change Detection Task)
python -m src.loader.cdt_analysis_loader

# Go/No-Go
python -m src.loader.gonogo_analysis_loader
```

### Machine Learning Pipeline

```bash
# Regression task (predict continuous variables like SSRT)
python -m src.model.run_models --task regression

# Classification task (predict categories)
python -m src.model.run_models --task classification
```

### Testing

```bash
# Run specific test file
python -m pytest src/model/datasetbuilder/test_dataset_builder.py

# Note: Test coverage is limited - primarily permutation tests and dataset builder
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

### Module Structure

**src/loader/**: Data loading and preprocessing
- `analysis_loader.py` - TMT analysis loader
- `sst_loader.py`, `sst_analysis_loader.py` - SST data and analysis
- `cdt_loader.py`, `cdt_analysis_loader.py` - CDT data and analysis
- `gonogo_loader.py`, `gonogo_analysis_loader.py` - Go/No-Go data and analysis
- `metadata/` - Subject metadata handling for both datapruebas and neuropruebas

**src/analysis/**: Core analysis algorithms
- `stop_signal_task.py` - SST metrics computation (SSRT, SSD, etc.)
- `change_detection_task.py` - CDT metrics computation (Cowan's K)
- `go_no_go_task.py` - Go/No-Go metrics computation (d', c, hit rate)

**src/model/**: Machine learning pipeline
- `run_models.py` - Main ML pipeline with LOOCV
- `datasetbuilder/dataset_builder.py` - Constructs X, y from multiple tasks
- `reg.py` - Regression utilities
- `permutation_tests.py` - Statistical permutation testing
- `shap/` - SHAP analysis (legacy, may need updates)

**src/mapper/**: Data format mappers
- `datapruebas/` - Mapper for datapruebas format (JSON-based)
- `neuropruebas/` - Mapper for neuropruebas format (CSV-based)

**src/runner/**: Execution entry points
- `run_all_analysis.py` - Sequential execution of all analyses
- `run_hand_analysis.py` - TMT-specific analysis runner

**src/visualization/**: Plotting utilities for results

**src/data_analysis/**: Ad-hoc data exploration scripts

### Configuration

All configuration is centralized in `src/config.py`:

- **Paths**: Data directories, analysis output folders
- **TMT Parameters**: `CONSECUTIVE_POINTS`, `CORRECT_THRESHOLD`, `CUT_CRITERIA`, etc.
- **ML Settings**:
  - Seeds: `MODEL_OUTER_SEED=47`, `MODEL_INNER_SEED=66`
  - Cross-validation: `INNER_CV_SPLITS=10`
  - Feature selection: `PERFORM_FEATURE_SELECTION=True`, `MAX_SELECTED_FEATURES=20`
  - Hyperparameter tuning: `TUNE_HYPERPARAMETERS=True`
- **Models**: Pre-configured scikit-learn and XGBoost models with hyperparameter grids
- **Datasets**: List of dataset configurations (e.g., `['tmt_ssrt']`)

### Data Sources

The project processes data from two origins:
- **datapruebas**: JSON format (e.g., `data/raw/tmt/datapruebas/subjects/*.json`)
- **neuropruebas**: CSV format (e.g., `data/raw/tmt/neuropruebas/subjects/*.csv`)

Both are merged after processing using their respective mappers.

## ML Pipeline Details

### Available Datasets

Defined in `src/model/datasetbuilder/dataset_builder.py`:

| Dataset | Features | Target | Description |
|---------|----------|--------|-------------|
| `tmt_ssrt` | TMT (~136 features) | `ssrt` | TMT → Stop Signal Reaction Time |
| `tmt_k` | TMT (~136 features) | `K` | TMT → CDT Capacity |
| `tmt_dprime` | TMT (~136 features) | `dprimer/sensibilidad` | TMT → Go/No-Go sensitivity |

### Dataset Construction Process

1. **Load TMT**: Read latest `analysis.csv` from `data/hand_analysis/`
2. **Filter**: Only `is_valid == True` trials
3. **Feature Detection**: Automatically detect numeric columns (excluding metadata)
4. **Aggregate**: Mean per subject, pivoted by `trial_type` (PART_A, PART_B)
5. **Load Target**: Read target metric from respective task analysis
6. **Merge**: Join by `subject_id`

Result: ~170 subjects × ~136 features

### Cross-Validation Strategy

- **Outer CV**: Leave-One-Out (LOOCV) - each subject used once as test
- **Inner CV**: 10-fold for hyperparameter tuning (if `TUNE_HYPERPARAMETERS=True`)

### Pipeline Components

```
SimpleImputer → StandardScaler → [SelectKBest] → Model
```

Feature selection (SelectKBest) is optional based on `PERFORM_FEATURE_SELECTION`.

### Output Structure

```
results/
├── regression/<timestamp>/<target>/<dataset>/
│   ├── config.json       # Experiment configuration
│   ├── folds.csv         # Per-fold predictions (LOOCV)
│   └── summary.csv       # Aggregated metrics per model
└── classification/<timestamp>/<target>/<dataset>/
    └── (same structure)
```

## Analysis Output Structure

Each analysis creates timestamped folders:

```
data/
├── hand_analysis/<timestamp>/          # TMT
│   ├── analysis.csv                    # Metrics per subject
│   └── configuration.json              # Analysis metadata
├── sst_analysis/<timestamp>/           # SST
│   ├── sst_analysis.csv
│   └── configuration.json
├── cdt_analysis/<timestamp>/           # CDT
│   ├── cdt_analysis.csv
│   └── configuration.json
└── gonogo_analysis/<timestamp>/        # Go/No-Go
    ├── gonogo_analysis.csv
    └── configuration.json
```

### Loading Latest Analysis Programmatically

```python
# TMT
from src.loader.analysis_loader import load_analysis
df, path = load_analysis(random_state=78, eval_size=None, split=False, old_split_config_date=None)

# SST
from src.loader import get_latest_sst_analysis
df, config = get_latest_sst_analysis()

# CDT
from src.loader import get_latest_cdt_analysis
df, config = get_latest_cdt_analysis()

# Go/No-Go
from src.loader import get_latest_gonogo_analysis
df, config = get_latest_gonogo_analysis()
```

## Adding New Datasets

To add a new ML dataset (e.g., predicting a new target variable):

1. **Edit `src/model/datasetbuilder/dataset_builder.py`**:
   ```python
   def get_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray, list, str]:
       match name:
           case 'new_dataset':
               return self._build_new()
           # ... existing cases

   def _build_new(self) -> Tuple[np.ndarray, np.ndarray, list, str]:
       # Load data, process, return (X, y, feature_names, target_name)
       ...
   ```

2. **Add to `src/config.py`**:
   ```python
   DATASETS = [
       'tmt_ssrt',
       'new_dataset',  # Add here
   ]
   ```

## Important Implementation Details

### TMT Analysis Parameters

- `CONSECUTIVE_POINTS=5`: Number of consecutive points for trajectory smoothing
- `CORRECT_THRESHOLD=10`: Threshold for correct target selection (pixels)
- `CUT_CRITERIA="MINIMUM_TARGETS"`: How to handle incomplete trials
- `CALCULATE_CROSSES=False`: Whether to compute trajectory crossings
- `INTERPOLATE_TRAJECTORY=False`: Whether to interpolate missing trajectory points

### SST Quality Thresholds

Defined in `src/analysis/stop_signal_task.py`:
- `MIN_RT_THRESHOLD=150`: Minimum reaction time (ms)
- `MIN_GO_ACCURACY=0.60`: Minimum Go trial accuracy (60%)
- `MIN_PRESP=0.10`: Minimum P(respond|stop) (10%)
- `MAX_PRESP=0.90`: Maximum P(respond|stop) (90%)

### Feature Selection

The `DatasetBuilder.EXCLUDE_COLS` set defines metadata columns excluded from ML features:
- Subject identifiers: `subject_id`, `trial_id`, `mail`
- Trial metadata: `trial_type`, `is_valid`, `invalid_cause`
- Timestamps: `recorded_at`, `start_date`
- Demographic/config: `age`, `gender`, `education_level`, `device`, etc.

## Notebooks

Exploratory analysis notebooks in `notebooks/`:
- `tmt_analysis.ipynb` - TMT exploration
- `sst_analysis.ipynb` - SST exploration
- `cdt_analysis.ipynb` - CDT exploration
- `gonogo_analysis.ipynb` - Go/No-Go exploration
- `ml_analysis.ipynb` - ML results analysis
- `metadata_eda.ipynb` - Subject metadata exploration

## Dependencies

Core dependencies (see `requirements.txt`):
- `pandas==2.0.3` - Data manipulation
- `numpy==1.24.4` - Numerical computing
- `scikit-learn==1.7.1` - Machine learning
- `xgboost==3.0.4` - Gradient boosting
- `matplotlib==3.9.2`, `seaborn==0.13.2` - Visualization
- `shap==0.48.0` - Model interpretability
- `pyxations` - Custom utilities
- `pydantic` - Data validation
- `neurotask==0.0.0` - Not available in pip (installed separately)

## Git Workflow

- Main branch: `main`
- Current branch: `plot_seg`
- Use descriptive commit messages following existing style
- Include co-authorship when appropriate
