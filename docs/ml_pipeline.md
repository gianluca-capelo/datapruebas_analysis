# Machine Learning Pipeline

**Date:** December 19, 2025

This document explains how to run the ML pipeline to predict cognitive variables using digital task data.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Running the Pipeline](#running-the-pipeline)
3. [Available Datasets](#available-datasets)
4. [Results Structure](#results-structure)
5. [Configuration](#configuration)
6. [Available Models](#available-models)
7. [Adding New Datasets](#adding-new-datasets)

---

## Prerequisites

1. **Generated analysis data:**
   - TMT: An analysis must exist in `data/hand_analysis/` with `analysis.csv` and `configuration.json`
   - SST: An analysis must exist in `data/sst_analysis/` with `analysis.csv`

2. **Installed dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Pipeline

### Regression (predict continuous variable)

```bash
python -m src.model.run_models --task regression
```

### Classification (predict category)

```bash
python -m src.model.run_models --task classification
```

> **Note:** The target is defined in each dataset, not passed as a parameter. See section [Available Datasets](#available-datasets).

---

## Available Datasets

Each dataset defines its own features and target:

| Dataset | Features | Target | Description |
|---------|----------|--------|-------------|
| `tmt_ssrt` | TMT (136 features) | `ssrt` | Trail Making Test features to predict Stop Signal Reaction Time |

### How `tmt_ssrt` is built

1. **Load TMT:** Reads `analysis.csv` from the latest run in `data/hand_analysis/`
2. **Filter:** Only trials with `is_valid == True`
3. **Detect features:** Numeric columns automatically (excluding metadata)
4. **Aggregate:** Mean per subject, pivoted by `trial_type` (PART_A, PART_B)
5. **Load SST:** Reads SSRT from `data/sst_analysis/`
6. **Merge:** Join by `subject_id`

Result: ~170 subjects × 136 features

---

## Results Structure

```
results/
├── classification/
│   └── {timestamp}/
│       └── {target}/
│           └── {dataset}/
│               ├── config.json    # Experiment configuration
│               ├── folds.csv      # Predictions per fold (LOOCV)
│               └── summary.csv    # Metrics per model
│
└── regression/
    └── {timestamp}/
        └── {target}/
            └── {dataset}/
                ├── config.json
                ├── folds.csv
                └── summary.csv
```

### Output Files

#### `config.json`
Experiment configuration:
```json
{
    "dataset": "tmt_ssrt",
    "feature_selection": true,
    "tune_hyperparameters": false,
    "is_classification": false,
    "timestamp": "2025-12-19_1200",
    "n_folds": 170,
    "feature_names": ["rt_PART_A", "rt_PART_B", ...]
}
```

#### `summary.csv`
Aggregated metrics per model:

| Column | Description |
|--------|-------------|
| `model` | Model name |
| `r2` | R² score (regression) |
| `mse` | Mean Squared Error |
| `mae` | Mean Absolute Error |
| `accuracy` | Accuracy (classification) |
| `roc_auc` | ROC AUC (classification) |
| `selected_features` | Selected features |

#### `folds.csv`
Detailed predictions per fold (Leave-One-Out):

| Column | Description |
|--------|-------------|
| `fold` | Fold index |
| `model` | Model name |
| `y_true` | Actual value |
| `y_pred` | Prediction |
| `best_params` | Optimal hyperparameters |

---

## Configuration

Edit `src/config.py`:

```python
# Seeds for reproducibility
MODEL_OUTER_SEED = 47
MODEL_INNER_SEED = 66
INNER_CV_SPLITS = 10

# Feature selection
PERFORM_FEATURE_SELECTION = True
MAX_SELECTED_FEATURES = 20

# Hyperparameter tuning
TUNE_HYPERPARAMETERS = True

# Datasets to evaluate (each defines its own target)
DATASETS = ['tmt_ssrt']
```

---

## Available Models

### Regression

| Model | Description |
|-------|-------------|
| `DummyRegressor` | Baseline (predicts the mean) |
| `LinearRegression` | Linear regression |
| `Ridge` | Regression with L2 regularization |
| `Lasso` | Regression with L1 regularization |
| `ElasticNet` | L1 + L2 combination |
| `SVR` | Support Vector Regression |
| `RandomForestRegressor` | Random Forest |
| `XGBRegressor` | XGBoost |

### Classification

| Model | Description |
|-------|-------------|
| `RandomForestClassifier` | Random Forest |
| `SVC` | Support Vector Classification |
| `LogisticRegression` | Logistic regression |
| `XGBClassifier` | XGBoost |

---

## Adding New Datasets

Each dataset encapsulates its features and target. To add a new one:

### 1. Edit `src/model/datasetbuilder/dataset_builder.py`

```python
def get_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray, list, str]:
    """
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        target_name: Name of target column
    """
    match name:
        case 'tmt_ssrt':
            return self._build_tmt_ssrt()
        case 'new_dataset':           # Add new case
            return self._build_new()
        case _:
            raise ValueError(f"Unknown dataset: {name}")

def _build_new(self) -> Tuple[np.ndarray, np.ndarray, list, str]:
    target_name = 'my_target'
    
    # Load data...
    # Process...
    
    X = ...  # np.ndarray (n_samples, n_features)
    y = ...  # np.ndarray (n_samples,)
    feature_names = [...]  # list[str]
    
    return X, y, feature_names, target_name
```

### 2. Add to `src/config.py`

```python
DATASETS = [
    'tmt_ssrt',
    'new_dataset',  # Add here
]
```

---

## Methodology

### Cross-Validation
- **Leave-One-Out (LOOCV):** Each subject is used once as test
- **Inner CV:** 10-fold for hyperparameter tuning (if enabled)

### Feature Selection
- **Method:** SelectKBest with f_regression or f_classif
- **k:** Maximum 20 features (configurable)

### Pipeline
```
Imputer → StandardScaler → [FeatureSelection] → Model
```

---

## Complete Example

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Verify data
ls data/hand_analysis/  # Should have folders with analysis.csv
ls data/sst_analysis/   # Should have folders with analysis.csv

# 3. Run regression
python -m src.model.run_models --task regression

# 4. View results
cat results/regression/*/ssrt/tmt_ssrt/summary.csv
```

---

## Architecture

```
src/model/
├── run_models.py              # Main pipeline
├── datasetbuilder/
│   └── dataset_builder.py     # Dataset construction (X, y, features, target)
├── shap/                      # SHAP analysis (legacy, requires update)
├── classification/
│   └── roc_curves.py          # ROC curves
└── permutation_tests.py       # Permutation tests
```

### Data Flow

```
DatasetBuilder.get_dataset('tmt_ssrt')
       │
       ▼
  (X, y, feature_names, target_name)
       │
       ▼
  perform() → LOOCV + GridSearchCV
       │
       ▼
  save_results() → results/{task}/{timestamp}/{target}/{dataset}/
```
