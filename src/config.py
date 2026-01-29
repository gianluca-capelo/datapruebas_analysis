import os

RANDOM_STATE = 78
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

LOG_DIR = os.path.join(BASE_DIR, "logs")

DATASET_DIR = os.path.join(BASE_DIR, "data")

EXPERIMENT_FILE_NAME = "datapruebas_12_9_2025.json"

HAND_ANALYSIS_FOLDER = os.path.join(DATA_DIR, "hand_analysis")

ANALYSIS_PATH = os.path.join(DATA_DIR, "hand_analysis", "analysis.csv")
METADATA_CSV = os.path.join(DATA_DIR, "metadata", "metadata.csv")

TMT_NEUROPRUEBAS_METRICS_PATH = os.path.join(DATA_DIR, "processed", "tmt", "neuropruebas", "metrics.csv")
TMT_DATAPRUEBAS_METRICS_PATH = os.path.join(DATA_DIR, "processed", "tmt", "datapruebas", "metrics.csv")
TMT_METRICS_PATH = os.path.join(DATA_DIR, "processed", "tmt", "metrics.csv")

# SST paths
SST_DATAPRUEBAS_PATH = os.path.join(DATA_DIR, "raw", "sst", "datapruebas")
SST_NEUROPRUEBAS_PATH = os.path.join(DATA_DIR, "raw", "sst", "neuropruebas")
SST_ANALYSIS_FOLDER = os.path.join(DATA_DIR, "sst_analysis")

# CDT paths
CDT_DATAPRUEBAS_PATH = os.path.join(DATA_DIR, "raw", "cdt", "datapruebas")
CDT_NEUROPRUEBAS_PATH = os.path.join(DATA_DIR, "raw", "cdt", "neuropruebas")
CDT_ANALYSIS_FOLDER = os.path.join(DATA_DIR, "cdt_analysis")

# Go/No-Go paths
GONOGO_DATAPRUEBAS_PATH = os.path.join(DATA_DIR, "raw", "gonogo", "datapruebas")
GONOGO_NEUROPRUEBAS_PATH = os.path.join(DATA_DIR, "raw", "gonogo", "neuropruebas")
GONOGO_ANALYSIS_FOLDER = os.path.join(DATA_DIR, "gonogo_analysis")

DATAPRUEBAS_METADATA_PATH = os.path.join(DATA_DIR, "raw", "tmt", "datapruebas", "metadata", "metadata.csv")
NEUROPRUEBAS_METADATA_PATH = os.path.join(DATA_DIR, "raw", "tmt", "neuropruebas", "metadata", "metadata.csv")
OLD_NEUROPRUEBAS_METADATA_PATH = os.path.join(DATA_DIR, "raw", "tmt", "neuropruebas", "metadata", "Sujetxs TMT Nacho - participantes_con_genero_inferido.csv")
TRAIN_SET_PATH = os.path.join(DATA_DIR, "hand_analysis", "train_set.csv")
EVAL_SET_PATH = os.path.join(DATA_DIR, "hand_analysis", "eval_set.csv")

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s'
}

FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# TMT parameters
CONSECUTIVE_POINTS = 5
CORRECT_THRESHOLD = 10
CUT_CRITERIA = "MINIMUM_TARGETS"
CALCULATE_CROSSES = True
CROSSES_TIME_THRESHOLD = 1000  # milliseconds
INTERPOLATE_TRAJECTORY = True
INTERPOLATION_TARGET_FREQ_HZ = 60
TARGET_RADIUS_MULTIPLIER = 1.15  # Multiplier for target radius (1.0 = no change, 1.1 = 10% increase)

# =============================================================================
# ML Pipeline Configuration
# =============================================================================

import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC

# Paths for ML
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CLASSIFICATION_RESULTS_DIR = os.path.join(RESULTS_DIR, "classification")
REGRESSION_RESULTS_DIR = os.path.join(RESULTS_DIR, "regression")

# Seeds and CV settings
MODEL_OUTER_SEED = 47
MODEL_INNER_SEED = 66
INNER_CV_SPLITS = 10

# Feature selection and tuning
TUNE_HYPERPARAMETERS = True
PERFORM_FEATURE_SELECTION = True
MAX_SELECTED_FEATURES = 20

# Datasets configuration
DATASETS = [
    'tmt_ssrt',    # TMT features → SSRT target (Stop Signal Task)
    'tmt_k',       # TMT features → K capacity target (Change Detection Task)
    'tmt_dprime',  # TMT features → d' sensitivity target (Go/No-Go Task)
    'tmt_k_v2'
]

# Target columns
CLASSIFICATION_TARGET = 'group'
REGRESSION_TARGETS = [
    "ssrt",  # Stop Signal Reaction Time
]


def CLASSIFICATION_MODELS(random_state):
    return [
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        SVC(random_state=random_state, probability=True),
        LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1),
        xgb.XGBClassifier(random_state=random_state, n_jobs=-1)
    ]


CLASSIFICATION_PARAM_GRID = {
    "RandomForestClassifier": {
        "classifier__n_estimators": [100, 200, 500],
        "classifier__max_depth": [None, 8, 16],
        "classifier__min_samples_leaf": [1, 2, 5, 10],
        "classifier__max_features": ["sqrt", "log2"],
    },
    "SVC": [
        {
            "classifier__kernel": ["linear"],
            "classifier__C": [0.1, 1, 10],
        },
        {
            "classifier__kernel": ["rbf"],
            "classifier__C": [0.1, 1, 10],
            "classifier__gamma": ["scale", "auto"],
        },
    ],
    "LogisticRegression": [
        {
            "classifier__penalty": ["l2"],
            "classifier__C": [0.1, 1, 10],
            "classifier__solver": ["lbfgs", "liblinear", "saga"],
        },
        {
            "classifier__penalty": ["l1"],
            "classifier__C": [0.1, 1, 10],
            "classifier__solver": ["liblinear", "saga"],
        },
        {
            "classifier__penalty": ["elasticnet"],
            "classifier__C": [0.1, 1, 10],
            "classifier__l1_ratio": [0.5],
            "classifier__solver": ["saga"],
        },
    ],
    "XGBClassifier": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 6],
        "classifier__learning_rate": [0.1, 0.3],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
    },
}


def REGRESSION_MODELS(random_state):
    return [
        DummyRegressor(),
        LinearRegression(n_jobs=-1),
        Ridge(random_state=random_state),
        Lasso(random_state=random_state),
        xgb.XGBRegressor(random_state=random_state, n_jobs=-1),
        ElasticNet(random_state=random_state),
        SVR(),
        RandomForestRegressor(random_state=random_state, n_jobs=-1),
    ]


REGRESSION_PARAM_GRID = {
    "RandomForestRegressor": {
        "regressor__n_estimators": [100, 200, 500],
        "regressor__max_depth": [None, 8, 16],
        "regressor__min_samples_leaf": [2, 5, 10],
        "regressor__max_features": ["sqrt", "log2"]
    },
    "SVR": [
        {
            "regressor__kernel": ["linear"],
            "regressor__C": [0.01, 0.1, 1, 10, 100, 1000],
            "regressor__epsilon": [0.01, 0.1, 0.5, 1.0]
        },
        {
            "regressor__kernel": ["rbf"],
            "regressor__C": [0.01, 0.1, 1, 10, 100, 1000],
            "regressor__epsilon": [0.01, 0.1, 0.5, 1.0],
            "regressor__gamma": ["scale", "auto"],
        },
    ],
    "LinearRegression": {},
    "Ridge": {
        "regressor__alpha": [0.0001, 0.001, 0.01, 0.1,
                            1.0, 5, 10.0, 100.0, 1000.0, 10000.0]
    },
    "Lasso": {
        "regressor__alpha": [0.0001, 0.001, 0.01, 0.1,
                            1.0, 5, 10.0]
    },
    "XGBRegressor": {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [3, 6],
        "regressor__learning_rate": [0.1, 0.3],
        "regressor__subsample": [0.8, 1.0],
        "regressor__colsample_bytree": [0.8, 1.0],
    },
    "ElasticNet": {
        "regressor__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        "regressor__l1_ratio": [0.05, 0.2, 0.5, 0.8, 0.95]
    }
}
