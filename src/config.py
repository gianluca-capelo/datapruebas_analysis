import os

RANDOM_STATE = 78
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

DATASET_DIR = os.path.join(BASE_DIR, "data")

EXPERIMENT_FILE_NAME = "datapruebas_12_9_2025.json"

HAND_ANALYSIS_FOLDER = os.path.join(DATA_DIR, "hand_analysis")

ANALYSIS_PATH = os.path.join(DATA_DIR, "hand_analysis", "analysis.csv")
METADATA_CSV = os.path.join(DATA_DIR, "metadata", "metadata.csv")

TMT_NEUROPRUEBAS_METRICS_PATH = os.path.join(DATA_DIR, "processed", "tmt", "neuropruebas", "metrics.csv")
TMT_DATAPRUEBAS_METRICS_PATH = os.path.join(DATA_DIR, "processed", "tmt", "datapruebas", "metrics.csv")
TMT_METRICS_PATH = os.path.join(DATA_DIR, "processed", "tmt", "metrics.csv")

TRAIN_SET_PATH = os.path.join(DATA_DIR, "hand_analysis", "train_set.csv")
EVAL_SET_PATH = os.path.join(DATA_DIR, "hand_analysis", "eval_set.csv")

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s'
}

FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# Screen parameters
RADIUS_HEIGHT = 0.0275

# TMT parameters
CONSECUTIVE_POINTS = 5
CORRECT_THRESHOLD = None  # 8
CUT_CRITERIA = None  # "MINIMUM_TARGETS"
CALCULATE_CROSSES = False
