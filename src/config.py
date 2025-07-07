import os

RANDOM_STATE = 78
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

DATASET_DIR = os.path.join(BASE_DIR, "data")


HAND_ANALYSIS_FOLDER = os.path.join(DATA_DIR, "hand_analysis")

ANALYSIS_PATH = os.path.join(DATA_DIR, "hand_analysis", "analysis.csv")
METADATA_CSV = os.path.join(DATA_DIR, "metadata", "metadata.csv")

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
CORRECT_THRESHOLD = 8
CONSECUTIVE_POINTS = 5
CUT_CRITERIA = "MINIMUM_TARGETS"
CALCULATE_CROSSES = False
