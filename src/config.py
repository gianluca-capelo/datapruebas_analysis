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

# Screen parameters
RADIUS_HEIGHT = 0.0275

# TMT parameters
CONSECUTIVE_POINTS = 5
CORRECT_THRESHOLD = 10
CUT_CRITERIA = "MINIMUM_TARGETS"
CALCULATE_CROSSES = False
