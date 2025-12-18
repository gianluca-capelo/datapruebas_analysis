"""Loader utilities for neuropsychological experiments."""

from src.loader.experiment_loader import (
    load_datapruebas,
    load_experiment,
    load_neuropruebas,
)
from src.loader.sst_loader import (
    load_all_sst_experiments,
    load_sst_experiment,
    load_sst_subjects,
)
from src.loader.sst_analysis_loader import (
    load_sst_analysis,
    compute_sst_metrics,
    get_latest_sst_analysis,
)
from src.loader.cdt_loader import (
    load_all_cdt_experiments,
    load_cdt_experiment,
    load_cdt_subjects,
)
from src.loader.cdt_analysis_loader import (
    load_cdt_analysis,
    compute_cdt_metrics,
    get_latest_cdt_analysis,
)

__all__ = [
    # TMT loaders
    "load_datapruebas",
    "load_experiment",
    "load_neuropruebas",
    # SST loaders
    "load_sst_subjects",
    "load_sst_experiment",
    "load_all_sst_experiments",
    # SST analysis
    "load_sst_analysis",
    "compute_sst_metrics",
    "get_latest_sst_analysis",
    # CDT loaders
    "load_cdt_subjects",
    "load_cdt_experiment",
    "load_all_cdt_experiments",
    # CDT analysis
    "load_cdt_analysis",
    "compute_cdt_metrics",
    "get_latest_cdt_analysis",
]

