"""
Shared loader utilities for TMT experiments.

Provides functions to load datapruebas and neuropruebas experiments
using their respective mappers.
"""

import os

from src import config
from src.mapper.datapruebas.datapruebas_mapper import DatapruebasTMTMapper
from src.mapper.neuropruebas.neuropruebas_mapper import NeuropruebasTMTMapper


def load_datapruebas():
    """Load the datapruebas experiment."""
    dataset_path = os.path.join(
        config.DATA_DIR,
        "raw/tmt/datapruebas/subjects",
        config.EXPERIMENT_FILE_NAME
    )
    mapper = DatapruebasTMTMapper()
    return mapper.map(dataset_path)


def load_neuropruebas():
    """Load the neuropruebas experiment."""
    dataset_path = os.path.join(
        config.DATA_DIR,
        "raw/tmt/neuropruebas/subjects"
    )
    mapper = NeuropruebasTMTMapper()
    return mapper.map(dataset_path)


def load_experiment(origin: str):
    """Load experiment by origin name.
    
    Args:
        origin: Either "datapruebas" or "neuropruebas"
        
    Returns:
        TMTExperiment object
        
    Raises:
        ValueError: If origin is not recognized
    """
    if origin == "datapruebas":
        return load_datapruebas()
    elif origin == "neuropruebas":
        return load_neuropruebas()
    else:
        raise ValueError(f"Unknown origin: {origin}")

