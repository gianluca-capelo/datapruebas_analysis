import logging
import os

from neurotask.tmt.mapper.mapper import TMTMapper
from neurotask.tmt.tmt_analyzer import TMTAnalyzer

from src import config
from src.config import DATA_DIR, EXPERIMENT_FILE_NAME
from src.mapper.datapruebas.datapruebas_mapper import DatapruebasTMTMapper
from src.mapper.neuropruebas.neuropruebas_mapper import NeuropruebasTMTMapper


def log_and_run_tmt_analysis(dataset_path, output_path, correct_targets_minimum, consecutive_points, cut_criteria,
                             calculate_crosses, mapper: TMTMapper):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    hand_analysis = TMTAnalyzer(
        mapper=mapper,
        dataset_path=dataset_path,
        output_path=output_path,
    )

    hand_analysis.run(correct_targets_minimum, consecutive_points, cut_criteria=cut_criteria,
                      calculate_crosses=calculate_crosses)

    return hand_analysis


def run_analysis_with_configuration_parameters(output_path, experiment_origin):
    threshold = config.CORRECT_THRESHOLD
    cut_criteria = config.CUT_CRITERIA
    points = config.CONSECUTIVE_POINTS

    if threshold is None and cut_criteria == "MINIMUM_TARGETS":
        raise ValueError("`correct_targets_minimum` must be set when `cut_criteria` is 'MINIMUM_TARGETS'.")

    raw_data_path = os.path.join(DATA_DIR, f"raw/tmt/{experiment_origin}")
    subjects_folder = os.path.join(raw_data_path, "subjects")

    # In 'datapruebas', all subjects must be in one file;
    # in 'neuropruebas', there is one file per subject within the folder.
    if experiment_origin == 'datapruebas':
        dataset_dir = os.path.join(subjects_folder, EXPERIMENT_FILE_NAME)
    else:
        dataset_dir = subjects_folder

    analysis = log_and_run_tmt_analysis(
        dataset_path=dataset_dir,
        output_path=os.path.join(output_path, experiment_origin),
        correct_targets_minimum=threshold,
        consecutive_points=points,
        cut_criteria=cut_criteria,
        calculate_crosses=config.CALCULATE_CROSSES,
        mapper=DatapruebasTMTMapper() if experiment_origin == 'datapruebas' else NeuropruebasTMTMapper()
    )

    df_metrics = analysis.get_metrics_dataframe()
    print("Metrics DataFrame:")
    print(df_metrics.head())

    experiment = analysis.experiment
    print(f"Len of subjects: {len(experiment.subjects)}")

    return analysis


if __name__ == "__main__":
    run_analysis_with_configuration_parameters(
        '/home/gianluca/Research/datapruebas_analysis/results',
        'datapruebas'
    )
