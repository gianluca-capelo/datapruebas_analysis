import logging
import os

from neurotask.tmt.tmt_analyzer import TMTAnalyzer

from src import config
from src.config import DATA_DIR
from src.mapper.datapruebas.datapruebas_mapper import DatapruebasTMTMapper


def log_and_run_tmt_analysis(dataset_path, output_path, correct_targets_minimum, consecutive_points, cut_criteria,
                             calculate_crosses):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    hand_analysis = TMTAnalyzer(
        mapper=DatapruebasTMTMapper(),
        dataset_path=dataset_path,
        output_path=output_path
    )

    hand_analysis.run(correct_targets_minimum, consecutive_points, cut_criteria=cut_criteria,
                      calculate_crosses=calculate_crosses)

    return hand_analysis


def run_analysis_with_configuration_parameters(output_path):
    threshold = config.CORRECT_THRESHOLD
    cut_criteria = config.CUT_CRITERIA
    points = config.CONSECUTIVE_POINTS

    if threshold is None and cut_criteria == "MINIMUM_TARGETS":
        raise ValueError("`correct_targets_minimum` must be set when `cut_criteria` is 'MINIMUM_TARGETS'.")

    experiment_origin = "datapruebas"
    dataset_dir = os.path.join(DATA_DIR, "raw/tmt/" + experiment_origin + "/subjects/datapruebas_7_9_2024.json")
    analysis = log_and_run_tmt_analysis(
        dataset_path=dataset_dir,
        output_path=output_path,
        correct_targets_minimum=threshold,
        consecutive_points=points,
        cut_criteria=cut_criteria,
        calculate_crosses=config.CALCULATE_CROSSES
    )

    df_metrics = analysis.get_metrics_dataframe()
    print("Metrics DataFrame:")
    print(df_metrics.head())

    experiment = analysis.experiment
    print(f"Len of subjects: {len(experiment.subjects)}")

    return analysis


if __name__ == "__main__":
    run_analysis_with_configuration_parameters('/home/gianluca/Research/datapruebas_analysis/results')
