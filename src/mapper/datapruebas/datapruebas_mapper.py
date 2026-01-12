import logging
import os
from datetime import datetime, timezone

import pandas as pd
from neurotask.tmt.invalid_cause import InvalidCause
from neurotask.tmt.mapper.mapper import TMTMapper
from neurotask.tmt.metrics.distance_calculation import calculate_distance
from neurotask.tmt.model.tmt_model import *
from neurotask.tmt.preprocessing.interpolation import interpolate_trajectory

from src.config import LOG_DIR
from src.mapper.datapruebas.datapruebas_model import *


def parse_iso_datetime(date_str: str) -> datetime:
    """
    Convierte una cadena ISO 8601 (ej. '2024-09-05T21:31:22.375Z')
    a un objeto datetime con zona horaria UTC.
    """
    # La Z al final indica que está en UTCN
    return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


class DatapruebasTMTMapper(TMTMapper):

    def map(self, data_path: str) -> TMTExperiment:

        experiment = self._read_datapruebas_output(data_path)

        print(f"Mapping {len(experiment.experiments)} subjects...")

        tmt_experiment = self.map_to_experiment(experiment)

        print(f"Mapped {len(tmt_experiment.subjects)} valid subjects.")

        return tmt_experiment

    def _read_datapruebas_output(self, file_path: str) -> ExperimentRunCollection:
        with open(file_path, "r") as file:
            data = json.load(file)

        tmt_experiment_data = ExperimentRunCollection(**data)
        return tmt_experiment_data

    def map_to_experiment(self, datapruebas_experiment: ExperimentRunCollection) -> TMTExperiment:
        subjects = {}
        errors = []

        for experiment in datapruebas_experiment.experiments:
            if self._has_finished_status(experiment):

                if len(experiment.records) == 0:
                    logging.warning(f"No records found for subject {experiment.subject_id}")
                    errors.append({
                        "subject_id": experiment.subject_id,
                        "error": "No records found",
                        "num_records": 0
                    })
                    continue
                try:
                    start_date = parse_iso_datetime(experiment.start_date)
                    subjects[experiment.subject_id] = self.map_to_subject(experiment.records[0], start_date)

                except Exception as e:
                    logging.exception(f"Error processing experiment for subject {experiment.subject_id}")
                    errors.append({
                        "subject_id": experiment.subject_id,
                        "error": str(e),
                        "num_records": len(experiment.records)
                    })

        if errors:
            logging.warning(f"Errors found for subjects: {len(errors)} of total {len(datapruebas_experiment.experiments)}.")
            save_file_path = os.path.join(LOG_DIR, "datapruebas_mapping_errors.csv")
            errors_df = pd.DataFrame(errors)
            errors_df.to_csv(save_file_path, index=False)

        return TMTExperiment(subjects)

    def _has_finished_status(self, experiment):
        return experiment.experiment_status == "Finalizado"

    def map_to_subject(self, record: Record, start_date: datetime) -> TMTSubject:
        subject_data_list = record.data

        if len(subject_data_list) < 2:
            raise ValueError("Subject must have at least two trials (training and testing)")
        training_stimuli = subject_data_list[1].training_stimuli
        testing_stimuli = subject_data_list[1].testing_stimuli

        if training_stimuli is None or testing_stimuli is None:
            raise ValueError("Subject must have both training and testing stimuli")

        (
            position_coordinates_for_every_trial,
            first_click_cursor_info_for_every_trial,
            cursor_times_for_every_trial
        ) = self._extract_position_and_click_and_time_data(subject_data_list)

        training_trials = self.map_to_trials(
            training_stimuli,
            position_coordinates_for_every_trial[0:2],
            cursor_times_for_every_trial[0:2],
            first_click_cursor_info_for_every_trial[0:2]
        )

        testing_trials = self.map_to_trials(
            testing_stimuli,
            position_coordinates_for_every_trial[2:],
            cursor_times_for_every_trial[2:],
            first_click_cursor_info_for_every_trial[2:]
        )

        if len(testing_trials) == 0:
            raise ValueError("Subject must have at least one testing trial")

        return TMTSubject(
            training_trials=training_trials,
            testing_trials=testing_trials,
            target_radius=self._extract_first_valid(subject_data_list, 'radius'),
            canvas_size=self._extract_first_valid(subject_data_list, 'canvas_size'),
            session_data=self._extract_session_data(subject_data_list, start_date)
        )

    def map_to_trials(self, stimulus: Optional[List[StimulusTrial]],
                      position_coordinates: List[List[Tuple[float, float]]],
                      cursor_times: List[List[int]],
                      first_clicks_info: List[CursorInfo]
                      ) -> List[TMTTrial]:

        if not stimulus:
            return []

        trials = []

        for i, (stimuli, positions, times, first_click) in enumerate(
                zip(stimulus, position_coordinates, cursor_times, first_clicks_info)):

            if first_click is None:
                first_click = self.get_default_first_trial(positions, times)

            trials.append(self.map_to_trial(first_click, positions, times, stimuli, trial_id=f"DATAPRUEBAS_{str(i)}",
                                            trial_order_of_appearance=i))

        return trials

    def get_default_first_trial(self, trial_positions, trial_times):
        first_position = trial_positions[0] if len(trial_positions) > 0 else (0.0, 0.0)
        first_time = trial_times[0] if len(trial_times) > 0 else 0
        first_click = CursorInfo(position=Coordinate(x=first_position[0], y=first_position[1]), time=first_time)
        return first_click

    def _extract_position_and_click_and_time_data(self, subject_data_list: List[SubjectData]):
        position_coordinates_for_every_trial = []
        first_click_cursor_info_for_every_trial = []
        cursor_time: List[List[int]] = []

        subject_data_list = [sd for sd in subject_data_list if sd.trial_type == 'trail-making-test']

        if len(subject_data_list) == 0:
            raise ValueError("Subject data must have at least one trial")

        for subject_data in subject_data_list:

            missing_info = (subject_data.position_coordinates is None
                            or subject_data.cursor_time is None)

            if missing_info:
                raise ValueError("Subject has missing info")

            position_coordinates_for_every_trial.append(subject_data.position_coordinates)

            cursor_time.append(subject_data.cursor_time)

            x_y_clicked_position = subject_data.x_y_clicked_position
            if x_y_clicked_position is not None:
                first_click_cursor_info_for_every_trial.append(CursorInfo(
                    position=Coordinate(x=x_y_clicked_position[0], y=x_y_clicked_position[1]),
                    time=x_y_clicked_position[2]
                ))
            else:
                first_click_cursor_info_for_every_trial.append(None)

        return position_coordinates_for_every_trial, first_click_cursor_info_for_every_trial, cursor_time

    def map_to_trial(self, first_click: CursorInfo, positions: List[Tuple[float, float]],
                     times: List[int], stimuli: StimulusTrial, trial_id: str,
                     trial_order_of_appearance: int, interpolate: bool = True) -> TMTTrial:

        targets = [
            TMTTarget(
                content=target.content,
                position=Coordinate(x=target.x, y=target.y)
            ) for target in stimuli.targets
        ]

        trial_type = self._resolve_trial_type(targets)

        if len(positions) < 2 or len(positions) != len(times):
            return TMTTrial.invalid_trial(
                trial_id=trial_id,
                order_of_appearance=trial_order_of_appearance,
                stimuli=targets,
                trial_type=trial_type,
                invalid_cause=InvalidCause.INVALID_LENGTH
            )

        # Paso de Interpolación
        if interpolate:
            # 1. Separar la lista de tuplas [(x,y), ...] en dos listas [x, ...] y [y, ...]
            raw_x = [p[0] for p in positions]
            raw_y = [p[1] for p in positions]

            # 2. Llamar a la función con los argumentos correctos (x, y, t)
            # Recibir los 3 valores de retorno (x, y, t)
            interp_x, interp_y, interp_t = interpolate_trajectory(raw_x, raw_y, times)
            
            # 3. Crear el cursor_trail iterando sobre las 3 listas interpoladas
            cursor_trail = [
                CursorInfo(
                    position=Coordinate(x=x, y=y),
                    time=t
                ) for x, y, t in zip(interp_x, interp_y, interp_t)
            ]
        else:
            cursor_trail = [
                CursorInfo(
                    position=Coordinate(x=pos[0], y=pos[1]),
                    time=time
                ) for pos, time in zip(positions, times)
            ]

        trial = TMTTrial(
            stimuli=targets,
            cursor_trail=cursor_trail,
            trial_type=self._resolve_trial_type(targets),
            rt=times[-1] - times[0],
            id=trial_id,
            order_of_appearance=trial_order_of_appearance,
            with_custom_start=True,
            start=first_click
        )

        return trial

    def _resolve_trial_type(self, targets: List[TMTTarget]) -> TrialType:
        return TrialType.PART_B if targets[1].content == 'A' else TrialType.PART_A

    def _extract_first_valid(self, data_list: List[SubjectData], attribute: str):
        return next((getattr(data, attribute) for data in data_list if getattr(data, attribute) is not None), None)

    def _extract_px2mm_from_chinrest(self, subject_data_list: List[SubjectData]) -> Optional[float]:
        for subject_data in subject_data_list:
            if subject_data.trial_type == "virtual-chinrest" and subject_data.px2mm is not None:
                return subject_data.px2mm
        return None

    def _extract_scale_factor_from_chinrest(self, subject_data_list: List[SubjectData]) -> Optional[float]:
        for subject_data in subject_data_list:
            if subject_data.trial_type == "virtual-chinrest" and subject_data.scale_factor is not None:
                return subject_data.scale_factor
        return None

    def _extract_session_data(self, subject_data_list: List[SubjectData], start_date: datetime) -> Optional[dict]:
        session_data = None
        px2mm = self._extract_px2mm_from_chinrest(subject_data_list)
        scale_factor = self._extract_scale_factor_from_chinrest(subject_data_list)
        
        for subject_data in subject_data_list:
            if isinstance(subject_data.response, ResponseDetail):
                session_data = {
                    "device": subject_data.response.dispositivo,
                    "hand": subject_data.response.mano,
                    "device_config": subject_data.response.dispositivo_config,
                    "alcohol_drugs": subject_data.response.alcohol_drogas,
                    "treatment": subject_data.response.tratamiento,
                    "pad_usage": subject_data.response.usoDelPad,
                    "final_comment": subject_data.response.comentarioFinal,
                    "start_date": start_date.isoformat(),
                    "px2mm": px2mm,
                    "scale_factor": scale_factor
                }
                break
        return session_data
