import logging

from neurotask.tmt.mapper.mapper import TMTMapper
from neurotask.tmt.metrics.distance_calculation import calculate_distance
from neurotask.tmt.model.tmt_model import *

from src.mapper.datapruebas.datapruebas_model import *


class DatapruebasTMTMapper(TMTMapper):

    def map(self, data_path: str, metadata_path: Optional[str] = None) -> TMTExperiment:

        experiment = self._read_datapruebas_output(data_path)

        return self.map_to_experiment(experiment)

    def _read_datapruebas_output(self, file_path: str) -> ExperimentRunCollection:
        with open(file_path, "r") as file:
            data = json.load(file)

        tmt_experiment_data = ExperimentRunCollection(**data)
        return tmt_experiment_data

    def map_to_experiment(self, datapruebas_experiment: ExperimentRunCollection) -> TMTExperiment:
        subjects = {}
        for experiment in datapruebas_experiment.experiments:
            if self._has_finished_status(experiment):

                if len(experiment.records) == 0:
                    logging.warning(f"No records found for subject {experiment.subject_id}")
                    continue
                try:

                    subjects[experiment.subject_id] = self.map_to_subject(experiment.records[0])

                except IndexError:
                    logging.exception(f"Error processing experiment for subject {experiment.subject_id}")
                except ValueError:
                    logging.exception(f"Subject {experiment.subject_id} has no testing trials")

        return TMTExperiment(subjects)

    def _has_finished_status(self, experiment):
        return experiment.experiment_status == "Finalizado"

    def map_to_subject(self, record: Record) -> TMTSubject:
        subject_data_list = record.data

        (
            position_coordinates_for_every_trial,
            first_click_cursor_info_for_every_trial,
            cursor_times_for_every_trial
        ) = self._extract_position_and_click_and_time_data(subject_data_list)

        self._validate_trial_data(cursor_times_for_every_trial, first_click_cursor_info_for_every_trial,
                                  position_coordinates_for_every_trial)

        training_trials = self.map_to_trials(
            subject_data_list[1].training_stimuli,
            position_coordinates_for_every_trial[0:2],
            cursor_times_for_every_trial[0:2],
            first_click_cursor_info_for_every_trial[0:2]
        )

        testing_trials = self.map_to_trials(
            subject_data_list[1].testing_stimuli,
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
            session_data=self._extract_session_data(subject_data_list)
        )

    def _validate_trial_data(self, cursor_times_for_every_trial, first_click_cursor_info_for_every_trial,
                             position_coordinates_for_every_trial):
        valid_data = (
                len(position_coordinates_for_every_trial) == len(first_click_cursor_info_for_every_trial) and
                len(first_click_cursor_info_for_every_trial) == len(cursor_times_for_every_trial)
        )
        if not valid_data:
            raise ValueError(
                f"Position coordinates, first click cursor info and cursor times and total time must have the same length" +
                f"position coords =  {len(position_coordinates_for_every_trial)}, " +
                f"first clicks = {len(first_click_cursor_info_for_every_trial)}, " +
                f"times = {len(cursor_times_for_every_trial)} respectively.")

    def map_to_trials(self, stimulus: Optional[List[StimulusTrial]],
                      position_coordinates: List[List[Tuple[float, float]]],
                      cursor_times: List[List[int]],
                      first_clicks_info: List[CursorInfo]
                      ) -> List[TMTTrial]:

        if not stimulus:
            return []

        return [
            self.map_to_trial(first_click, positions, times, stimuli, trial_id=f"DATAPRUEBAS_{str(i)}",
                              trial_order_of_appearance=i)
            for i, (stimuli, positions, times, first_click) in
            enumerate(zip(stimulus, position_coordinates, cursor_times, first_clicks_info))
        ]

    def _extract_position_and_click_and_time_data(self, subject_data_list: List[SubjectData]):
        position_coordinates_for_every_trial = []
        first_click_cursor_info_for_every_trial = []
        cursor_time: List[List[int]] = []

        for subject_data in subject_data_list:
            if subject_data.position_coordinates is not None:
                position_coordinates_for_every_trial.append(subject_data.position_coordinates)
                x_y_clicked_position = subject_data.x_y_clicked_position
                if x_y_clicked_position is not None:
                    first_click_cursor_info_for_every_trial.append(CursorInfo(
                        position=Coordinate(x=x_y_clicked_position[0], y=x_y_clicked_position[1]),
                        time=x_y_clicked_position[2]
                    ))
                else:
                    first_click_cursor_info_for_every_trial.append(None)
                if subject_data.cursor_time is not None:
                    cursor_time.append(subject_data.cursor_time)

        return position_coordinates_for_every_trial, first_click_cursor_info_for_every_trial, cursor_time

    def map_to_trial(self, first_click: CursorInfo, positions: List[Tuple[float, float]],
                     times: List[int], stimuli: StimulusTrial, trial_id: str,
                     trial_order_of_appearance: int) -> TMTTrial:

        self._validate_trial_positions_and_times(positions, times)

        targets = [
            TMTTarget(
                content=target.content,
                position=Coordinate(x=target.x, y=target.y)
            ) for target in stimuli.targets
        ]

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

    def _validate_trial_positions_and_times(self, positions, times):
        if len(positions) == 0:
            raise ValueError("Positions must not be empty")
        if len(positions) != len(times):
            raise ValueError("Positions and times must have the same length")

    def _resolve_trial_type(self, targets: List[TMTTarget]) -> TrialType:
        return TrialType.PART_B if targets[1].content == 'A' else TrialType.PART_A

    def _extract_first_valid(self, data_list: List[SubjectData], attribute: str):
        return next((getattr(data, attribute) for data in data_list if getattr(data, attribute) is not None), None)

    def _extract_session_data(self, subject_data_list: List[SubjectData]) -> Optional[dict]:
        session_data = None
        for subject_data in subject_data_list:
            if isinstance(subject_data.response, ResponseDetail):
                session_data = {
                    "device": subject_data.response.dispositivo,
                    "hand": subject_data.response.mano,
                    "device_config": subject_data.response.dispositivo_config,
                    "alcohol_drugs": subject_data.response.alcohol_drogas,
                    "treatment": subject_data.response.tratamiento,
                    "pad_usage": subject_data.response.usoDelPad,
                    "final_comment": subject_data.response.comentarioFinal
                }
                break
        return session_data
