import glob
import json
import logging
import os
from typing import Optional, Dict, Tuple, List

import pandas as pd
from neurotask.tmt.mapper.mapper import TMTMapper
from neurotask.tmt.model.tmt_model import TMTExperiment, TMTSubject, CursorInfo, Coordinate, TMTTarget, TrialType, \
    TMTTrial, SubjectPersonalInformation, SessionContext

from src.config import LOG_DIR
from src.mapper.datapruebas.datapruebas_model import SubjectData
from src.mapper.neuropruebas.neuropruebas_model import NeuropruebasTarget

class LengthMismatchError(Exception):
    """Excepción personalizada para listas con longitudes inconsistentes."""

    def __init__(self, position_coords_len, first_clicks_len, times_len, training_stimuli_len, testing_stimuli_len):
        self.position_coords_len = position_coords_len
        self.first_clicks_len = first_clicks_len
        self.times_len = times_len
        self.training_stimuli_len = training_stimuli_len
        self.testing_stimuli_len = testing_stimuli_len

        message = (
            "Position coordinates, first click cursor info, cursor times, and stimuli lists "
            "must have the same length. "
            f"position coords = {position_coords_len}, "
            f"first clicks = {first_clicks_len}, "
            f"times = {times_len}, "
            f"training stimuli = {training_stimuli_len}, "
            f"testing stimuli = {testing_stimuli_len}."
        )
        super().__init__(message)

    def as_dict(self):
        return {
            "position_coords_len": self.position_coords_len,
            "first_clicks_len": self.first_clicks_len,
            "times_len": self.times_len,
            "training_stimuli_len": self.training_stimuli_len,
            "testing_stimuli_len": self.testing_stimuli_len
        }


class NeuropruebasFormatDetectionException(Exception):
    pass


class NeuropruebasTMTMapper(TMTMapper):

    def map(self, data_path: str, metadata_path: Optional[str] = None) -> TMTExperiment:

        experiment, session_data_dict = self._read_neuropruebas_output(data_path)

        return self.map_to_experiment(experiment, session_data_dict)

    def _read_neuropruebas_survey(self, df):
        survey_rows = df[df['trial_type'] == 'survey-html-form']
        if 'response' not in survey_rows.columns:
            return None

        responses = survey_rows['response']

        if len(responses) == 0:
            return None

        survey_response = {}

        for response in responses:
            # Eliminar la parte "comentarioFinal":"..."
            clean_response = response.replace(',"comentarioFinal":"}', '}')

            data = eval(clean_response)
            survey_response.update(data)

        serie = df["recorded_at"].dropna().astype(str)
        serie = serie[serie.str.strip() != ""]
        recorded_at = serie.iloc[0] if not serie.empty else None

        if recorded_at:
            survey_response['recorded_at'] = recorded_at

        return survey_response

    def _read_neuropruebas_output(self, folder_path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict]]:
        """
        Input:
               folder_path: Path de la carpeta donde se encuentran los datos a cargar

        Output:
               dictionary: Diccionario de la forma: {nombreDelSujeto: dataFrameConSusDatos}

        """
        dictionary = {}
        session_data_dict = {}

        for filename in glob.glob(os.path.join(folder_path, "*.csv")):
            with open(filename, "r") as f:
                nombre_de_archivo = f.name
                nombre_de_archivo = nombre_de_archivo.split("/")[-1]
                df = pd.read_csv(f.name, on_bad_lines="skip")
                session_data = self._read_neuropruebas_survey(df.copy())

                if "SSD" in list(df.columns):  # es sst
                    raise ValueError("El archivo corresponde a SST, no a Neuropruebas TMT")

                id_suj = self.resolve_subject_id(df, nombre_de_archivo)

                dictionary[id_suj] = df

                if session_data is not None:
                    session_data_dict[id_suj] = session_data

            # renombro el archivo
            if not id_suj.endswith(".csv"):
                os.rename(filename, os.path.join(folder_path, f"{id_suj}.csv"))

        return dictionary, session_data_dict

    def resolve_subject_id(self, df, nombre_de_archivo):
        if "hash" in list(df.columns):
            id_suj = df["hash"].iloc[0]
        elif "id" in list(df.columns):
            id_suj = df["id"].iloc[0]
        else:
            id_suj = nombre_de_archivo
        return id_suj

    def map_to_experiment(self, neuropruebas_experiment: dict, session_data_dict: dict) -> TMTExperiment:
        subjects = {}
        errors = []
        session_data = None
        for subject_id, subject_data in neuropruebas_experiment.items():
            try:
                session_data = session_data_dict.get(subject_id, None)
                subjects[subject_id] = self.map_to_subject(subject_data, session_data)
            except Exception as e:
                logging.exception(f"Error processing experiment for subject {subject_id}")

                error_info = self._build_error_info(e, session_data, subject_data, subject_id)

                errors.append(error_info)

                continue
        if errors:
            logging.warning(f"Errors found for subjects: {len(errors)} of total {len(neuropruebas_experiment)}. ")

            # Save errors to a CSV file
            save_file_path = os.path.join(LOG_DIR, "neuropruebas_mapping_errors.csv")
            errors_df = pd.DataFrame(errors)
            errors_df.to_csv(save_file_path, index=False)

        return TMTExperiment(subjects)

    def _build_error_info(self, e, session_data, subject_data, subject_id):
        error_info = {
            "subject_id": subject_id,
            "error": str(e),
            "num_rows": len(subject_data)
        }
        if isinstance(e, LengthMismatchError):
            error_info.update(e.as_dict())
        if session_data:
            final_comment = session_data.get("comentarioFinal", None)
            if final_comment:
                error_info["final_comment"] = final_comment
        return error_info

    def map_to_subject(self, subject_data: pd.DataFrame, session_data: dict) -> TMTSubject:

        training_stimuli, testing_stimuli = self.get_stimuli(subject_data)

        position_coordinates_for_every_trial, cursor_times_for_every_trial = self._extract_position_and_time_data(
            subject_data)

        first_click_cursor_info_for_every_trial = self.get_first_clicks_cursor_info(subject_data)

        self._validate_trial_data(cursor_times_for_every_trial, first_click_cursor_info_for_every_trial,
                                  position_coordinates_for_every_trial, testing_stimuli, training_stimuli)

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
            target_radius=self._extract_first_valid_numeric(subject_data, 'radius'),
            canvas_size=self._extract_first_valid_numeric(subject_data, 'canvas_size'),
            session_data=session_data
        )

    def _validate_trial_data(self, cursor_times_for_every_trial, first_click_cursor_info_for_every_trial,
                             position_coordinates_for_every_trial, testing_stimuli, training_stimuli):

        valid_length = (
                len(position_coordinates_for_every_trial) == len(first_click_cursor_info_for_every_trial) and
                len(first_click_cursor_info_for_every_trial) == len(cursor_times_for_every_trial) and
                len(cursor_times_for_every_trial) == len(training_stimuli) + len(testing_stimuli)
        )

        if not valid_length:
            raise LengthMismatchError(
                len(position_coordinates_for_every_trial),
                len(first_click_cursor_info_for_every_trial),
                len(cursor_times_for_every_trial),
                len(training_stimuli),
                len(testing_stimuli)
            )

    def get_stimuli(self, df):

        columns_of_interest = [str(i) for i in range(20)]

        if len(set(columns_of_interest).intersection(df.columns)) > 0:
            # Comentario Gus: Solo los de nacho tenian columnas separadas `columns_of_interest`
            return self.process_columns_of_interest(df, columns_of_interest)
        else:
            return self.process_training_test_stimuli(df)

    def process_columns_of_interest(self, df, columns_of_interest):
        stimulus = []
        # Filter DataFrame to only include columns of interest
        df_items_pos = df[columns_of_interest]

        # Filter rows where the first column contains '{'
        df_items_pos_cleaned = df_items_pos[df_items_pos["0"].str.contains("{", na=False)]

        # Transpose DataFrame
        df_items_pos_cleaned_transposed = df_items_pos_cleaned.T

        for column in df_items_pos_cleaned_transposed.columns:
            # Extract column data
            item_position_list = df_items_pos_cleaned_transposed[column].values

            trial_stimuli = []
            for item in item_position_list:
                try:
                    item_data = json.loads(item)
                    x = item_data["x"]
                    y = item_data["y"]
                    content = item_data["content"]
                    trial_stimuli.append(NeuropruebasTarget(content=content, x=x, y=y))

                except (json.JSONDecodeError, KeyError):
                    continue

            stimulus.append(trial_stimuli)

        return stimulus[0:2], stimulus[2:]

    def process_training_test_stimuli(self, df):
        training_stimulus = [stim["stimulus"] for stim in json.loads(df["train_stimuli"][1])]
        test_stimulus = [stim["stimulus"] for stim in json.loads(df["test_stimuli"][1])]

        processed_training_stimulus = [
            [NeuropruebasTarget(content=target["content"], x=target["x"], y=target["y"]) for target in trial]
            for trial in training_stimulus
        ]
        processed_test_stimulus = [
            [NeuropruebasTarget(content=target["content"], x=target["x"], y=target["y"]) for target in trial]
            for trial in test_stimulus
        ]

        return processed_training_stimulus, processed_test_stimulus

    def _extract_position_and_time_data(
            self, df: pd.DataFrame
    ) -> Tuple[List[List[Tuple[float, float]]], List[List[int]]]:

        is_valid_trial = self.determine_trial_validation(df)
        trial_mouse_positions = self.get_trial_mouse_positions(df, is_valid_trial)
        trial_cursor_times = self.get_trial_cursor_times(df, is_valid_trial)

        position_coordinates = [[eval(i) for i in t] for t in trial_mouse_positions]

        return position_coordinates, trial_cursor_times

    def determine_trial_validation(self, df):

        position_has_quotes = (df["position"] == '"').any()
        cursor_time_has_quotes = (df["cursor_time"] == '"').any()
        position_has_nan = df["position"].isna().any()
        cursor_time_has_nan = df["cursor_time"].isna().any()

        quotes_means_no_trial = position_has_quotes and cursor_time_has_quotes
        nan_means_no_trial = position_has_nan and cursor_time_has_nan

        if quotes_means_no_trial:
            return lambda trial_data: isinstance(trial_data, str) and trial_data != '"'
        if nan_means_no_trial:
            return lambda trial_data: isinstance(trial_data, str)
        else:
            raise NeuropruebasFormatDetectionException(
                f"Unable to determine the format of the data. "
                f"position_has_nan: {position_has_nan}, cursor_time_has_nan: {cursor_time_has_nan}, "
                f"position_has_quotes: {position_has_quotes}, cursor_time_has_quotes: {cursor_time_has_quotes}"
            )

    def get_trial_mouse_positions(self, df, is_valid_trial):
        return self.get_trial_data(df, "position", is_valid_trial)

    def get_trial_cursor_times(self, df, is_valid_trial):
        return self.get_trial_data(df, "cursor_time", is_valid_trial)

    def get_trial_data(self, df, column_name, is_valid_trial):

        raw_mouse_data = [trial_data for trial_data in df[column_name] if is_valid_trial(trial_data)]

        return [eval(t) for t in raw_mouse_data]

    def get_first_clicks_cursor_info(self, df):

        df_tmt = df[df["trial_type"] == "trail-making-test"]

        # TODO GIAN: por el momento usamos este preguntar a Gus con cual ir
        try:
            x_y_clicked_position = [
                eval(i) for i in df_tmt["x_y_clicked_position"] if isinstance(i, str)
            ]

            first_clicks = [
                CursorInfo(position=Coordinate(x=x, y=y), time=t) for (x, y, t) in x_y_clicked_position
            ]

            return first_clicks

        except:
            # First click position

            df_tmt["X_click"] = pd.to_numeric(df["X_click"], errors="coerce")
            df_tmt["Y_click"] = pd.to_numeric(df["Y_click"], errors="coerce")
            df_tmt["T_click"] = pd.to_numeric(df["T_click"], errors="coerce")

            x_first_clicks = df_tmt["X_click"].tolist()
            y_first_clicks = df_tmt["Y_click"].tolist()
            t_first_clicks = df_tmt["T_click"].tolist()

            first_clicks_second_option = [
                CursorInfo(position=Coordinate(x=x, y=y), time=t) for x, y, t in
                zip(x_first_clicks, y_first_clicks, t_first_clicks)
            ]

            return first_clicks_second_option

    def map_to_trials(self, stimulus: Optional[List[List[NeuropruebasTarget]]],
                      position_coordinates: List[List[Tuple[float, float]]],
                      cursor_times: List[List[int]],
                      first_clicks_info: List[CursorInfo]
                      ) -> List[TMTTrial]:

        if not stimulus:
            return []

        return [
            self.map_to_trial(first_click, positions, times, stimuli, trial_id=f"NEUROPRUEBAS_{str(i)}",
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
                     times: List[int], stimuli: List[NeuropruebasTarget], trial_id: str,
                     trial_order_of_appearance: int) -> TMTTrial:

        self._validate_trial_positions_and_times(positions, times)

        targets = [
            TMTTarget(
                content=target.content,
                position=Coordinate(x=target.x, y=target.y)
            ) for target in stimuli
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

    def _extract_first_valid(self, df, column: str):
        return df[column][df[column].notna()].values[0]

    def _extract_first_valid_numeric(self, df, column: str):
        df[column] = pd.to_numeric(df[column], errors="coerce")

        return df[column][df[column].notna()].values[0]
