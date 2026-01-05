import glob
import json
import logging
import os
from ast import literal_eval
from typing import Optional, Dict, Tuple, List

import pandas as pd
from neurotask.tmt.invalid_cause import InvalidCause
from neurotask.tmt.mapper.mapper import TMTMapper
from neurotask.tmt.model.tmt_model import TMTExperiment, TMTSubject, CursorInfo, Coordinate, TMTTarget, TrialType, \
    TMTTrial, SubjectPersonalInformation, SessionContext

from neurotask.tmt.preprocessing.interpolation import interpolate_trajectory

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

    def map(self, data_path: str) -> TMTExperiment:

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
                id_suj = self.resolve_subject_id(df, nombre_de_archivo)

                session_data = self._read_neuropruebas_survey(df.copy())

                if "SSD" in list(df.columns):  # es sst
                    raise ValueError("El archivo corresponde a SST, no a Neuropruebas TMT")


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
                subjects[subject_id] = self.map_to_subject(subject_id, subject_data, session_data)
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
        if isinstance(e, ValueError):
            error_info['value_error_message'] = e.args[0] if e.args else str(e)
        if session_data:
            final_comment = session_data.get("comentarioFinal", None)
            if final_comment:
                error_info["final_comment"] = final_comment
        return error_info

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

    def map_to_subject(self, subject_id, subject_data: pd.DataFrame, session_data: dict) -> TMTSubject:
        px2mm = self._extract_px2mm_from_chinrest(subject_data)
        scale_factor = self._extract_scale_factor_from_chinrest(subject_data)
        if session_data is None:
            session_data = {}
        session_data['px2mm'] = px2mm
        session_data['scale_factor'] = scale_factor

        try:
            training_stimuli, testing_stimuli = self.get_stimuli(subject_data)
        except Exception as e:
            raise ValueError(f"Error obtaining stimuli for subject {subject_id}: {e}")

        testing_trials, training_trials = self.map_to_testing_training_trials(subject_data, testing_stimuli,
                                                                              training_stimuli)

        if len(testing_trials) == 0:
            raise ValueError("Subject must have at least one testing trial")

        return TMTSubject(
            training_trials=training_trials,
            testing_trials=testing_trials,
            target_radius=self._extract_first_valid_numeric(subject_data, 'radius'),
            canvas_size=self._extract_first_valid_numeric(subject_data, 'canvas_size'),
            session_data=session_data
        )

    def map_to_testing_training_trials(self, subject_data, testing_stimuli, training_stimuli):
        training_trials, testing_trials = (
            self._extract_position_and_time_data(subject_data, training_stimuli, testing_stimuli))

        return testing_trials, training_trials

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
            self, df: pd.DataFrame, train_stimulus: List[List[NeuropruebasTarget]],
            test_stimulus: List[List[NeuropruebasTarget]],
    ) -> Tuple[List[TMTTrial], List[TMTTrial]]:

        df_tmt = df[df["trial_type"] == "trail-making-test"].copy()

        n_trials = len(df_tmt)
        if n_trials == 0:
            raise ValueError("Subject csv does not contain any row with trial type 'trail-making-test'")
        trials = []

        for i, (_, row) in enumerate(df_tmt.iterrows()):
            trial_positions = self.get_trial_positions(row)
            trial_times = self.get_trial_times(row)
            first_click = self.get_first_click(row)

            if first_click is None:
                first_click = self.get_default_first_trial(trial_positions, trial_times)

            stimuli = train_stimulus[i] if i < 2 else test_stimulus[i - 2]
            trial = self.map_to_trial(
                first_click=first_click,
                positions=trial_positions,
                times=trial_times,
                stimuli=stimuli,
                trial_id=f"NEUROPRUEBAS_{i}",
                trial_order_of_appearance=i,
            )
            trials.append(trial)

        return trials[0:2], trials[2:]

    def get_default_first_trial(self, trial_positions, trial_times):
        first_position = trial_positions[0] if len(trial_positions) > 0 else (0.0, 0.0)
        first_time = trial_times[0] if len(trial_times) > 0 else 0
        first_click = CursorInfo(position=Coordinate(x=first_position[0], y=first_position[1]), time=first_time)
        return first_click

    def get_trial_times(self, row):
        try:
            raw_times = row.get("cursor_time")
            if isinstance(raw_times, str):
                times = literal_eval(raw_times)
            elif isinstance(raw_times, list):
                times = raw_times
            else:
                times = []

            # Aseguramos enteros
            trial_times = [int(t) for t in times if pd.notna(t)]

            return trial_times
        except Exception:
            return []

    def get_trial_positions(self, row):
        try:
            # Obtiene la lista cruda del campo 'position'
            raw_positions = row.get("position")

            # Asegura que sea lista o texto
            if isinstance(raw_positions, str):
                positions = literal_eval(raw_positions)
            elif isinstance(raw_positions, list):
                positions = raw_positions
            else:
                positions = []

            # Convierte cada punto a tupla (x, y)
            trial_positions = []
            for p in positions:
                if isinstance(p, str):
                    try:
                        x, y = literal_eval(p)
                    except Exception:
                        continue
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    x, y = p[0], p[1]
                else:
                    continue
                trial_positions.append((float(x), float(y)))

            return trial_positions
        except Exception:
            return []

    def get_first_click(self, row):

        # Intento 1: columna x_y_clicked_position
        try:
            if isinstance(row.get("x_y_clicked_position"), str):
                x, y, t = eval(row["x_y_clicked_position"])
                return CursorInfo(position=Coordinate(x=x, y=y), time=t)
        except Exception:
            pass

        # Intento 2: columnas separadas X_click, Y_click, T_click
        try:
            x = pd.to_numeric(row.get("X_click"), errors="coerce")
            y = pd.to_numeric(row.get("Y_click"), errors="coerce")
            t = pd.to_numeric(row.get("T_click"), errors="coerce")

            if pd.notna(x) and pd.notna(y) and pd.notna(t):
                return CursorInfo(position=Coordinate(x=x, y=y), time=t)
        except Exception:
            pass

        return None

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
                     trial_order_of_appearance: int, interpolate: bool = True) -> TMTTrial:

        targets = [
            TMTTarget(
                content=target.content,
                position=Coordinate(x=target.x, y=target.y)
            ) for target in stimuli
        ]

        trial_type = self._resolve_trial_type(targets)

        if len(positions) == 0 or len(positions) != len(times):
            return TMTTrial.invalid_trial(
                trial_id=trial_id,
                order_of_appearance=trial_order_of_appearance,
                stimuli=targets,
                trial_type=trial_type,
                invalid_cause=InvalidCause.INVALID_LENGTH
            )

        # Lógica de interpolación añadida
        if interpolate:
            # 1. Separar coordenadas
            raw_x = [p[0] for p in positions]
            raw_y = [p[1] for p in positions]

            # 2. Interpolar
            interp_x, interp_y, interp_t = interpolate_trajectory(raw_x, raw_y, times)
            
            # 3. Reconstruir cursor_trail
            cursor_trail = [
                CursorInfo(
                    position=Coordinate(x=x, y=y),
                    time=t
                ) for x, y, t in zip(interp_x, interp_y, interp_t)
            ]
            final_times = interp_t
        else:
            cursor_trail = [
                CursorInfo(
                    position=Coordinate(x=pos[0], y=pos[1]),
                    time=time
                ) for pos, time in zip(positions, times)
            ]
            final_times = times

        trial = TMTTrial(
            stimuli=targets,
            cursor_trail=cursor_trail,
            trial_type=trial_type,
            rt=final_times[-1] - final_times[0],  # Usar tiempos finales
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

    def _extract_px2mm_from_chinrest(self, df: pd.DataFrame) -> Optional[float]:
        chinrest_rows = df[df['trial_type'] == 'virtual-chinrest']
        if len(chinrest_rows) == 0:
            return None
        if 'px2mm' not in df.columns:
            return None
        px2mm = chinrest_rows['px2mm'].dropna()
        if len(px2mm) == 0:
            return None
        return float(px2mm.iloc[0])

    def _extract_scale_factor_from_chinrest(self, df: pd.DataFrame) -> Optional[float]:
        chinrest_rows = df[df['trial_type'] == 'virtual-chinrest']
        if len(chinrest_rows) == 0:
            return None
        if 'scale_factor' not in df.columns:
            return None
        scale_factor = chinrest_rows['scale_factor'].dropna()
        if len(scale_factor) == 0:
            return None
        return float(scale_factor.iloc[0])
