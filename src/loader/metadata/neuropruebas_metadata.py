import logging
from datetime import datetime
from typing import Any

import pandas as pd
from pandas import DataFrame

from src.config import NEUROPRUEBAS_METADATA_PATH, OLD_NEUROPRUEBAS_METADATA_PATH


def retrieve_metadata(metadata_path):
    if metadata_path is None:
        raise ValueError("Metadata path is required for Neuropruebas data")

    tmt_metadata = pd.read_csv(metadata_path, sep=';')

    tmt_metadata = tmt_metadata.drop_duplicates(subset=['id'], keep='first')

    return tmt_metadata


def calculate_age(birth_date, recorded_at):
    """
    Calcula la edad en base al año de nacimiento y la fecha registrada.

    :param birth_date: Año de nacimiento (int o str)
    :param recorded_at: Fecha en formato 'YYYY-MM-DD HH:MM:SS'
    :return: Edad (int)
    """
    birth_year = int(birth_date)
    age = recorded_at.year - birth_year
    return age


def get_metadata_for_subject(subject_id, metadata_df, recorded_at):
    """
    Busca el subject_id en metadata_df por 'id' o 'mail'.
    Devuelve un diccionario con todas las columnas de la fila encontrada.

    Parámetros:
        subject_id: valor a buscar (puede ser id numérico o email).
        metadata_df (pd.DataFrame): DataFrame con metadata, debe contener 'id' y 'mail'.

    Retorna:
        dict: diccionario con columna:valor de la fila correspondiente.

    Lanza:
        ValueError: si se encuentran múltiples filas coincidentes.
    """

    # Preparar columnas para comparación
    df = metadata_df.copy()
    df["_id_str"] = df["id"].astype(str).str.strip().str.lower()
    df["_email_str"] = df["email"].astype(str).str.strip().str.lower()

    # Filtrar por id
    matched = df[df["_id_str"] == subject_id]

    # Si no encontró por id, buscar por mail
    if matched.empty:
        matched = df[df["_email_str"] == subject_id]

    # Si hay más de una fila, lanzar error
    if len(matched) > 1:
        raise ValueError(f"Más de una fila encontrada para subject_id: {subject_id}")
    # Si no hay coincidencias
    if matched.empty:
        raise ValueError(f"Ninguna fila encontrada para subject_id: {subject_id}")

    # Convertir la fila a diccionario y devolver (sin las columnas auxiliares)
    result = matched.iloc[0].drop(labels=["_id_str", "_email_str"]).to_dict()

    result["edad"] = calculate_age(result["año_de_nacimiento"], recorded_at )

    return result


def add_neuropruebas_metadata(metrics_df: pd.DataFrame, subject_col: str = "subject_id") -> tuple[
    DataFrame, set[Any]]:
    metadata_errors = set()

    metadata_df = retrieve_metadata(NEUROPRUEBAS_METADATA_PATH)

    old_subjects_metadata = pd.read_csv(OLD_NEUROPRUEBAS_METADATA_PATH)

    # Crear una lista para guardar todos los DataFrames parciales con metadata
    df_list = []

    # Iterar sobre cada subject_id único
    for subject_id in metrics_df[subject_col].unique():
        # Filtrar filas del subject_id actual
        subject_rows = metrics_df[metrics_df[subject_col] == subject_id].copy()

        recorded_at = subject_rows["recorded_at"]
        recorded_at = recorded_at[recorded_at.notnull()].iloc[0]
        recorded_at = datetime.strptime(recorded_at, "%Y-%m-%d %H:%M:%S")
        if pd.isna(recorded_at) or recorded_at == "":
            raise ValueError("No se puede inferir año de nacimiento sin fecha registrada " + subject_id)

        # Obtener metadata (diccionario)
        try:
            metadata = get_metadata_for_subject(subject_id, metadata_df, recorded_at)
        except:
            try:
                metadata = get_metadata_from_metrics(subject_id, subject_rows.copy(), old_subjects_metadata)
            except ValueError as e:
                logging.error(f"No se pudo inferir metadata para subject_id {subject_id}: {e}")
                metadata_errors.add(subject_id)
                continue

        # Agregar metadata como nuevas columnas
        for metadata_value in ["edad", "genero", "nivel_educativo", "nacionalidad"]:
            subject_rows[metadata_value] = metadata[metadata_value]

        # Agregar al listado
        df_list.append(subject_rows)

    # Concatenar todos los resultados
    result_df = pd.concat(df_list, ignore_index=True)

    column_renames = {
        "edad": "age",
        "genero": "gender",
        "nivel_educativo": "education_level",
        "nacionalidad": "nationality"
    }

    result_df = result_df.rename(columns=column_renames)

    return result_df, metadata_errors


def get_metadata_from_metrics(subject_id, subject_rows: pd.DataFrame, old_subjects_metadata: pd.DataFrame) -> dict:
    age = subject_rows["age"].iloc[0]
    if pd.isna(age) or age == "":
        raise ValueError("No se puede inferir edad")

    mail = subject_rows["mail"].iloc[0]

    old_subject_metadata = old_subjects_metadata[
        old_subjects_metadata["Mail"].str.strip().str.lower() == str(mail).strip().lower()]

    if old_subject_metadata.empty:
        raise ValueError("No se puede inferir sin metadata antigua")
    gender = old_subject_metadata["genero"].iloc[0]
    education_level = old_subject_metadata["nivel_educativo"].iloc[0]
    nationality = old_subject_metadata["pais"].iloc[0]

    metadata = {
        'edad': age,
        'genero': gender,
        'nivel_educativo': education_level,
        'nacionalidad': nationality
    }

    return metadata




def main():
    metrics = pd.read_csv(
        '/home/gianluca/Research/datapruebas_analysis/data/hand_analysis/2025-10-30_08-48-59/processed/tmt/neuropruebas/metrics.csv'
    )
    print("Cantidad de suject_id en metrics:", len(metrics['subject_id'].unique()))

    nuevo_df, metadata_errors = add_neuropruebas_metadata(metrics, subject_col='subject_id')

    print("Cantidad de metadata errors (subject_id not found):", len(metadata_errors))
    print("Metadata errors (subject_id not found):", metadata_errors)

    return nuevo_df


if __name__ == "__main__":
    main()
