import logging
from typing import Any

import pandas as pd
from pandas import DataFrame

from src.config import NEUROPRUEBAS_METADATA_PATH


def retrieve_metadata(metadata_path):
    if metadata_path is None:
        raise ValueError("Metadata path is required for Neuropruebas data")

    tmt_metadata = pd.read_csv(metadata_path, sep=';')

    tmt_metadata = tmt_metadata.drop_duplicates(subset=['id'], keep='first')

    return tmt_metadata


def get_metadata_for_subject(subject_id, metadata_df):
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

    return result


def add_neuropruebas_metadata(metrics_df: pd.DataFrame, subject_col: str = "subject_id") -> tuple[
    DataFrame, set[Any]]:
    metadata_errors = set()

    metadata_df = retrieve_metadata(NEUROPRUEBAS_METADATA_PATH)

    # Crear una lista para guardar todos los DataFrames parciales con metadata
    df_list = []

    # Iterar sobre cada subject_id único
    for subject_id in metrics_df[subject_col].unique():
        # Filtrar filas del subject_id actual
        subject_rows = metrics_df[metrics_df[subject_col] == subject_id].copy()

        # Obtener metadata (diccionario)
        try:
            metadata = get_metadata_for_subject(subject_id, metadata_df)
        except:
            # TODO GIAN: por el momento si no encuentra metadata, poner None
            logging.warning(f"No se encontró metadata para subject_id: {subject_id}")
            metadata = {"año_de_nacimiento": None, "genero": None, "nivel_educativo": None, "nacionalidad": None}
            metadata_errors.add(subject_id)

        # Agregar metadata como nuevas columnas
        for metadata_value in ["año_de_nacimiento", "genero", "nivel_educativo", "nacionalidad"]:
            subject_rows[metadata_value] = metadata[metadata_value]

        # Agregar al listado
        df_list.append(subject_rows)

    # Concatenar todos los resultados
    result_df = pd.concat(df_list, ignore_index=True)

    column_renames = {
        "año_de_nacimiento": "birth_date",
        "genero": "gender",
        "nivel_educativo": "education_level",
        "nacionalidad": "nationality"
    }

    result_df = result_df.rename(columns=column_renames)

    return result_df, metadata_errors


def main():
    metrics = pd.read_csv(
        '/home/gianluca/Research/datapruebas_analysis/data/hand_analysis/2025-09-28_16-00-13/processed/tmt/neuropruebas/metrics.csv')
    nuevo_df, metadata_errors = add_neuropruebas_metadata(metrics, subject_col='subject_id')

    print("Cantidad de suject_id en metrics:", len(metrics['subject_id'].unique()))
    print("Cantidad de metadata errors (subject_id not found):", len(metadata_errors))
    print("Metadata errors (subject_id not found):", metadata_errors)


if __name__ == "__main__":
    main()
