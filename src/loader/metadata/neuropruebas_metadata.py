import pandas as pd

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
    df["_mail_str"] = df["email"].astype(str).str.strip().str.lower()

    # Filtrar por id
    matched = df[df["_id_str"] == subject_id]

    # Si no encontró por id, buscar por mail
    if matched.empty:
        matched = df[df["_mail_str"] == subject_id]

    # Si hay más de una fila, lanzar error
    if len(matched) > 1:
        raise ValueError(f"Más de una fila encontrada para subject_id: {subject_id}")
    # Si no hay coincidencias
    if matched.empty:
        raise ValueError(f"Ninguna fila encontrada para subject_id: {subject_id}")

    # Convertir la fila a diccionario y devolver (sin las columnas auxiliares)
    result = matched.iloc[0].drop(labels=["_id_str", "_mail_str"]).to_dict()
    return result


def add_neuropruebas_metadata(metrics_df: pd.DataFrame, subject_col: str = "subject_id", ) -> pd.DataFrame:
    metadata_df = retrieve_metadata(NEUROPRUEBAS_METADATA_PATH)

    # Crear una lista para guardar todos los DataFrames parciales con metadata
    df_list = []

    # Iterar sobre cada subject_id único
    for subject_id in metrics_df[subject_col].unique():
        # Filtrar filas del subject_id actual
        subject_rows = metrics_df[metrics_df[subject_col] == subject_id].copy()

        # Obtener metadata (diccionario)
        metadata = get_metadata_for_subject(subject_id, metadata_df)

        # Agregar metadata como nuevas columnas
        for key, value in metadata.items():
            subject_rows[key] = value

        # Agregar al listado
        df_list.append(subject_rows)

    # Concatenar todos los resultados
    result_df = pd.concat(df_list, ignore_index=True)

    return result_df
