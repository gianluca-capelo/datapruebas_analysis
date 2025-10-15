import pandas as pd

from src.config import DATAPRUEBAS_METADATA_PATH


def retrieve_metadata(metadata_path):
    if metadata_path is None:
        raise ValueError("El archivo de metadata es requerido para el mapper DatapruebasTMTMapper")

    tmt_metadata = pd.read_csv(metadata_path)
    tmt_metadata = tmt_metadata.drop_duplicates(subset=['Id sujeto'], keep='first')

    return tmt_metadata


def add_datapruebas_metadata(metrics_df):
    metadata_df = retrieve_metadata(DATAPRUEBAS_METADATA_PATH)

    missing_subjects = set(metrics_df["subject_id"]) - set(metadata_df["Id sujeto"])

    #TODO GIAN: descomentar
    #if missing_subjects:
     #   raise ValueError(f"Los siguientes subject_id no están en metadata: {missing_subjects}")

    merged_df = pd.merge(metrics_df, metadata_df, left_on='subject_id', right_on='Id sujeto', how='inner')

    merged_df = merged_df.drop(columns=['Id sujeto', 'Id ejecucion', 'Estado'])

    # Removemos estas columnas ya que no estan en Neuropruebas
    merged_df = merged_df.drop(columns=['Pais de residencia', 'Región de residencia'])

    column_renames = {
        "Fecha de nacimiento": "birth_date",
        "Género": "gender",
        "Nivel educativo": "education_level",
        "Nacionalidad": "nationality"
    }

    merged_df = merged_df.rename(columns=column_renames)

    return merged_df
