import pandas as pd

from src.config import DATAPRUEBAS_METADATA_PATH


def retrieve_metadata(metadata_path):
    if metadata_path is None:
        raise ValueError("El archivo de metadata es requerido para el mapper DatapruebasTMTMapper")

    tmt_metadata = pd.read_csv(metadata_path, sep=';')
    tmt_metadata = tmt_metadata.drop_duplicates(subset=['id'], keep='first')

    return tmt_metadata


def add_datapruebas_metadata(metrics_df):
    metadata_df = retrieve_metadata(DATAPRUEBAS_METADATA_PATH)

    missing_subjects = set(metrics_df["subject_id"]) - set(metadata_df["id"])

    #TODO GIAN: descomentar cuando actualicemos la metadata para tener todos los sujetos
    if missing_subjects:
        raise ValueError(f"Los siguientes subject_id no están en metadata: {missing_subjects}")

    merged_df = pd.merge(metrics_df, metadata_df, left_on='subject_id', right_on='id', how='inner')

    merged_df = merged_df.drop(columns=['id', 'email'])

    # Removemos estas columnas ya que no estan en Neuropruebas
    merged_df = merged_df.drop(columns=['residence_country', 'residence_region'])

    return merged_df


def main():
    metrics = pd.read_csv(
        '/home/gianluca/Research/datapruebas_analysis/data/hand_analysis/2025-10-17_10-42-08/processed/tmt/datapruebas/metrics.csv'
    )
    print("Cantidad de suject_id en metrics:", len(metrics['subject_id'].unique()))

    nuevo_df = add_datapruebas_metadata(metrics)



if __name__ == "__main__":
    main()
