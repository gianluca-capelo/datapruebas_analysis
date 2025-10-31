import pandas as pd

from src.config import DATAPRUEBAS_METADATA_PATH


def retrieve_metadata(metadata_path):
    if metadata_path is None:
        raise ValueError("El archivo de metadata es requerido para el mapper DatapruebasTMTMapper")

    tmt_metadata = pd.read_csv(metadata_path, sep=';')
    tmt_metadata = tmt_metadata.drop_duplicates(subset=['id'], keep='first')

    return tmt_metadata


from datetime import datetime


def calculate_age(birthdate_str, recorded_at_str):
    try:
        if pd.isna(birthdate_str) or pd.isna(recorded_at_str):
            return None

        # birthdate tiene formato "YYYY-MM-DD"
        birthdate = datetime.strptime(str(birthdate_str), "%Y-%m-%d")

        # recorded_at puede tener formato ISO con zona horaria
        recorded_at = datetime.fromisoformat(str(recorded_at_str).replace("Z", "+00:00"))

        age = recorded_at.year - birthdate.year - (
                (recorded_at.month, recorded_at.day) < (birthdate.month, birthdate.day)
        )
        return age
    except Exception:
        raise ValueError("NO se puede calcular edad")


def add_datapruebas_metadata(metrics_df):
    metadata_df = retrieve_metadata(DATAPRUEBAS_METADATA_PATH)
    metadata_df = metadata_df[['id', 'birthdate', 'gender', 'level_of_education', 'nationality']]

    missing_subjects = set(metrics_df["subject_id"]) - set(metadata_df["id"])

    if missing_subjects:
        raise ValueError(f"Los siguientes subject_id no están en metadata: {missing_subjects}")

    merged_df = pd.merge(metrics_df, metadata_df, left_on='subject_id', right_on='id', how='inner')

    merged_df["age"] = merged_df.apply(
        lambda row: calculate_age(row["birthdate"], row["start_date"]),
        axis=1
    )

    merged_df = merged_df.drop(columns=['id', 'birthdate'])

    column_renames = {
        "level_of_education": "education_level"
    }

    merged_df = merged_df.rename(columns=column_renames)

    metadata_cols = ["age", "gender", "education_level", "nationality"]
    missing_mask = merged_df[metadata_cols].isna().any(axis=1)

    if missing_mask.any():
        bad_rows = merged_df.loc[missing_mask, ["subject_id"] + metadata_cols]
        raise ValueError(
            f"Algunas filas tienen metadatos faltantes:\n{bad_rows.to_string(index=False)}"
        )

    return merged_df


def main():
    metrics = pd.read_csv(
        '/home/gianluca/Research/datapruebas_analysis/data/hand_analysis/2025-10-31_09-58-40/processed/tmt/datapruebas/metrics.csv'
    )
    print("Cantidad de suject_id en metrics:", len(metrics['subject_id'].unique()))

    nuevo_df = add_datapruebas_metadata(metrics)

    return nuevo_df


if __name__ == "__main__":
    main()
