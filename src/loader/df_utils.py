import pandas as pd

from src.loader.metadata.datapruebas_metadata import add_datapruebas_metadata
from src.loader.metadata.neuropruebas_metadata import add_neuropruebas_metadata


def concat_dataframes(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        report: bool = True
) -> pd.DataFrame:
    """
    Concatena df1 y df2 aunque tengan columnas diferentes.
    - Imprime las columnas que están solo en df1 y solo en df2.
    - El orden final de columnas es: columnas de df1 + columnas extra de df2.
      Puedes ordenar alfabéticamente las extra con sort_extra=True.
    - Con check_dtypes=True, avisa si los dtypes difieren en columnas comunes.
    """
    cols1 = list(df1.columns)
    cols2 = list(df2.columns)

    only_in_df1 = [c for c in cols1 if c not in cols2]
    only_in_df2 = [c for c in cols2 if c not in cols1]

    if report:
        if only_in_df1 or only_in_df2:
            print("🔍 Diferencias de columnas:")
            if only_in_df1:
                print(f"  • Solo en df1: {only_in_df1}")
            if only_in_df2:
                print(f"  • Solo en df2: {only_in_df2}")
        else:
            print("✅ Ambos DataFrames tienen las mismas columnas.")

    # Orden final: columnas de df1 + extras de df2 (en su orden o alfabéticas)
    extras = [c for c in cols2 if c not in cols1]
    all_cols = cols1 + extras

    dup1 = df1.columns[df1.columns.duplicated()].tolist()
    dup2 = df2.columns[df2.columns.duplicated()].tolist()
    if dup1 or dup2:
        raise ValueError(f"df1 columnas duplicadas: {dup1} | df2 columnas duplicadas: {dup2}")

    # Alinear ambos DataFrames al mismo conjunto de columnas
    df1_aligned = df1.reindex(columns=all_cols)
    df2_aligned = df2.reindex(columns=all_cols)

    # Concatenar
    return pd.concat([df1_aligned, df2_aligned], ignore_index=True)

