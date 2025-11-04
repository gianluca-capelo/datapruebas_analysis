import pandas as pd


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

    # Alinear ambos DataFrames al mismo conjunto de columnas
    df1_aligned = df1.reindex(columns=all_cols)
    df2_aligned = df2.reindex(columns=all_cols)

    # Concatenar
    return pd.concat([df1_aligned, df2_aligned], ignore_index=True)



def main():
    np_metrics = pd.read_csv(
        '/home/gianluca/Research/datapruebas_analysis/data/hand_analysis/2025-11-04_09-43-00/processed/tmt/neuropruebas/metrics.csv'
    )

    dp_metrics = pd.read_csv(
        '/home/gianluca/Research/datapruebas_analysis/data/hand_analysis/2025-11-04_09-43-00/processed/tmt/datapruebas/metrics.csv'
    )

    np_metrics["experiment_origin"] = "neuropruebas"
    dp_metrics["experiment_origin"] = "datapruebas"


    return concat_dataframes(np_metrics, dp_metrics)


if __name__ == "__main__":
    main()
