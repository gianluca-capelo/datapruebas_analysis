import pandas as pd

from src.config import TMT_NEUROPRUEBAS_METRICS_PATH, TMT_DATAPRUEBAS_METRICS_PATH, TMT_METRICS_PATH


def join_tmt_analysis() -> pd.DataFrame:
    neuropruebas_metrics = pd.read_csv(TMT_NEUROPRUEBAS_METRICS_PATH)
    datapruebas_metrics = pd.read_csv(TMT_DATAPRUEBAS_METRICS_PATH)

    neuropruebas_metrics["experiment_origin"] = "neuropruebas"
    datapruebas_metrics["experiment_origin"] = "datapruebas"

    metrics = concat_dataframes(neuropruebas_metrics, datapruebas_metrics)

    metrics.to_csv(TMT_METRICS_PATH, index=False)

    return metrics


def concat_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    if list(df1.columns) != list(df2.columns):
        raise ValueError("DataFrames must have the same columns to be concatenated.")

    return pd.concat([df1, df2], ignore_index=True)


if __name__ == "__main__":
    df_metrics = join_tmt_analysis()
    print(df_metrics.head())
