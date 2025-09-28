import pandas as pd

from src.config import TMT_NEUROPRUEBAS_METRICS_PATH, TMT_DATAPRUEBAS_METRICS_PATH, TMT_METRICS_PATH


def concat_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    if list(df1.columns) != list(df2.columns):
        raise ValueError("DataFrames must have the same columns to be concatenated.")

    return pd.concat([df1, df2], ignore_index=True)


