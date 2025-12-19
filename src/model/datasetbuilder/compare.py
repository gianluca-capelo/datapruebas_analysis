import os
import pandas as pd

from src import config

# Directories

data_dir = config.DATA_DIR
dir_new = os.path.join(data_dir, "processed")
dir_old = os.path.join(data_dir, "processed_old")

# Archivos a comparar
files = [
    "demographic_df.csv",
    "df_digital_tmt_with_target.csv",
    "non_digital_df.csv"
]

def compare_dataframes(df1, df2):
    """Returns True if the DataFrames are identical (same content and columns)"""
    try:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=False, check_like=True)
        return True
    except AssertionError:
        return False

for file in files:
    path_new = os.path.join(dir_new, file)
    path_old = os.path.join(dir_old, file)

    if not os.path.exists(path_new) or not os.path.exists(path_old):
        print(f"❌ File {file} does not exist in one of the directories.")
        continue

    # Read CSV
    df_new = pd.read_csv(path_new)
    df_old = pd.read_csv(path_old)

    # Compare
    if compare_dataframes(df_new, df_old):
        print(f"✅ {file} → They are equal.")
    else:
        print(f"⚠️ {file} → They are different.")

        # Show some differences
        diff_rows = df_new.compare(df_old) if df_new.shape == df_old.shape else "Tamaños distintos"
        print(f"Diferencias encontradas en {file}:")
        print(diff_rows)

