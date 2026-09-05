"""SHAP data loading, feature labels and panel drawing shared by the SHAP figures.

`shap_main` (2x2 panel), `shap_panels` (one image per panel),
`shap_accuracy_ridge` (supplementary) and `shap_k_mean_slides` (talk) all read
the same model/dataset table and draw the same kind of horizontal bar panel.
"""
import glob
import os

import pandas as pd

from src.config import BASE_DIR
from src.model.shap.analyze_shap_results import analyze_shap_results
from src.model.shap.run_shap import run_shap

from analysis.scripts.utils import THESIS_RUN, THESIS_RUN_K_MEAN

TOP_N = 15

# Panel geometry of the combined 2x2 figure. The font sizes of the notebook
# style are scaled by the same factor as the canvas so a panel keeps its
# proportions whether it is drawn alone or inside the grid.
FIG_W = 26
FIG_H = 16
_BASE_W = 10
_SCALE = FIG_W / _BASE_W

TITLE_FS = 13.5 * _SCALE
LABEL_FS = 16 * _SCALE
TICK_FS = 14 * _SCALE
ANNOT_FS = 10 * _SCALE
YTICK_FS = 14 * _SCALE * 0.7

# Okabe-Ito task colors, shared with the violin figure.
C_DEMO = "#777777"    # Age (demographic)
C_AMBER = "#E69F00"   # CDT / K_mean
C_PURPLE = "#9B59B6"  # Go/No-Go

XLABEL = {
    "es": "Media |SHAP| (entre folds seleccionados)",
    "en": "Mean |SHAP| (across selected folds)",
}

PANEL_TITLE = {
    "es": "Importancia media |SHAP| para {name}",
    "en": "Mean |SHAP| importance for {name}",
}

COMBINATIONS = [
    {"label": "A. Age - SVR", "label_es": "A. Edad - SVR",
     "dataset": "tmt_age", "model": "SVR", "task": "regression",
     "timestamp": THESIS_RUN, "color": C_DEMO},
    {"label": "B. $K_{mean}$ - RandomForestRegressor",
     "label_es": "B. $K_{mean}$ - RandomForestRegressor",
     "dataset": "tmt_k_mean", "model": "RandomForestRegressor", "task": "regression",
     "timestamp": THESIS_RUN_K_MEAN, "color": C_AMBER},
    {"label": "C. Accuracy - SVR", "label_es": "C. Accuracy - SVR",
     "dataset": "tmt_accuracy", "model": "SVR", "task": "regression",
     "timestamp": THESIS_RUN, "color": C_PURPLE},
    {"label": "D. $c$ coefficient - Ridge", "label_es": "D. coeficiente c - Ridge",
     "dataset": "tmt_c", "model": "Ridge", "task": "regression",
     "timestamp": THESIS_RUN, "color": C_PURPLE},
]

# Human-readable feature labels — mirrors paper_figures.ipynb FEATURE_LABELS
FEATURE_LABELS = {
    "is_valid_sum_A": "Valid sum (Part A)",
    "is_valid_sum_B": "Valid sum (Part B)",

    "area_difference_from_ideal_PART_A": "Area difference from ideal (Part A)",
    "area_difference_from_ideal_PART_B": "Area difference from ideal (Part B)",
    "area_difference_from_ideal_B_A_ratio": "Area difference from ideal (B/A ratio)",

    "average_duration_PART_A": "Average duration (Part A)",
    "average_duration_PART_B": "Average duration (Part B)",
    "average_duration_B_A_ratio": "Average duration (B/A ratio)",

    "distance_difference_from_ideal_PART_A": "Distance difference from ideal (Part A)",
    "distance_difference_from_ideal_PART_B": "Distance difference from ideal (Part B)",
    "distance_difference_from_ideal_B_A_ratio": "Distance difference from ideal (B/A ratio)",

    "hesitation_avg_speed_PART_A": "Hesitation average speed (Part A)",
    "hesitation_avg_speed_PART_B": "Hesitation average speed (Part B)",
    "hesitation_avg_speed_B_A_ratio": "Hesitation average speed (B/A ratio)",

    "hesitation_distance_PART_A": "Hesitation distance (Part A)",
    "hesitation_distance_PART_B": "Hesitation distance (Part B)",
    "hesitation_distance_B_A_ratio": "Hesitation distance (B/A ratio)",

    "hesitation_time_PART_A": "Hesitation time (Part A)",
    "hesitation_time_PART_B": "Hesitation time (Part B)",
    "hesitation_time_B_A_ratio": "Hesitation time (B/A ratio)",

    "inter_target_time_PART_A": "Inter-target time (Part A)",
    "inter_target_time_PART_B": "Inter-target time (Part B)",
    "inter_target_time_B_A_ratio": "Inter-target time (B/A ratio)",

    "intra_target_time_PART_A": "Intra-target time (Part A)",
    "intra_target_time_PART_B": "Intra-target time (Part B)",
    "intra_target_time_B_A_ratio": "Intra-target time (B/A ratio)",

    "max_duration_PART_A": "Hesitation max duration (Part A)",
    "max_duration_PART_B": "Hesitation max duration (Part B)",
    "max_duration_B_A_ratio": "Hesitation max duration (B/A ratio)",

    "mean_abs_acceleration_PART_A": "Mean absolute acceleration (Part A)",
    "mean_abs_acceleration_PART_B": "Mean absolute acceleration (Part B)",
    "mean_abs_acceleration_B_A_ratio": "Mean absolute acceleration (B/A ratio)",

    "mean_acceleration_PART_A": "Mean acceleration (Part A)",
    "mean_acceleration_PART_B": "Mean acceleration (Part B)",
    "mean_acceleration_B_A_ratio": "Mean acceleration (B/A ratio)",

    "mean_negative_acceleration_PART_A": "Mean negative acceleration (Part A)",
    "mean_negative_acceleration_PART_B": "Mean negative acceleration (Part B)",
    "mean_negative_acceleration_B_A_ratio": "Mean negative acceleration (B/A ratio)",

    "mean_speed_PART_A": "Mean speed (Part A)",
    "mean_speed_PART_B": "Mean speed (Part B)",
    "mean_speed_B_A_ratio": "Mean speed (B/A ratio)",

    "non_cut_correct_targets_touches_PART_A": "Complete correct touches (Part A)",
    "non_cut_correct_targets_touches_PART_B": "Complete correct touches (Part B)",
    "non_cut_correct_targets_touches_B_A_ratio": "Complete correct touches (B/A ratio)",

    "non_cut_rt_PART_A": "Complete time in trial (Part A)",
    "non_cut_rt_PART_B": "Complete time in trial (Part B)",
    "non_cut_rt_B_A_ratio": "Complete time in trial (B/A ratio)",

    "peak_abs_acceleration_PART_A": "Peak absolute acceleration (Part A)",
    "peak_abs_acceleration_PART_B": "Peak absolute acceleration (Part B)",
    "peak_abs_acceleration_B_A_ratio": "Peak absolute acceleration (B/A ratio)",

    "peak_acceleration_PART_A": "Peak acceleration (Part A)",
    "peak_acceleration_PART_B": "Peak acceleration (Part B)",
    "peak_acceleration_B_A_ratio": "Peak acceleration (B/A ratio)",

    "peak_negative_acceleration_PART_A": "Peak negative acceleration (Part A)",
    "peak_negative_acceleration_PART_B": "Peak negative acceleration (Part B)",
    "peak_negative_acceleration_B_A_ratio": "Peak negative acceleration (B/A ratio)",

    "peak_speed_PART_A": "Peak speed (Part A)",
    "peak_speed_PART_B": "Peak speed (Part B)",
    "peak_speed_B_A_ratio": "Peak speed (B/A ratio)",

    "rt_PART_A": "Time in trial (Part A)",
    "rt_PART_B": "Time in trial (Part B)",
    "rt_B_A_ratio": "Time in trial (B/A ratio)",

    "search_avg_speed_PART_A": "Search average speed (Part A)",
    "search_avg_speed_PART_B": "Search average speed (Part B)",
    "search_avg_speed_B_A_ratio": "Search average speed (B/A ratio)",

    "search_distance_PART_A": "Search distance (Part A)",
    "search_distance_PART_B": "Search distance (Part B)",
    "search_distance_B_A_ratio": "Search distance (B/A ratio)",

    "search_time_PART_A": "Search time (Part A)",
    "search_time_PART_B": "Search time (Part B)",
    "search_time_B_A_ratio": "Search time (B/A ratio)",

    "state_transitions_PART_A": "State transitions (Part A)",
    "state_transitions_PART_B": "State transitions (Part B)",
    "state_transitions_B_A_ratio": "State transitions (B/A ratio)",

    "std_abs_acceleration_PART_A": "STD absolute acceleration (Part A)",
    "std_abs_acceleration_PART_B": "STD absolute acceleration (Part B)",
    "std_abs_acceleration_B_A_ratio": "STD absolute acceleration (B/A ratio)",

    "std_acceleration_PART_A": "STD acceleration (Part A)",
    "std_acceleration_PART_B": "STD acceleration (Part B)",
    "std_acceleration_B_A_ratio": "STD acceleration (B/A ratio)",

    "std_negative_acceleration_PART_A": "STD negative acceleration (Part A)",
    "std_negative_acceleration_PART_B": "STD negative acceleration (Part B)",
    "std_negative_acceleration_B_A_ratio": "STD negative acceleration (B/A ratio)",

    "std_speed_PART_A": "STD speed (Part A)",
    "std_speed_PART_B": "STD speed (Part B)",
    "std_speed_B_A_ratio": "STD speed (B/A ratio)",

    "total_distance_PART_A": "Total distance (Part A)",
    "total_distance_PART_B": "Total distance (Part B)",
    "total_distance_B_A_ratio": "Total distance (B/A ratio)",

    "total_hesitations_PART_A": "Hesitations (Part A)",
    "total_hesitations_PART_B": "Hesitations (Part B)",
    "total_hesitations_B_A_ratio": "Hesitations (B/A ratio)",

    "travel_avg_speed_PART_A": "Travel average speed (Part A)",
    "travel_avg_speed_PART_B": "Travel average speed (Part B)",
    "travel_avg_speed_B_A_ratio": "Travel average speed (B/A ratio)",

    "travel_distance_PART_A": "Travel distance (Part A)",
    "travel_distance_PART_B": "Travel distance (Part B)",
    "travel_distance_B_A_ratio": "Travel distance (B/A ratio)",

    "travel_time_PART_A": "Travel time (Part A)",
    "travel_time_PART_B": "Travel time (Part B)",
    "travel_time_B_A_ratio": "Travel time (B/A ratio)",

    "age": "Age",

    "wrong_targets_touches_PART_A":           "Wrong target touches (Part A)",
    "wrong_targets_touches_PART_B":           "Wrong target touches (Part B)",
    "wrong_targets_touches_B_A_ratio":        "Wrong target touches (B/A ratio)",

    "scale_factor_PART_A":                    "Scale factor (Part A)",
    "scale_factor_PART_B":                    "Scale factor (Part B)",
    "scale_factor_B_A_ratio":                 "Scale factor (B/A ratio)",

    "number_of_crosses_PART_A":               "Number of crosses (Part A)",
    "number_of_crosses_PART_B":               "Number of crosses (Part B)",
    "number_of_crosses_B_A_ratio":            "Number of crosses (B/A ratio)",

    "sample_count_PART_A":                    "Sample count (Part A)",
    "sample_count_PART_B":                    "Sample count (Part B)",
    "sample_count_B_A_ratio":                 "Sample count (B/A ratio)",

    "valid_interval_count_PART_A":            "Valid interval count (Part A)",
    "valid_interval_count_PART_B":            "Valid interval count (Part B)",
    "valid_interval_count_B_A_ratio":         "Valid interval count (B/A ratio)",

    "non_cut_search_distance_PART_A":         "Complete search distance (Part A)",
    "non_cut_search_distance_PART_B":         "Complete search distance (Part B)",
    "non_cut_search_distance_B_A_ratio":      "Complete search distance (B/A ratio)",

    "non_cut_total_distance_PART_A":          "Complete total distance (Part A)",
    "non_cut_total_distance_PART_B":          "Complete total distance (Part B)",
    "non_cut_total_distance_B_A_ratio":       "Complete total distance (B/A ratio)",

    "non_cut_intra_target_time_PART_A":       "Complete intra-target time (Part A)",
    "non_cut_intra_target_time_PART_B":       "Complete intra-target time (Part B)",
    "non_cut_intra_target_time_B_A_ratio":    "Complete intra-target time (B/A ratio)",

    "non_cut_inter_target_time_PART_A":       "Complete inter-target time (Part A)",
    "non_cut_inter_target_time_PART_B":       "Complete inter-target time (Part B)",
    "non_cut_inter_target_time_B_A_ratio":    "Complete inter-target time (B/A ratio)",

    "non_cut_search_avg_speed_PART_A":        "Complete search average speed (Part A)",
    "non_cut_search_avg_speed_PART_B":        "Complete search average speed (Part B)",
    "non_cut_search_avg_speed_B_A_ratio":     "Complete search average speed (B/A ratio)",

    "non_cut_mean_speed_PART_A":              "Complete mean speed (Part A)",
    "non_cut_mean_speed_PART_B":              "Complete mean speed (Part B)",
    "non_cut_mean_speed_B_A_ratio":           "Complete mean speed (B/A ratio)",

    "non_cut_state_transitions_PART_A":       "Complete state transitions (Part A)",
    "non_cut_state_transitions_PART_B":       "Complete state transitions (Part B)",
    "non_cut_state_transitions_B_A_ratio":    "Complete state transitions (B/A ratio)",

    "non_cut_search_time_PART_A":             "Complete search time (Part A)",
    "non_cut_search_time_PART_B":             "Complete search time (Part B)",
    "non_cut_search_time_B_A_ratio":          "Complete search time (B/A ratio)",

    "non_cut_number_of_crosses_PART_A":       "Complete number of crosses (Part A)",
    "non_cut_number_of_crosses_PART_B":       "Complete number of crosses (Part B)",
    "non_cut_number_of_crosses_B_A_ratio":    "Complete number of crosses (B/A ratio)",
}


def translate_parts(label: str) -> str:
    """Translate the 'Part A'/'Part B' suffix of a feature label to Spanish."""
    return label.replace("Part A", "Parte A").replace("Part B", "Parte B")


def compute_shap(combo: dict) -> pd.DataFrame:
    """Mean |SHAP| per feature, re-fitting the model on every LOO fold."""
    print(f"  [{combo['dataset']}/{combo['model']}] Computing SHAP (may take several minutes)...")
    explanations, _ = run_shap(
        task=combo["task"], dataset_name=combo["dataset"],
        timestamp=combo["timestamp"], model_name_to_explain=combo["model"],
    )
    return analyze_shap_results(explanations, task=combo["task"])


def shap_from_csv(combo: dict) -> pd.DataFrame:
    """Rebuild mean |SHAP| from the per-fold CSV that run_shap already saved.

    Same aggregation as build_mean_shap_df (mean of |value| over the folds where
    the feature was selected; NaN = not selected), so figures can be re-styled
    without re-fitting every LOO fold.
    """
    pattern = os.path.join(BASE_DIR, "results", combo["task"], combo["timestamp"],
                           "*", combo["dataset"], f"shap_values_{combo['model']}.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No hay SHAP guardado para {combo['dataset']}/{combo['model']}: {pattern}\n"
            "Corré el script sin --from-csv al menos una vez.")

    print(f"Reusing saved SHAP values -> {matches[0]}")
    df = pd.read_csv(matches[0]).drop(columns=["fold", "base_value"])
    return pd.DataFrame({"mean_abs_shap": df.abs().mean(skipna=True)})


def top_features(shap_df: pd.DataFrame, top_n: int = TOP_N, lang: str = "en",
                 relabel=None) -> pd.DataFrame:
    """Top-N features sorted ascending, so barh draws the largest on top."""
    df_plot = shap_df.sort_values("mean_abs_shap", ascending=True).tail(top_n)
    df_plot.index = df_plot.index.map(lambda name: FEATURE_LABELS.get(name, name))
    if relabel is not None:
        df_plot.index = df_plot.index.map(relabel)
    elif lang == "es":
        df_plot.index = df_plot.index.map(translate_parts)
    return df_plot


def draw_shap_panel(ax, title: str, df_plot: pd.DataFrame, color: str, lang: str,
                    xlabel: str | None = None):
    """One horizontal mean-|SHAP| bar panel, identical across the SHAP figures."""
    bars = ax.barh(df_plot.index, df_plot["mean_abs_shap"], color=color, alpha=0.75)

    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, f" {width:.3f}",
                va="center", ha="left", fontsize=ANNOT_FS, color=color)
    ax.set_xlim(0, df_plot["mean_abs_shap"].max() * 1.5)

    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel(XLABEL[lang] if xlabel is None else xlabel, fontsize=LABEL_FS * 0.9)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=YTICK_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
