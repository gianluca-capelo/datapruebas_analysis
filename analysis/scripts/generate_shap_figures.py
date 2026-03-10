"""Generate SHAP summary figure for all dataset/model combinations.

Produces a single multi-panel figure (2×2) that mirrors the style of
paper_figures.ipynb Figure 3.

Usage:
    python -m analysis.scripts.generate_shap_figures
"""
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots  # noqa: F401 — registers styles

from src.config import BASE_DIR
from src.model.shap.analyze_shap_results import analyze_shap_results
from src.model.shap.run_shap import run_shap

FIGURES_DIR = os.path.join(BASE_DIR, "analysis", "figures")

# ---------------------------------------------------------------------------
# Style — mirrors paper_figures.ipynb Setup cell
# ---------------------------------------------------------------------------
plt.style.use(["science", "no-latex"])

TITLE_FS = 13.5
LABEL_FS = 16
TICK_FS  = 14
ANNOT_FS = 10

_FIG_W  = 26
_FIG_H  = 16
_BASE_W = 10
_S      = _FIG_W / _BASE_W  # 2.6

_TITLE_FS = TITLE_FS * _S
_LABEL_FS = LABEL_FS * _S
_TICK_FS  = TICK_FS  * _S
_ANNOT_FS = ANNOT_FS * _S
_YTICK_FS = TICK_FS  * _S * 0.7

DPI   = 300
TOP_N = 15

# Okabe-Ito task-specific colors (same as notebook)
C_DEMO   = "#777777"  # Age (demographic)
C_AMBER  = "#E69F00"  # CDT / K_mean
C_PURPLE = "#9B59B6"  # Go/No-Go

COMBINATIONS = [
    {"label": "A. Age — SVR",                 "dataset": "tmt_age",      "model": "SVR",          "task": "regression", "timestamp": "2026-03-07_1213", "color": C_DEMO},
    {"label": "B. $K_{mean}$ — XGBRegressor", "dataset": "tmt_k_mean",   "model": "XGBRegressor", "task": "regression", "timestamp": "2026-03-06_2028", "color": C_AMBER},
    {"label": "C. Accuracy — SVR",            "dataset": "tmt_accuracy", "model": "SVR",          "task": "regression", "timestamp": "2026-03-07_1213", "color": C_PURPLE},
    {"label": "D. $c$ coefficient — Ridge",   "dataset": "tmt_c",        "model": "Ridge",        "task": "regression", "timestamp": "2026-03-07_1213", "color": C_PURPLE},
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


def _compute_shap(dataset, model, timestamp, task="regression"):
    """Compute SHAP values by re-fitting each model per LOO fold."""
    print(f"  [{dataset}/{model}] Computing SHAP (may take several minutes)...")
    explanations, _ = run_shap(
        task=task, dataset_name=dataset,
        timestamp=timestamp, model_name_to_explain=model,
    )
    return analyze_shap_results(explanations, task=task)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load / compute SHAP data for every panel
    print("Loading SHAP data...")
    shap_dfs = []
    for combo in COMBINATIONS:
        df = _compute_shap(combo["dataset"], combo["model"], combo["timestamp"], combo["task"])
        shap_dfs.append(df)
        print(f"  Done: {combo['label']}")

    # ---------------------------------------------------------------------------
    # Build 2×2 figure — mirrors paper_figures.ipynb Figure 3 cell
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(_FIG_W, _FIG_H), constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.12)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    for ax, combo, shap_df in zip(axes, COMBINATIONS, shap_dfs):
        color = combo["color"]
        df_plot = shap_df.sort_values("mean_abs_shap", ascending=True).tail(TOP_N)
        df_plot.index = df_plot.index.map(lambda x: FEATURE_LABELS.get(x, x))

        bars = ax.barh(df_plot.index, df_plot["mean_abs_shap"], color=color, alpha=0.75)

        max_val = df_plot["mean_abs_shap"].max()
        for bar in bars:
            w = bar.get_width()
            ax.text(w, bar.get_y() + bar.get_height() / 2,
                    f" {w:.3f}", va="center", ha="left",
                    fontsize=_ANNOT_FS, color=color)
        ax.set_xlim(0, max_val * 1.5)

        ax.set_title(combo["label"], fontsize=_TITLE_FS)
        ax.set_xlabel("Mean |SHAP|", fontsize=_LABEL_FS * 0.9)
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=_YTICK_FS)
        ax.tick_params(axis="x", labelsize=_TICK_FS)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save_path = os.path.join(FIGURES_DIR, "fig3_shap_importance.png")
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"\nSaved -> {save_path}")


if __name__ == "__main__":
    main()
