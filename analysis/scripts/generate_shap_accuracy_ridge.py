"""Generate the supplementary single-panel SHAP figure (Accuracy — Ridge).

Same as panel C of the main SHAP figure (Accuracy) but using the Ridge model
instead of SVR, for thesis Supplementary Figure 1 (comparable to the c
coefficient model, which also uses Ridge).

Reuses FEATURE_LABELS / _compute_shap / style constants from
generate_shap_figures.py (importing it also applies its scienceplots style).

Usage:
    python -m analysis.scripts.generate_shap_accuracy_ridge
"""
import os

import matplotlib.pyplot as plt

from analysis.scripts.generate_shap_figures import (
    FEATURE_LABELS, _compute_shap, _translate_parts,
    C_PURPLE, FIGURES_DIR, DPI, TOP_N,
    _FIG_W, _FIG_H, _TITLE_FS, _LABEL_FS, _TICK_FS, _ANNOT_FS, _YTICK_FS,
)

# Single model to explain
DATASET   = "tmt_accuracy"
MODEL     = "Ridge"
TIMESTAMP = "2026-03-07_1213"
TASK      = "regression"
TITLE     = "Accuracy - Ridge"
OUT_NAME  = "figS1_shap_accuracy_ridge.png"


def _draw_panel(ax, label, shap_df, color):
    """Render one SHAP importance panel (mirrors generate_shap_figures.main loop)."""
    df_plot = shap_df.sort_values("mean_abs_shap", ascending=True).tail(TOP_N)
    df_plot.index = df_plot.index.map(lambda x: FEATURE_LABELS.get(x, x))
    df_plot.index = df_plot.index.map(_translate_parts)

    bars = ax.barh(df_plot.index, df_plot["mean_abs_shap"], color=color, alpha=0.75)

    max_val = df_plot["mean_abs_shap"].max()
    for bar in bars:
        w = bar.get_width()
        ax.text(w, bar.get_y() + bar.get_height() / 2,
                f" {w:.3f}", va="center", ha="left",
                fontsize=_ANNOT_FS, color=color)
    ax.set_xlim(0, max_val * 1.5)

    ax.set_title(label, fontsize=_TITLE_FS)
    ax.set_xlabel("Media |SHAP| (entre folds seleccionados)", fontsize=_LABEL_FS * 0.9)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=_YTICK_FS)
    ax.tick_params(axis="x", labelsize=_TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading SHAP data...")
    shap_df = _compute_shap(DATASET, MODEL, TIMESTAMP, TASK)

    # Single panel sized like one quadrant of the 2x2 main figure (keeps fonts proportional)
    fig = plt.figure(figsize=(_FIG_W / 2, _FIG_H / 2))
    ax = fig.add_subplot(111)
    _draw_panel(ax, TITLE, shap_df, C_PURPLE)
    fig.tight_layout()

    save_path = os.path.join(FIGURES_DIR, OUT_NAME)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"\nSaved -> {save_path}")


if __name__ == "__main__":
    main()
