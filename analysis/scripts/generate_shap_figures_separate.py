"""Generate each SHAP panel of the main figure as a SEPARATE image.

Same four model/dataset combinations as the combined 2x2 SHAP figure
(generate_shap_figures.py), but one standalone image per panel — so they can
be stacked vertically in the thesis, larger and more legible.

Reuses COMBINATIONS / _compute_shap / _translate_parts / FEATURE_LABELS and the
style constants from generate_shap_figures.py (so the panels stay identical to
the combined figure).

Usage:
    python -m analysis.scripts.generate_shap_figures_separate            # inglés
    python -m analysis.scripts.generate_shap_figures_separate --lang es  # castellano
"""
import os

import matplotlib.pyplot as plt

from analysis.scripts.generate_shap_figures import (
    COMBINATIONS, _compute_shap, _translate_parts, FEATURE_LABELS,
    FIGURES_DIR, DPI, TOP_N, _FIG_W, _FIG_H,
    _TITLE_FS, _LABEL_FS, _TICK_FS, _ANNOT_FS, _YTICK_FS,
)


def _draw_panel(ax, title, shap_df, color, lang):
    """Render one SHAP importance panel (mirrors generate_shap_figures.main loop)."""
    df_plot = shap_df.sort_values("mean_abs_shap", ascending=True).tail(TOP_N)
    df_plot.index = df_plot.index.map(lambda x: FEATURE_LABELS.get(x, x))
    if lang == "es":
        df_plot.index = df_plot.index.map(_translate_parts)

    bars = ax.barh(df_plot.index, df_plot["mean_abs_shap"], color=color, alpha=0.75)

    max_val = df_plot["mean_abs_shap"].max()
    for bar in bars:
        w = bar.get_width()
        ax.text(w, bar.get_y() + bar.get_height() / 2,
                f" {w:.3f}", va="center", ha="left",
                fontsize=_ANNOT_FS, color=color)
    ax.set_xlim(0, max_val * 1.5)

    xlabel = ("Media |SHAP| (entre folds seleccionados)" if lang == "es"
              else "Mean |SHAP| (across selected folds)")
    ax.set_title(title, fontsize=_TITLE_FS)
    ax.set_xlabel(xlabel, fontsize=_LABEL_FS * 0.9)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=_YTICK_FS)
    ax.tick_params(axis="x", labelsize=_TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_letter(label):
    """Extract the panel letter (a/b/c/d) from a label like 'A. Edad - SVR'."""
    head = label.strip()[0]
    return head.lower() if head.isalpha() else "x"


def main(lang="en"):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    suffix = "_es" if lang == "es" else ""

    print("Loading SHAP data...")
    for combo in COMBINATIONS:
        raw = combo["label_es"] if lang == "es" else combo["label"]
        # strip the "A. " panel-letter prefix -> "{target} - {model}"
        target_model = raw.split(". ", 1)[1] if ". " in raw else raw
        title = (f"Importancia media |SHAP| para {target_model}" if lang == "es"
                 else f"Mean |SHAP| importance for {target_model}")
        shap_df = _compute_shap(combo["dataset"], combo["model"], combo["timestamp"], combo["task"])

        fig = plt.figure(figsize=(_FIG_W / 2, _FIG_H / 2))
        ax = fig.add_subplot(111)
        _draw_panel(ax, title, shap_df, combo["color"], lang)
        fig.tight_layout()

        letter = _panel_letter(combo["label"])
        out_name = f"fig3_shap_{letter}_{combo['dataset']}{suffix}.png"
        save_path = os.path.join(FIGURES_DIR, out_name)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate each SHAP panel as a separate figure")
    parser.add_argument("--lang", choices=["en", "es"], default="en",
                        help="Idioma de los títulos (default: en)")
    args = parser.parse_args()
    main(args.lang)
