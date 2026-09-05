"""Generate the combined SHAP importance figure (2x2 panel, thesis Figure 7).

One panel per reported model/target pair, each showing the top mean |SHAP|
features. The pairs, labels and colors live in `shap_common.COMBINATIONS`.

Usage:
    python -m analysis.scripts.figures.shap_main            # castellano
    python -m analysis.scripts.figures.shap_main --lang en  # inglés
"""
import argparse

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from analysis.scripts.figures import shap_common
from analysis.scripts.figures._style import (
    PRINT_DPI,
    add_lang_argument,
    lang_suffix,
    save_fig,
    use_science_style,
)

use_science_style()


def main(lang="es"):
    print("Loading SHAP data...")
    shap_dfs = []
    for combo in shap_common.COMBINATIONS:
        shap_dfs.append(shap_common.compute_shap(combo))
        print(f"  Done: {combo['label']}")

    fig = plt.figure(figsize=(shap_common.FIG_W, shap_common.FIG_H),
                     constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.12)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[row, col]) for row in (0, 1) for col in (0, 1)]

    for ax, combo, shap_df in zip(axes, shap_common.COMBINATIONS, shap_dfs):
        title = combo["label_es"] if lang == "es" else combo["label"]
        shap_common.draw_shap_panel(
            ax, title, shap_common.top_features(shap_df, lang=lang),
            combo["color"], lang, xlabel="Mean |SHAP|",
        )

    save_fig(fig, f"fig3_shap_importance{lang_suffix(lang)}",
             formats=("png",), dpi=PRINT_DPI)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the combined 2x2 SHAP figure")
    add_lang_argument(parser)
    args = parser.parse_args()
    main(args.lang)
