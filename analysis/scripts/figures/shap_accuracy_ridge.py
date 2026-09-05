"""Generate the supplementary single-panel SHAP figure (Accuracy — Ridge).

Same as panel C of the main SHAP figure (Accuracy) but with Ridge instead of
SVR, for thesis Supplementary Figure 1 — comparable to the c coefficient model,
which also uses Ridge.

Usage:
    python -m analysis.scripts.figures.shap_accuracy_ridge            # castellano
    python -m analysis.scripts.figures.shap_accuracy_ridge --lang en  # inglés
"""
import argparse

import matplotlib.pyplot as plt

from analysis.scripts.figures import shap_common
from analysis.scripts.figures._style import (
    PRINT_DPI,
    add_lang_argument,
    lang_suffix,
    save_fig,
    use_science_style,
)
from analysis.scripts.utils import THESIS_RUN

use_science_style()

COMBINATION = {
    "dataset": "tmt_accuracy",
    "model": "Ridge",
    "task": "regression",
    "timestamp": THESIS_RUN,
    "color": shap_common.C_PURPLE,
}
TITLE = "Accuracy - Ridge"


def main(lang="es"):
    print("Loading SHAP data...")
    shap_df = shap_common.compute_shap(COMBINATION)

    fig, ax = plt.subplots(figsize=(shap_common.FIG_W / 2, shap_common.FIG_H / 2))
    shap_common.draw_shap_panel(
        ax, shap_common.PANEL_TITLE[lang].format(name=TITLE),
        shap_common.top_features(shap_df, lang=lang), COMBINATION["color"], lang,
    )
    fig.tight_layout()

    save_fig(fig, f"figS1_shap_accuracy_ridge{lang_suffix(lang)}",
             formats=("png",), dpi=PRINT_DPI)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the supplementary Accuracy-Ridge SHAP figure")
    add_lang_argument(parser)
    args = parser.parse_args()
    main(args.lang)
