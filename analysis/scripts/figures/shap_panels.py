"""Generate each panel of the main SHAP figure as a separate image.

Same model/dataset pairs as `shap_main`, but one standalone image per panel so
they can be stacked vertically in the thesis, larger and more legible.

Usage:
    python -m analysis.scripts.figures.shap_panels            # castellano
    python -m analysis.scripts.figures.shap_panels --lang en  # inglés
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

use_science_style()


def _panel_letter(label: str) -> str:
    """Panel letter (a/b/c/d) of a label like 'A. Edad - SVR'."""
    head = label.strip()[0]
    return head.lower() if head.isalpha() else "x"


def _panel_name(label: str) -> str:
    """Drop the 'A. ' panel-letter prefix, leaving '{target} - {model}'."""
    return label.split(". ", 1)[1] if ". " in label else label


def main(lang="es"):
    print("Loading SHAP data...")
    for combo in shap_common.COMBINATIONS:
        label = combo["label_es"] if lang == "es" else combo["label"]
        shap_df = shap_common.compute_shap(combo)

        fig, ax = plt.subplots(figsize=(shap_common.FIG_W / 2, shap_common.FIG_H / 2))
        shap_common.draw_shap_panel(
            ax, shap_common.PANEL_TITLE[lang].format(name=_panel_name(label)),
            shap_common.top_features(shap_df, lang=lang), combo["color"], lang,
        )
        fig.tight_layout()

        letter = _panel_letter(combo["label"])
        save_fig(fig, f"fig3_shap_{letter}_{combo['dataset']}{lang_suffix(lang)}",
                 formats=("png",), dpi=PRINT_DPI)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate each SHAP panel as a separate figure")
    add_lang_argument(parser)
    args = parser.parse_args()
    main(args.lang)
