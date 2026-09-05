"""Plot a single cTMT Part B trial with the cursor trail colored by segmentation.

Each cursor point is colored by its movement state (hesitation / search /
travel), reproducing the Methods segmentation figure (thesis Figure 2A).

Usage:
    python -m analysis.scripts.figures.segmentation              # castellano
    python -m analysis.scripts.figures.segmentation --lang en    # inglés
    python -m analysis.scripts.figures.segmentation --big-fonts  # fuentes de slide
"""
import argparse

import matplotlib.pyplot as plt

from neurotask.tmt.segmentation.segmentation import classify_cursor_positions_with_hesitation
from src import config

from analysis.scripts.figures import trial_data
from analysis.scripts.figures._style import (
    TRIAL_DPI,
    TRIAL_FONTS,
    add_lang_argument,
    lang_suffix,
    save_fig,
    use_science_style,
)

use_science_style()

# Okabe-Ito segmentation colors. Keys stay in English because that is what
# classify_cursor_positions_with_hesitation returns.
SEG_COLORS = {
    "Hesitation": "#FF7F0E",
    "Search": "#4CAF50",
    "Travel": "#1F77B4",
}
C_UNKNOWN = "#888888"

SEG_NAMES = {
    "es": {"Hesitation": "Duda", "Search": "Búsqueda", "Travel": "Viaje"},
    "en": {"Hesitation": "Hesitation", "Search": "Search", "Travel": "Travel"},
}
LEGEND_TITLE = {"es": "Segmentación", "en": "Segmentation"}


def main(lang="es", subject_id=None, trial_id=None, big_fonts=False):
    fonts = TRIAL_FONTS.scaled() if big_fonts else TRIAL_FONTS

    row, subject, trial = trial_data.load_trial(subject_id, trial_id)
    segmentation = classify_cursor_positions_with_hesitation(
        trial, subject.target_radius, config.TARGET_RADIUS_MULTIPLIER,
        row["speed_threshold"],
    )
    point_colors = [SEG_COLORS.get(label, C_UNKNOWN) for label, _ in segmentation]

    _cursor_trail, cursor_x, cursor_y = trial_data.cursor_coordinates(trial)

    fig, ax = plt.subplots(figsize=(7, 7))
    trial_data.draw_faint_trail(ax, cursor_x, cursor_y)
    ax.scatter(cursor_x, cursor_y, c=point_colors, s=20, alpha=0.9,
               linewidths=0, zorder=4)
    trial_data.draw_trial_background(ax, trial, subject, cursor_x, cursor_y, lang, fonts)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=SEG_NAMES[lang][name],
                   markerfacecolor=SEG_COLORS[name], markersize=11)
        for name in SEG_COLORS
    ]
    ax.legend(**trial_data.legend_kwargs(
        fonts, outside=big_fonts, handles=handles, title=LEGEND_TITLE[lang],
        frameon=True, title_fontsize=fonts.legend,
    ))

    big = "_big" if big_fonts else ""
    save_fig(fig, f"fig2a_tmt_segmentation{big}{lang_suffix(lang)}", dpi=TRIAL_DPI)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a cTMT Part B trial colored by segmentation state")
    add_lang_argument(parser)
    parser.add_argument("--subject", default=None, help="Override subject_id")
    parser.add_argument("--trial", default=None, help="Override trial_id")
    parser.add_argument("--big-fonts", action="store_true",
                        help="Agranda las fuentes y saca la leyenda del plot (slides)")
    args = parser.parse_args()
    main(args.lang, args.subject, args.trial, args.big_fonts)
