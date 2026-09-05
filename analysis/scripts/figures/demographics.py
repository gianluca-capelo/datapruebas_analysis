"""Generate the sample-demographics figure for the FULL sample (thesis Figure 1).

Four panels — age, gender, education level and nationality — over every subject
seen across the four tasks.

    default        -> one 2x2 panel (fig1_demographics[_es].png)
    --presentation -> four separate PNGs, no titles, slide typography

Usage:
    python -m analysis.scripts.figures.demographics                  # 2x2, castellano
    python -m analysis.scripts.figures.demographics --lang en        # inglés
    python -m analysis.scripts.figures.demographics --presentation   # 4 PNGs para slides
    python -m analysis.scripts.figures.demographics --presentation --stats  # + media/DE
"""
import argparse

import matplotlib.pyplot as plt

from analysis.scripts.figures import demographics_common as dc
from analysis.scripts.figures._style import (
    PRINT_DPI,
    add_lang_argument,
    lang_suffix,
    save_fig,
)

dc.use_demographics_style()

# The paper panel is monochrome; the slide variant gives each demographic its
# own color. Teal keeps the age panel distinct from the modeling-sample variant.
C_PAPER = "#777777"
SLIDE_COLORS = {
    "age": "#2A9D8F",
    "gender": "#E1812C",
    "education": "#3A923A",
    "nationality": "#8A5CA8",
}
C_STAT = "#1A1A1A"

N_FS = 22


def _panels(metadata, labels):
    """(key, counts-or-ages, display map, N) for each of the four panels."""
    return [
        ("age", metadata["age"].dropna(), None, None),
        ("gender", dc.ordered_counts(metadata, "gender_desc", dc.GENDER_ORDER),
         labels["gender"], metadata["gender_desc"].notna().sum()),
        ("education", dc.ordered_counts(metadata, "education_level", dc.EDU_ORDER),
         labels["education"], metadata["education_level"].notna().sum()),
        ("nationality", dc.ordered_counts(metadata, "nationality_clean"),
         labels["nationality"], metadata["nationality_clean"].notna().sum()),
    ]


def _draw_panel(ax, key, data, display_map, labels, color):
    if key == "age":
        dc.draw_age_panel(ax, data, labels, color)
    else:
        dc.draw_category_panel(ax, data, display_map, labels, color)


def _paper_figure(metadata, labels, lang):
    """The 2x2 grid: monochrome bars with the N reported in each panel title."""
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    letters = {"age": "A", "gender": "B", "education": "C", "nationality": "D"}

    for ax, (key, data, display_map, n) in zip(axes.flat, _panels(metadata, labels)):
        _draw_panel(ax, key, data, display_map, labels, C_PAPER)
        n = len(data) if key == "age" else n
        ax.set_title(f"{labels['titles'][letters[key]]} ($N$={n})")

    fig.tight_layout()
    save_fig(fig, f"fig1_demographics{lang_suffix(lang)}",
             formats=("png",), dpi=PRINT_DPI)


def _slide_figures(metadata, labels, lang, stats):
    """One title-less, large-font PNG per panel, N reported inside the age plot."""
    dc.use_demographics_style(slide=True)
    suffix = lang_suffix(lang)

    for key, data, display_map, _n in _panels(metadata, labels):
        fig, ax = plt.subplots(figsize=dc.PANEL_FIGSIZE)
        _draw_panel(ax, key, data, display_map, labels, SLIDE_COLORS[key])

        name = f"fig1_demographics_{key}"
        if key == "age":
            # N counts the subjects actually plotted, i.e. those left after
            # dropping missing or corrupt ages.
            dc.annotate_n(ax, len(data), N_FS)
            if stats:
                dc.annotate_age_stats(ax, data, labels, C_STAT, N_FS)
                name += "_stats"

        fig.tight_layout()
        save_fig(fig, f"{name}{suffix}", formats=("png",), dpi=PRINT_DPI)
        plt.close(fig)


def main(lang="es", presentation=False, stats=False):
    labels = dc.display_labels(lang)

    print("Loading data...")
    metadata = dc.build_metadata(*dc.load_task_analyses())

    if presentation:
        _slide_figures(metadata, labels, lang, stats)
    else:
        _paper_figure(metadata, labels, lang)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the demographics figure for the full sample")
    add_lang_argument(parser)
    parser.add_argument("--presentation", action="store_true",
                        help="Figuras separadas, sin títulos y con fuentes grandes "
                             "para presentación")
    parser.add_argument("--stats", action="store_true",
                        help="Superpone media y ±1 DE en el panel de edad "
                             "(solo con --presentation)")
    args = parser.parse_args()
    main(args.lang, args.presentation, args.stats)
