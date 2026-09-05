"""Generate the "No Free Lunch" conceptual figure (stem / lollipop plot).

A schematic — not data-driven — illustrating the No Free Lunch theorem: two
algorithms each excel on different "possible problems", yet averaged over all
problems their performance is identical. To make that point exact, both series
are shifted to share the same mean.

Usage:
    python -m analysis.scripts.figures.no_free_lunch            # castellano
    python -m analysis.scripts.figures.no_free_lunch --lang en  # inglés
    python -m analysis.scripts.figures.no_free_lunch --models   # series con nombres de modelos
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from analysis.scripts.figures._style import (
    PRINT_DPI,
    add_lang_argument,
    lang_suffix,
    save_fig,
    use_slide_style,
)

use_slide_style()

C_A = "#D62728"
C_B = "#1F77B4"
C_STEM = "#1A1A1A"
C_AXIS = "#000000"
C_LABEL = "#000000"

LABEL_FS = 20
SERIES_FS = 20

# Schematic "performance" per problem, hand-tuned to resemble the reference
# figure: A tops the first problems, B the last ones, and both series are built
# with the same mean so they tie when averaged over all problems.
A_RAW = np.array([8.0, 7.5, 4.0, 3.5])
B_RAW = np.array([3.5, 4.0, 7.5, 8.0])
AVG = 5.5


def _labels(lang):
    es = lang == "es"
    return {
        "perf": "Rendimiento" if es else "Performance",
        "prob": "Problemas posibles" if es else "Possible Problems",
        "alg_a": "Algoritmo A" if es else "Algorithm A",
        "alg_b": "Algoritmo B" if es else "Algorithm B",
        "avg": "Promedio" if es else "Average",
    }


def _dots(ax, x, y, color):
    """Filled circles for one algorithm's performance across problems."""
    ax.scatter(x, y, s=150, color=color, edgecolors="white",
               linewidths=0.8, zorder=3)


def main(lang="es", models=False, name_suffix=""):
    lab = _labels(lang)

    # Optionally name the two series after concrete models instead of A/B.
    if models:
        lab["alg_a"] = "Regresión Lineal" if lang == "es" else "Linear Regression"
        lab["alg_b"] = "Random Forest"

    # Shift each series to the common mean so both average to the same line.
    a = A_RAW - A_RAW.mean() + AVG
    b = B_RAW - B_RAW.mean() + AVG

    n = len(a)
    xs = np.arange(1, n + 1)

    x0 = 0.3           # where the y axis stands
    x_right = n + 0.9  # tip of the x-axis arrow
    y_top = max(a.max(), b.max()) + 1.0

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    # --- Axis arrows (custom, no spines/ticks) ---
    arrow = dict(arrowstyle="-|>", color=C_AXIS, lw=2.2)
    ax.annotate("", xy=(x_right, 0), xytext=(x0, 0), arrowprops=arrow)
    ax.annotate("", xy=(x0, y_top), xytext=(x0, 0), arrowprops=arrow)

    # --- Series ---
    # One vertical line per problem (reaching the higher of the two dots),
    # with both algorithms' performance marked on that same line.
    ax.vlines(xs, 0, np.maximum(a, b), color=C_STEM, lw=2.5, zorder=2)
    _dots(ax, xs, a, C_A)
    _dots(ax, xs, b, C_B)

    # --- Series labels ---
    ax.text(1.4, y_top - 0.3, lab["alg_a"], color=C_A,
            ha="center", va="center", fontsize=SERIES_FS, fontweight="bold")
    ax.text(3.6, y_top - 0.3, lab["alg_b"], color=C_B,
            ha="center", va="center", fontsize=SERIES_FS, fontweight="bold")

    # --- Axis titles ---
    # Y title horizontal, to the left of the y-axis, vertically centered.
    ax.text(x0 - 0.45, y_top / 2, lab["perf"], rotation=0,
            ha="right", va="center", color=C_LABEL, fontsize=LABEL_FS)
    ax.text((x0 + x_right) / 2, -0.9, lab["prob"],
            ha="center", va="center", color=C_LABEL, fontsize=LABEL_FS)

    ax.set_xlim(x0 - 0.7, x_right + 0.3)
    ax.set_ylim(-1.4, y_top + 0.3)
    ax.axis("off")
    fig.tight_layout()

    suffix = lang_suffix(lang)
    suffix += "_random_forest" if models else ""
    if name_suffix:
        suffix += f"_{name_suffix.lstrip('_')}"
    save_fig(fig, f"fig_no_free_lunch{suffix}", formats=("png",), dpi=PRINT_DPI)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the No Free Lunch figure")
    add_lang_argument(parser)
    parser.add_argument("--models", action="store_true",
                        help="Etiqueta las series como Regresión Lineal / Random Forest "
                             "en lugar de Algoritmo A / Algoritmo B")
    parser.add_argument("--suffix", default="",
                        help="Sufijo extra para el nombre del archivo de salida")
    args = parser.parse_args()
    main(args.lang, args.models, args.suffix)
