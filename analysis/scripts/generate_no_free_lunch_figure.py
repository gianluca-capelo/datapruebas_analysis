"""Generate the "No Free Lunch" conceptual figure (stem / lollipop plot).

A schematic — not data-driven — illustrating the No Free Lunch theorem: two
algorithms (A and B) each excel on different "possible problems", yet averaged
over all problems their performance is identical (the dashed "Average" line).

To make that point exact, A and B are shifted to share the *same* mean, which is
where the Average line sits.

Outputs one PNG:
    fig_no_free_lunch[_en].png

Usage:
    python -m analysis.scripts.generate_no_free_lunch_figure            # castellano
    python -m analysis.scripts.generate_no_free_lunch_figure --lang en  # inglés
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 — registers styles

from src.config import BASE_DIR

FIGURES_DIR = os.path.join(BASE_DIR, "analysis", "figures")
DPI = 300

# ---------------------------------------------------------------------------
# Typography — same big-font look as generate_demographics_figure_368.py
# (scienceplots "science" style + presentation rcParams)
# ---------------------------------------------------------------------------
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "axes.labelsize":  20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

# ---------------------------------------------------------------------------
# Palette — mirrors the reference schematic
# ---------------------------------------------------------------------------
C_A = "#D62728"      # Algorithm A — red
C_B = "#1F77B4"      # Algorithm B — blue
C_STEM = "#1A1A1A"   # the vertical problem line (black, as in the original)
C_AXIS = "#000000"   # axis arrows — black
C_LABEL = "#000000"  # axis titles — black
C_AVG = "#1F3A5F"    # average line — navy, dashed

# Font sizes (match the big-font demographics figures)
LABEL_FS = 20
SERIES_FS = 20

# ---------------------------------------------------------------------------
# Schematic "performance" per problem (hand-tuned to resemble the reference).
# A dominates the left problems, B the middle/right ones.
# ---------------------------------------------------------------------------
# Non-alternating winners: A tops the first problems, B the last ones. Built
# with the same mean so, averaged over all problems, both algorithms tie.
A_RAW = np.array([8.0, 7.5, 4.0, 3.5])
B_RAW = np.array([3.5, 4.0, 7.5, 8.0])
AVG = 5.5  # common mean both series are shifted to


def _labels(lang):
    es = lang != "en"
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
    os.makedirs(FIGURES_DIR, exist_ok=True)
    lab = _labels(lang)

    # Optionally name the two series after concrete models instead of A/B.
    if models:
        lab["alg_a"] = "Regresión Lineal" if lang != "en" else "Linear Regression"
        lab["alg_b"] = "Random Forest"

    # Shift each series to the common mean so both average to the same line.
    a = A_RAW - A_RAW.mean() + AVG
    b = B_RAW - B_RAW.mean() + AVG

    n = len(a)
    xs = np.arange(1, n + 1)

    x0 = 0.3                 # y-axis position
    x_right = n + 0.9        # x-axis arrow tip
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

    suffix = "_en" if lang == "en" else ""
    suffix += "_random_forest" if models else ""
    if name_suffix:
        suffix += f"_{name_suffix.lstrip('_')}"
    path = os.path.join(FIGURES_DIR, f"fig_no_free_lunch{suffix}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate the No Free Lunch figure")
    parser.add_argument("--lang", choices=["en", "es"], default="es",
                        help="Idioma de las etiquetas (default: es)")
    parser.add_argument("--models", action="store_true",
                        help="Etiqueta las series como Regresión Lineal / Random Forest "
                             "en lugar de Algoritmo A / Algoritmo B")
    parser.add_argument("--suffix", default="",
                        help="Sufijo extra para el nombre del archivo de salida")
    args = parser.parse_args()
    main(args.lang, args.models, args.suffix)
