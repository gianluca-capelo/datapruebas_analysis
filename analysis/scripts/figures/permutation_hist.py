"""Permutation-test figure: null MAE distribution vs. the observed MAE.

The null is recomputed from the model's stored predictions with the procedure
described in `permutation_common`, so with the default settings the figure is
exactly the test behind the reported p-value.

Defaults reproduce the age panel: SVR on tmt_age.

Usage:
    python -m analysis.scripts.figures.permutation_hist
    python -m analysis.scripts.figures.permutation_hist --model Ridge
    python -m analysis.scripts.figures.permutation_hist --target ssrt --dataset tmt_ssrt --unit ms
    python -m analysis.scripts.figures.permutation_hist --n-permutations 10000  # más suave
    python -m analysis.scripts.figures.permutation_hist --aligned  # eje x alineado con el dot plot
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.scripts.figures import permutation_common as pc
from analysis.scripts.figures._style import (
    PRINT_DPI,
    add_lang_argument,
    lang_suffix,
    save_fig,
)
from analysis.scripts.utils import THESIS_RUN, get_latest_regression_timestamp

N_BINS = 18

# Sized for projection in a talk, not for a journal column.
LABEL_FS = 26
TICK_FS = 22
ANNOT_FS = 32
LEGEND_FS = 22
OBSERVED_LW = 4.5
OBSERVED_LS = (0, (4, 2))
NULL_EDGE_LW = 0.9

pc.use_permutation_style(LABEL_FS, TICK_FS)


def plot_null_distribution(null_scores, observed, p_value, lang, unit,
                           bins=N_BINS, figsize=None, axes_rect=None):
    """Histogram of the permuted MAEs with the observed MAE and its p-value marked."""
    # Wide enough for the two legend entries to fit on one row above the axes.
    fig, ax = plt.subplots(figsize=figsize or (13, 6))

    ax.hist(null_scores, bins=bins, color=pc.NULL_GRAY,
            edgecolor=pc.NULL_EDGE, linewidth=NULL_EDGE_LW)
    ax.axvline(observed, color=pc.OBSERVED_GREEN, lw=OBSERVED_LW,
               linestyle=OBSERVED_LS, zorder=3)
    ax.set_xlim(*pc.null_xlim(observed, null_scores.min(), null_scores.max()))

    # The annotation goes on whichever side of the line has more room, with the
    # p-value stacked under the MAE so both read as one block.
    x_min, x_max = ax.get_xlim()
    label_on_left = (observed - x_min) > (x_max - observed)
    offset = 0.015 * (x_max - x_min)
    x_text = observed - offset if label_on_left else observed + offset
    ha = "right" if label_on_left else "left"
    y_top = ax.get_ylim()[1]

    ax.text(x_text, 0.97 * y_top, pc.format_value(observed, lang),
            color=pc.OBSERVED_GREEN, fontsize=ANNOT_FS, ha=ha, va="top")
    ax.text(x_text, 0.85 * y_top, pc.format_p_value(p_value, lang),
            color=pc.OBSERVED_GREEN, fontsize=ANNOT_FS, ha=ha, va="top")

    null_label, observed_label = pc.LEGEND_LABELS[lang]
    ax.legend(
        handles=[
            Patch(facecolor=pc.NULL_GRAY, edgecolor=pc.NULL_EDGE,
                  linewidth=NULL_EDGE_LW, label=null_label),
            Line2D([0], [0], color=pc.OBSERVED_GREEN, lw=OBSERVED_LW,
                   linestyle=OBSERVED_LS, label=observed_label),
        ],
        loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
        fontsize=LEGEND_FS, handlelength=1.2, handletextpad=0.6,
        columnspacing=2.5, borderaxespad=0.0,
    )

    ax.set_xlabel(pc.X_LABELS[lang][unit])
    ax.yaxis.set_visible(False)
    pc.hide_frame(ax)

    if axes_rect is None:
        fig.tight_layout()
    else:
        ax.set_position(axes_rect)
    return fig


def main(timestamp, target, dataset, model, lang, unit, n_permutations, seed, bins,
         aligned=False):
    if timestamp is None:
        timestamp = get_latest_regression_timestamp()
        print(f"Auto-detected latest timestamp: {timestamp}")

    y_true, y_pred = pc.load_predictions(timestamp, target, dataset, model)
    observed, p_value, null_scores = pc.run_permutation(y_true, y_pred, n_permutations, seed)
    print(f"{model} / {target} / {dataset}: MAE = {observed:.4f}, "
          f"null median = {np.median(null_scores):.4f}, p = {p_value:.4f} "
          f"({n_permutations} permutations)")

    fig = plot_null_distribution(
        null_scores, observed, p_value, lang, unit, bins,
        figsize=pc.ALIGNED_FIGSIZE if aligned else None,
        axes_rect=pc.ALIGNED_AXES_RECT if aligned else None,
    )
    aligned_tag = "_aligned" if aligned else ""
    save_fig(fig, f"fig_permutation_{target}_{model}{aligned_tag}{lang_suffix(lang)}",
             dpi=PRINT_DPI, tight=not aligned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the permutation null MAE distribution against the observed MAE"
    )
    parser.add_argument("--timestamp", default=THESIS_RUN,
                        help=f"Regression results timestamp (default: {THESIS_RUN}, "
                             "la corrida de edad). Use 'latest' to auto-detect.")
    parser.add_argument("--target", default="age", help="Target column (default: age)")
    parser.add_argument("--dataset", default="tmt_age", help="Dataset name (default: tmt_age)")
    parser.add_argument("--model", default="SVR", help="Model name (default: SVR)")
    add_lang_argument(parser)
    parser.add_argument("--unit", choices=["years", "ms", "none"], default="years",
                        help="Unidad del MAE en el label del eje x (default: years)")
    pc.add_permutation_arguments(parser)
    parser.add_argument("--bins", type=int, default=N_BINS,
                        help=f"Barras del histograma (default: {N_BINS})")
    parser.add_argument("--aligned", action="store_true",
                        help="Geometría fija compartida con "
                             "'permutation_dotplot --models MODELO --aligned': "
                             "el eje x cae en el mismo píxel en ambas figuras. "
                             "Guarda con sufijo '_aligned'.")
    args = parser.parse_args()

    main(
        timestamp=None if args.timestamp == "latest" else args.timestamp,
        target=args.target,
        dataset=args.dataset,
        model=args.model,
        lang=args.lang,
        unit=None if args.unit == "none" else args.unit,
        n_permutations=args.n_permutations,
        seed=args.seed,
        bins=args.bins,
        aligned=args.aligned,
    )
