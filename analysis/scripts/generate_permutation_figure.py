"""Generate the permutation-test figure: null MAE distribution vs. observed MAE.

The null distribution is not stored anywhere — only the p-value survives into
the consolidated CSVs — so it is recomputed here from the model's stored
predictions using the same procedure as `add_dispersion_metrics`
(y_true shuffled against fixed y_pred, seed=42, 1000 permutations). With those
defaults the figure is exactly the test behind the reported p-value.

Defaults reproduce the age panel: SVR on tmt_age, observed MAE = 6.78 years.

Usage:
    python -m analysis.scripts.generate_permutation_figure
    python -m analysis.scripts.generate_permutation_figure --model Ridge
    python -m analysis.scripts.generate_permutation_figure --target ssrt --dataset tmt_ssrt --unit ms
    python -m analysis.scripts.generate_permutation_figure --n-permutations 10000  # smoother histogram
    python -m analysis.scripts.generate_permutation_figure --aligned  # eje x alineado con el dot plot
"""
import ast
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401 — registers styles
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.scripts.utils import get_latest_regression_timestamp
from src.config import BASE_DIR, REGRESSION_RESULTS_DIR
from src.model.permutation_tests import permutation_test

FIGURES_DIR = os.path.join(BASE_DIR, "analysis", "figures")
DPI = 300

NULL_GRAY = "#B0B0B0"
NULL_EDGE = "#5A5A5A"
OBSERVED_GREEN = "#1B5E20"

N_BINS = 18

# The permutation p-value in the consolidated results was computed with these;
# changing them makes a prettier histogram but a different test.
DEFAULT_N_PERMUTATIONS = 1000
DEFAULT_SEED = 42

plt.style.use(["science", "no-latex"])

# Geometry shared with `generate_permutation_dotplot --aligned`: same canvas and
# same axes rectangle put the x axis on the same pixel in both figures, so the
# histogram can collapse into the dot plot row across two slides. Saving must
# skip bbox_inches="tight" for this to hold — see `save_fig(tight=False)`.
ALIGNED_FIGSIZE = (13, 6)
ALIGNED_AXES_RECT = (0.13, 0.22, 0.74, 0.62)

# Sized for projection in a talk, not for a journal column.
LABEL_FS = 26
TICK_FS = 22
ANNOT_FS = 32
LEGEND_FS = 22
OBSERVED_LW = 4.5
OBSERVED_LS = (0, (4, 2))  # trazo discontinuo, legible en proyección
NULL_EDGE_LW = 0.9

plt.rcParams.update({
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "xtick.top": False,
    "ytick.right": False,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
})

X_LABELS = {
    "es": {"years": "MAE (años)", "ms": "MAE (ms)", None: "MAE"},
    "en": {"years": "MAE (years)", "ms": "MAE (ms)", None: "MAE"},
}

LEGEND_LABELS = {
    "es": ("MAE bajo etiquetas permutadas", "MAE observado en mejor modelo"),
    "en": ("MAE under permuted labels", "Observed MAE"),
}

P_LABEL = {"es": "p valor", "en": "p-value"}


def _parse_array(array_str) -> np.ndarray:
    """Parse the stringified lists stored in summary.csv."""
    if isinstance(array_str, str):
        return np.array(ast.literal_eval(array_str))
    return np.asarray(array_str)


def load_predictions(timestamp: str, target: str, dataset: str, model: str):
    """Return (y_true, y_pred) for one model from a regression summary.csv."""
    summary_path = Path(REGRESSION_RESULTS_DIR) / timestamp / target / dataset / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.csv at {summary_path}")

    df = pd.read_csv(summary_path)
    row = df[df["model"] == model]
    if row.empty:
        raise ValueError(
            f"Model '{model}' not found in {summary_path}. "
            f"Available: {', '.join(df['model'])}"
        )

    row = row.iloc[0]
    return _parse_array(row["y_true"]), _parse_array(row["y_pred"])


def _localize(text: str, lang: str) -> str:
    return text.replace(".", ",") if lang == "es" else text


def format_value(value: float, lang: str) -> str:
    """Two decimals, with the decimal separator of the target language."""
    return _localize(f"{value:.2f}", lang)


def format_p_value(p_value: float, lang: str) -> str:
    """`p valor < 0.001` below the resolution of the test, `p valor = x.xxx` otherwise.

    With 1000 permutations the smallest attainable p is 1/1001 ~ 0.001, so the
    threshold is the floor of the test rather than an arbitrary cutoff.
    """
    label = P_LABEL[lang]
    if p_value < 0.001:
        return _localize(f"{label} < 0.001", lang)
    return _localize(f"{label} = {p_value:.3f}", lang)


def null_xlim(observed: float, null_min: float, null_max: float,
              margin_frac: float = 0.08) -> tuple[float, float]:
    """x limits keeping both the observed value and the whole null in frame.

    The observed MAE normally sits far to the left of the null. Shared with the
    dot plot so a single-model row can reuse the histogram's axis.
    """
    low = min(observed, null_min)
    high = max(observed, null_max)
    margin = margin_frac * (high - low)
    return low - margin, high + margin


def plot_null_distribution(null_scores: np.ndarray, observed: float, p_value: float,
                           lang: str, unit: str | None, bins: int = N_BINS,
                           figsize=None, axes_rect=None):
    """Histogram of the permuted MAEs with the observed MAE and its p-value marked."""
    # Wide enough for the two legend entries to fit on one row above the axes.
    fig, ax = plt.subplots(figsize=figsize or (13, 6))

    ax.hist(null_scores, bins=bins, color=NULL_GRAY,
            edgecolor=NULL_EDGE, linewidth=NULL_EDGE_LW)
    ax.axvline(observed, color=OBSERVED_GREEN, lw=OBSERVED_LW,
               linestyle=OBSERVED_LS, zorder=3)

    ax.set_xlim(*null_xlim(observed, null_scores.min(), null_scores.max()))

    # Labels go on whichever side of the line has more room: the observed MAE
    # first, the p-value stacked right under it so both read as one annotation.
    x_min, x_max = ax.get_xlim()
    space_left = observed - x_min
    label_on_left = space_left > (x_max - observed)
    offset = 0.015 * (x_max - x_min)
    x_text = observed - offset if label_on_left else observed + offset
    ha = "right" if label_on_left else "left"
    y_top = ax.get_ylim()[1]

    ax.text(x_text, 0.97 * y_top, format_value(observed, lang),
            color=OBSERVED_GREEN, fontsize=ANNOT_FS, ha=ha, va="top")
    ax.text(x_text, 0.85 * y_top, format_p_value(p_value, lang),
            color=OBSERVED_GREEN, fontsize=ANNOT_FS, ha=ha, va="top")

    # Legend sits above the axes, left-aligned, as a single row: outside the plot
    # area but keeping the figure compact.
    null_label, observed_label = LEGEND_LABELS[lang]
    ax.legend(
        handles=[
            Patch(facecolor=NULL_GRAY, edgecolor=NULL_EDGE,
                  linewidth=NULL_EDGE_LW, label=null_label),
            Line2D([0], [0], color=OBSERVED_GREEN, lw=OBSERVED_LW,
                   linestyle=OBSERVED_LS, label=observed_label),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FS,
        handlelength=1.2,
        handletextpad=0.6,
        columnspacing=2.5,
        borderaxespad=0.0,
    )

    ax.set_xlabel(X_LABELS[lang][unit])
    ax.yaxis.set_visible(False)
    ax.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    if axes_rect is None:
        fig.tight_layout()
    else:
        ax.set_position(axes_rect)
    return fig


def save_fig(fig, name: str, tight: bool = True):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    base = os.path.join(FIGURES_DIR, name)
    # A tight bounding box crops to whatever ink each figure happens to have, so
    # it must be off when the axes position is what has to be preserved. Passing
    # None is not enough: it falls back to scienceplots' savefig.bbox = tight,
    # hence the explicit full-canvas Bbox.
    bbox = "tight" if tight else fig.bbox_inches
    fig.savefig(f"{base}.png", dpi=DPI, bbox_inches=bbox)
    fig.savefig(f"{base}.pdf", bbox_inches=bbox)  # vectorial
    print(f"Saved -> {base}.png  (+ .pdf)")


def main(timestamp, target, dataset, model, lang, unit, n_permutations, seed, bins,
         aligned=False):
    if timestamp is None:
        timestamp = get_latest_regression_timestamp()
        print(f"Auto-detected latest timestamp: {timestamp}")

    y_true, y_pred = load_predictions(timestamp, target, dataset, model)
    observed, p_value, null_scores = permutation_test(
        y_true, y_pred, n_permutations=n_permutations, seed=seed, metric="mae",
        return_null_distribution=True,
    )
    print(f"{model} / {target} / {dataset}: MAE = {observed:.4f}, "
          f"null median = {np.median(null_scores):.4f}, p = {p_value:.4f} "
          f"({n_permutations} permutations)")

    fig = plot_null_distribution(
        null_scores, observed, p_value, lang, unit, bins,
        figsize=ALIGNED_FIGSIZE if aligned else None,
        axes_rect=ALIGNED_AXES_RECT if aligned else None,
    )
    suffix = "_es" if lang == "es" else ""
    aligned_tag = "_aligned" if aligned else ""
    save_fig(fig, f"fig_permutation_{target}_{model}{aligned_tag}{suffix}", tight=not aligned)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot the permutation null MAE distribution against the observed MAE"
    )
    parser.add_argument("--timestamp", default="2026-03-07_1213",
                        help="Regression results timestamp (default: the age run). "
                             "Use 'latest' to auto-detect.")
    parser.add_argument("--target", default="age", help="Target column (default: age)")
    parser.add_argument("--dataset", default="tmt_age", help="Dataset name (default: tmt_age)")
    parser.add_argument("--model", default="SVR", help="Model name (default: SVR)")
    parser.add_argument("--lang", choices=["en", "es"], default="es",
                        help="Idioma del eje y del número (default: es)")
    parser.add_argument("--unit", choices=["years", "ms", "none"], default="years",
                        help="Unidad del MAE en el label del eje x (default: years)")
    parser.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS,
                        help=f"Permutaciones (default: {DEFAULT_N_PERMUTATIONS}, "
                             "el valor con el que se reportó el p-valor)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Semilla del test (default: {DEFAULT_SEED})")
    parser.add_argument("--bins", type=int, default=N_BINS,
                        help=f"Barras del histograma (default: {N_BINS})")
    parser.add_argument("--aligned", action="store_true",
                        help="Geometría fija compartida con "
                             "'generate_permutation_dotplot --models MODELO --aligned': "
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
