"""Dot plot: observed MAE vs. its own permutation null, one row per model.

Companion to `generate_permutation_figure` (which shows a single model's null in
full). Here every model gets one row: a gray band spanning the 5th percentile to
the maximum of its null — the one-tailed test only cares about the left edge —
and a dark green dot at the observed MAE.

The null distributions are not stored anywhere; they are recomputed from the
stored predictions with the same procedure that produced the published p-values
(y_true shuffled against fixed y_pred, seed=42, 1000 permutations).

Caveat worth stating on the slide: each row's band sits wherever that model's
prediction spread puts it, so bands are NOT comparable across rows. Only the
gap between a dot and its own band is meaningful.

Usage:
    python -m analysis.scripts.generate_permutation_dotplot
    python -m analysis.scripts.generate_permutation_dotplot --lang en
    python -m analysis.scripts.generate_permutation_dotplot --keep-dummy

Para una presentación, el histograma de un modelo "achatándose" en su fila. Las dos
figuras comparten eje x y posición de ejes, así que al pasar de slide el eje no se
mueve:
    python -m analysis.scripts.generate_permutation_figure --model SVR --aligned
    python -m analysis.scripts.generate_permutation_dotplot --models SVR --aligned
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.scripts.generate_permutation_figure import (
    ALIGNED_AXES_RECT,
    ALIGNED_FIGSIZE,
    NULL_GRAY,
    OBSERVED_GREEN,
    P_LABEL,
    _localize,
    _parse_array,
    format_value,
    null_xlim,
)
from analysis.scripts.utils import get_latest_regression_timestamp
from src.config import BASE_DIR, REGRESSION_RESULTS_DIR
from src.model.permutation_tests import permutation_test

FIGURES_DIR = os.path.join(BASE_DIR, "analysis", "figures")
DPI = 300

# The baseline model is excluded by default: it has no signal to detect, and its
# near-constant predictions collapse its null onto its observed MAE.
BASELINE_MODEL = "DummyRegressor"

ROW_RULE_GRAY = "#E5E5E5"

DEFAULT_N_PERMUTATIONS = 1000
DEFAULT_SEED = 42
NULL_BAND_PERCENTILE = 5

plt.style.use(["science", "no-latex"])

# Sized for projection in a talk, matching generate_permutation_figure and the
# --big-fonts scale of generate_segmentation_figure.
LABEL_FS = 26
TICK_FS = 22
ANNOT_FS = 22

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

# Display names: the sklearn class names are too long for a slide.
MODEL_LABELS = {
    "SVR": "SVR",
    "Ridge": "Ridge",
    "ElasticNet": "ElasticNet",
    "Lasso": "Lasso",
    "LinearRegression": {"es": "Regresión lineal", "en": "Linear regression"},
    "RandomForestRegressor": {"es": "Random forest", "en": "Random forest"},
    "XGBRegressor": "XGBoost",
    "DummyRegressor": {"es": "Modelo trivial", "en": "Dummy"},
}


def model_label(model: str, lang: str) -> str:
    label = MODEL_LABELS.get(model, model)
    return label[lang] if isinstance(label, dict) else label


def format_p_bare(p_value: float, lang: str) -> str:
    """`<0.001` at the resolution floor of the test, `0.00x` otherwise."""
    if p_value < 0.001:
        return _localize("<0.001", lang)
    return _localize(f"{p_value:.3f}", lang)


def compute_model_stats(timestamp: str, target: str, dataset: str,
                        n_permutations: int, seed: int,
                        exclude: list[str],
                        include: list[str] | None = None) -> pd.DataFrame:
    """Per model: observed MAE, null percentiles, and the one-tailed p-value.

    `include`, when given, keeps only those models (and skips permuting the rest).
    """
    summary_path = (
        os.path.join(REGRESSION_RESULTS_DIR, timestamp, target, dataset, "summary.csv")
    )
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No summary.csv at {summary_path}")

    df = pd.read_csv(summary_path)
    if include is not None:
        missing = set(include) - set(df["model"])
        if missing:
            raise ValueError(
                f"Model(s) {sorted(missing)} not found in {summary_path}. "
                f"Available: {', '.join(df['model'])}"
            )
        df = df[df["model"].isin(include)]
    df = df[~df["model"].isin(exclude)]
    if df.empty:
        raise ValueError(f"No models left in {summary_path} after excluding {exclude}")

    rows = []
    for _, row in df.iterrows():
        y_true = _parse_array(row["y_true"])
        y_pred = _parse_array(row["y_pred"])
        observed, p_value, null = permutation_test(
            y_true, y_pred, n_permutations=n_permutations, seed=seed, metric="mae",
            return_null_distribution=True,
        )
        rows.append({
            "model": row["model"],
            "mae_observed": observed,
            "null_low": np.percentile(null, NULL_BAND_PERCENTILE),
            "null_median": np.median(null),
            "null_min": null.min(),
            "null_max": null.max(),
            "p_value": p_value,
        })

    # Best model first; the y axis is inverted later so it lands on top.
    return pd.DataFrame(rows).sort_values("mae_observed").reset_index(drop=True)


def plot_dotplot(stats: pd.DataFrame, lang: str, unit: str | None,
                 figsize=None, axes_rect=None, xlim=None):
    fig, ax = plt.subplots(figsize=figsize or (10, 5))

    y = np.arange(len(stats))

    # Faint rule per row, behind everything, to carry the eye from the model
    # name across to its p-value.
    for yi in y:
        ax.axhline(yi, color=ROW_RULE_GRAY, lw=1, zorder=0)

    ax.hlines(y, stats["null_low"], stats["null_max"],
              color=NULL_GRAY, lw=11, zorder=1)
    ax.plot(stats["mae_observed"], y, "o", color=OBSERVED_GREEN,
            markersize=11, linestyle="none", zorder=3)

    for yi, mae in zip(y, stats["mae_observed"]):
        ax.annotate(format_value(mae, lang), (mae, yi),
                    textcoords="offset points", xytext=(11, 0),
                    va="center", ha="left", color=OBSERVED_GREEN, fontsize=ANNOT_FS)

    ax.set_yticks(y)
    ax.set_yticklabels([model_label(m, lang) for m in stats["model"]])
    ax.invert_yaxis()  # best model on top
    ax.set_ylim(len(stats) - 0.4, -0.6)

    if xlim is None:
        span = stats["null_max"].max() - stats["mae_observed"].min()
        xlim = (stats["mae_observed"].min() - 0.10 * span,
                stats["null_max"].max() + 0.05 * span)
    ax.set_xlim(*xlim)
    ax.set_xlabel(X_LABELS[lang][unit])

    # p-values in their own column just outside the plot area, with a header
    # sitting on the x-label's line.
    p_x = 1.02
    for yi, p_value in zip(y, stats["p_value"]):
        ax.text(p_x, yi, format_p_bare(p_value, lang),
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=ANNOT_FS, clip_on=False)
    ax.text(p_x, -0.14, P_LABEL[lang], transform=ax.transAxes, va="top", ha="left",
            fontsize=LABEL_FS, clip_on=False)

    ax.grid(False)
    ax.tick_params(axis="y", length=0)
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
    # See generate_permutation_figure.save_fig: a tight bbox would undo the fixed
    # axes position that --aligned depends on.
    bbox = "tight" if tight else fig.bbox_inches
    fig.savefig(f"{base}.png", dpi=DPI, bbox_inches=bbox)
    fig.savefig(f"{base}.svg", bbox_inches=bbox)
    print(f"Saved -> {base}.png  (+ .svg)")


def main(timestamp, target, dataset, lang, unit, n_permutations, seed, keep_dummy,
         models=None, aligned=False):
    if timestamp is None:
        timestamp = get_latest_regression_timestamp()
        print(f"Auto-detected latest timestamp: {timestamp}")

    exclude = [] if keep_dummy else [BASELINE_MODEL]
    stats = compute_model_stats(timestamp, target, dataset, n_permutations, seed,
                                exclude, include=models)

    print(f"\n{target} / {dataset} — {n_permutations} permutations, seed {seed}")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    xlim = None
    if aligned:
        # Reuse the histogram's x limits, computed from the best model's null —
        # the same model whose histogram precedes this figure on the slide.
        best = stats.iloc[0]
        xlim = null_xlim(best["mae_observed"], best["null_min"], best["null_max"])

    fig = plot_dotplot(
        stats, lang, unit,
        figsize=ALIGNED_FIGSIZE if aligned else None,
        axes_rect=ALIGNED_AXES_RECT if aligned else None,
        xlim=xlim,
    )
    suffix = "_es" if lang == "es" else ""
    models_tag = f"_{'_'.join(models)}" if models else ""
    aligned_tag = "_aligned" if aligned else ""
    save_fig(fig, f"fig_permutation_models_{target}{models_tag}{aligned_tag}{suffix}",
             tight=not aligned)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dot plot of observed MAE vs. permutation null, one row per model"
    )
    parser.add_argument("--timestamp", default="2026-03-07_1213",
                        help="Regression results timestamp (default: the age run). "
                             "Use 'latest' to auto-detect.")
    parser.add_argument("--target", default="age", help="Target column (default: age)")
    parser.add_argument("--dataset", default="tmt_age", help="Dataset name (default: tmt_age)")
    parser.add_argument("--lang", choices=["en", "es"], default="es",
                        help="Idioma de etiquetas (default: es)")
    parser.add_argument("--unit", choices=["years", "ms", "none"], default="years",
                        help="Unidad del MAE en el label del eje x (default: years)")
    parser.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS,
                        help=f"Permutaciones (default: {DEFAULT_N_PERMUTATIONS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Semilla del test (default: {DEFAULT_SEED})")
    parser.add_argument("--keep-dummy", action="store_true",
                        help=f"Incluir {BASELINE_MODEL} (excluido por defecto)")
    parser.add_argument("--models", nargs="+", default=None, metavar="MODELO",
                        help="Graficar solo estos modelos (ej.: --models SVR). "
                             "Por defecto, todos.")
    parser.add_argument("--aligned", action="store_true",
                        help="Geometría fija y eje x tomado del null del mejor modelo, "
                             "idénticos a 'generate_permutation_figure --aligned'. "
                             "Guarda con sufijo '_aligned'.")
    args = parser.parse_args()

    main(
        timestamp=None if args.timestamp == "latest" else args.timestamp,
        target=args.target,
        dataset=args.dataset,
        lang=args.lang,
        unit=None if args.unit == "none" else args.unit,
        n_permutations=args.n_permutations,
        seed=args.seed,
        keep_dummy=args.keep_dummy,
        models=args.models,
        aligned=args.aligned,
    )
