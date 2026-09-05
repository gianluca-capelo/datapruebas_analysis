"""Dot plot: observed MAE vs. its own permutation null, one row per model.

Companion to `permutation_hist` (which shows a single model's null in full).
Here every model gets one row: a gray band spanning the 5th percentile to the
maximum of its null, and a dark green dot at the observed MAE.

Caveat worth stating on the slide: each row's band sits wherever that model's
prediction spread puts it, so bands are NOT comparable across rows. Only the
gap between a dot and its own band is meaningful.

Usage:
    python -m analysis.scripts.figures.permutation_dotplot
    python -m analysis.scripts.figures.permutation_dotplot --lang en
    python -m analysis.scripts.figures.permutation_dotplot --keep-dummy

Para una presentación, el histograma de un modelo "achatándose" en su fila. Las
dos figuras comparten eje x y posición de ejes, así que al pasar de slide el eje
no se mueve:
    python -m analysis.scripts.figures.permutation_hist --model SVR --aligned
    python -m analysis.scripts.figures.permutation_dotplot --models SVR --aligned
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.scripts.figures import permutation_common as pc
from analysis.scripts.figures._style import (
    PRINT_DPI,
    add_lang_argument,
    lang_suffix,
    save_fig,
)
from analysis.scripts.utils import THESIS_RUN, get_latest_regression_timestamp

# The baseline model is excluded by default: it has no signal to detect, and its
# near-constant predictions collapse its null onto its observed MAE.
BASELINE_MODEL = "DummyRegressor"

# Sized for projection, matching `permutation_hist`.
LABEL_FS = 26
TICK_FS = 22
ANNOT_FS = 22

pc.use_permutation_style(LABEL_FS, TICK_FS)


def compute_model_stats(timestamp, target, dataset, n_permutations, seed,
                        exclude, include=None) -> pd.DataFrame:
    """Per model: observed MAE, null percentiles and the one-tailed p-value.

    `include`, when given, keeps only those models (and skips permuting the rest).
    """
    df = pc.read_summary(timestamp, target, dataset)
    if include is not None:
        missing = set(include) - set(df["model"])
        if missing:
            raise ValueError(
                f"Model(s) {sorted(missing)} not found for {target}/{dataset} at "
                f"{timestamp}. Available: {', '.join(df['model'])}"
            )
        df = df[df["model"].isin(include)]
    df = df[~df["model"].isin(exclude)]
    if df.empty:
        raise ValueError(f"No models left for {target}/{dataset} after excluding {exclude}")

    rows = []
    for _, row in df.iterrows():
        observed, p_value, null = pc.run_permutation(
            pc.parse_array(row["y_true"]), pc.parse_array(row["y_pred"]),
            n_permutations, seed,
        )
        rows.append({"model": row["model"], "mae_observed": observed,
                     "p_value": p_value, **pc.null_summary(null)})

    # Best model first; the y axis is inverted later so it lands on top.
    return pd.DataFrame(rows).sort_values("mae_observed").reset_index(drop=True)


def plot_dotplot(stats, lang, unit, figsize=None, axes_rect=None, xlim=None):
    fig, ax = plt.subplots(figsize=figsize or (10, 5))
    y = np.arange(len(stats))

    # Faint rule per row, behind everything, to carry the eye from the model
    # name across to its p-value.
    for yi in y:
        ax.axhline(yi, color=pc.ROW_RULE_GRAY, lw=1, zorder=0)

    ax.hlines(y, stats["null_low"], stats["null_max"], color=pc.NULL_GRAY, lw=11, zorder=1)
    ax.plot(stats["mae_observed"], y, "o", color=pc.OBSERVED_GREEN,
            markersize=11, linestyle="none", zorder=3)

    for yi, mae in zip(y, stats["mae_observed"]):
        ax.annotate(pc.format_value(mae, lang), (mae, yi),
                    textcoords="offset points", xytext=(11, 0),
                    va="center", ha="left", color=pc.OBSERVED_GREEN, fontsize=ANNOT_FS)

    ax.set_yticks(y)
    ax.set_yticklabels([pc.model_label(m, lang) for m in stats["model"]])
    ax.invert_yaxis()
    ax.set_ylim(len(stats) - 0.4, -0.6)

    if xlim is None:
        span = stats["null_max"].max() - stats["mae_observed"].min()
        xlim = (stats["mae_observed"].min() - 0.10 * span,
                stats["null_max"].max() + 0.05 * span)
    ax.set_xlim(*xlim)
    ax.set_xlabel(pc.X_LABELS[lang][unit])

    # p-values in their own column just outside the plot area, with a header
    # sitting on the x-label's line.
    p_x = 1.02
    for yi, p_value in zip(y, stats["p_value"]):
        ax.text(p_x, yi, pc.format_p_bare(p_value, lang),
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=ANNOT_FS, clip_on=False)
    ax.text(p_x, -0.14, pc.P_LABEL[lang], transform=ax.transAxes, va="top", ha="left",
            fontsize=LABEL_FS, clip_on=False)

    pc.hide_frame(ax)
    ax.tick_params(axis="y", length=0)

    if axes_rect is None:
        fig.tight_layout()
    else:
        ax.set_position(axes_rect)
    return fig


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
        xlim = pc.null_xlim(best["mae_observed"], best["null_min"], best["null_max"])

    fig = plot_dotplot(
        stats, lang, unit,
        figsize=pc.ALIGNED_FIGSIZE if aligned else None,
        axes_rect=pc.ALIGNED_AXES_RECT if aligned else None,
        xlim=xlim,
    )
    models_tag = f"_{'_'.join(models)}" if models else ""
    aligned_tag = "_aligned" if aligned else ""
    save_fig(fig,
             f"fig_permutation_models_{target}{models_tag}{aligned_tag}{lang_suffix(lang)}",
             formats=("png", "svg"), dpi=PRINT_DPI, tight=not aligned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dot plot of observed MAE vs. permutation null, one row per model"
    )
    parser.add_argument("--timestamp", default=THESIS_RUN,
                        help=f"Regression results timestamp (default: {THESIS_RUN}, "
                             "la corrida de edad). Use 'latest' to auto-detect.")
    parser.add_argument("--target", default="age", help="Target column (default: age)")
    parser.add_argument("--dataset", default="tmt_age", help="Dataset name (default: tmt_age)")
    add_lang_argument(parser)
    parser.add_argument("--unit", choices=["years", "ms", "none"], default="years",
                        help="Unidad del MAE en el label del eje x (default: years)")
    pc.add_permutation_arguments(parser)
    parser.add_argument("--keep-dummy", action="store_true",
                        help=f"Incluir {BASELINE_MODEL} (excluido por defecto)")
    parser.add_argument("--models", nargs="+", default=None, metavar="MODELO",
                        help="Graficar solo estos modelos (ej.: --models SVR). "
                             "Por defecto, todos.")
    parser.add_argument("--aligned", action="store_true",
                        help="Geometría fija y eje x tomado del null del mejor modelo, "
                             "idénticos a 'permutation_hist --aligned'. "
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
