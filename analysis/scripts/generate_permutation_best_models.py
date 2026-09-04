"""Permutation figure for the best model of each non-age target.

Same reading as `generate_permutation_dotplot` — a gray band for the model's own
null distribution and a dark green dot at the observed MAE — but one panel per
target, because K_mean, accuracy and the c coefficient live on different scales
and a shared x axis would squash two of the three into a sliver.

The models are the ones reported in bold in Table 3 of the thesis (and used for
the SHAP panels of Figure 7). For the c coefficient the thesis bolds both Ridge
and Lasso; only Ridge is plotted here.

Null distributions are recomputed from the stored predictions with the same
procedure behind the published p-values (y_true shuffled against fixed y_pred,
seed=42, 1000 permutations).

Usage:
    python -m analysis.scripts.generate_permutation_best_models
    python -m analysis.scripts.generate_permutation_best_models --lang en
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

from analysis.scripts.generate_permutation_dotplot import (
    NULL_BAND_PERCENTILE,
    ROW_RULE_GRAY,
    format_p_bare,
)
from analysis.scripts.generate_permutation_figure import (
    NULL_GRAY,
    OBSERVED_GREEN,
    P_LABEL,
    _localize,
    load_predictions,
)
from src.config import BASE_DIR
from src.model.permutation_tests import permutation_test

FIGURES_DIR = os.path.join(BASE_DIR, "analysis", "figures")
DPI = 300

DEFAULT_N_PERMUTATIONS = 1000
DEFAULT_SEED = 42

# One color per reference task, matching the Okabe-Ito palette of
# generate_violin_figures (and Figure 6 of the thesis). Used on the task prefix
# of each title, not on the dots.
C_CDT = "#E69F00"     # ámbar
C_GONOGO = "#9B59B6"  # violeta
C_SST = "#009E73"     # teal

# The SST panel keeps red: its model failed to beat its null, and the color
# carries that on its own.
NOT_SIGNIFICANT_RED = "#B71C1C"

# One entry per panel. Timestamps differ because the March run was split across
# two days: K_mean landed in the first folder, accuracy and c in the second.
PANELS = [
    {
        "timestamp": "2026-03-06_2028",
        "target": "K_mean",
        "dataset": "tmt_k_mean",
        "model": "RandomForestRegressor",
        "prefix": {"es": "Memoria (CDT)", "en": "Memory (CDT)"},
        "rest": {"es": " — $K_{mean}$ (Random forest)",
                 "en": " — $K_{mean}$ (Random forest)"},
        "prefix_color": C_CDT,
        "decimals": 2,
    },
    {
        "timestamp": "2026-03-07_1213",
        "target": "accuracy",
        "dataset": "tmt_accuracy",
        "model": "SVR",
        "prefix": {"es": "Inhibición (GNG)", "en": "Inhibition (GNG)"},
        "rest": {"es": " — Accuracy (SVR)", "en": " — Accuracy (SVR)"},
        "prefix_color": C_GONOGO,
        # A proportion near 0.08 needs a third decimal to say anything.
        "decimals": 3,
    },
    {
        # Counter-example: no model reaches significance for SSRT, so the best
        # of them lands inside its own null. Marked in red.
        "timestamp": "2026-03-06_2028",
        "target": "ssrt",
        "dataset": "tmt_ssrt",
        "model": "SVR",
        "prefix": {"es": "Inhibición (SST)", "en": "Inhibition (SST)"},
        "rest": {"es": " — SSRT (SVR)", "en": " — SSRT (SVR)"},
        "prefix_color": C_SST,
        "decimals": 1,
        "color": NOT_SIGNIFICANT_RED,
    },
]


def format_mae(value: float, decimals: int, lang: str) -> str:
    return _localize(f"{value:.{decimals}f}", lang)


def add_two_tone_title(ax, prefix: str, prefix_color: str, rest: str):
    """Title whose task prefix is colored and whose remainder stays black.

    set_title() takes a single color, so the two runs are packed side by side
    and anchored above the axes.
    """
    box = HPacker(
        children=[
            TextArea(prefix, textprops=dict(color=prefix_color, fontsize=TITLE_FS)),
            TextArea(rest, textprops=dict(color="black", fontsize=TITLE_FS)),
        ],
        align="baseline", pad=0, sep=0,
    )
    ax.add_artist(AnchoredOffsetbox(
        loc="lower left", child=box, pad=0, borderpad=0, frameon=False,
        bbox_to_anchor=(0.0, 1.02), bbox_transform=ax.transAxes,
    ))

plt.style.use(["science", "no-latex"])

# Presentation sizes, a step above generate_demographics_figure_368.
LABEL_FS = 24
TICK_FS = 22
ANNOT_FS = 26
TITLE_FS = 26

X_LABEL = "MAE"

plt.rcParams.update({
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "xtick.top": False,
    "ytick.right": False,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
})


def compute_panel_stats(panel: dict, n_permutations: int, seed: int) -> dict:
    """Observed MAE, null band edges and p-value for one target's best model."""
    y_true, y_pred = load_predictions(
        panel["timestamp"], panel["target"], panel["dataset"], panel["model"]
    )
    observed, p_value, null = permutation_test(
        y_true, y_pred, n_permutations=n_permutations, seed=seed, metric="mae",
        return_null_distribution=True,
    )
    return {
        **panel,
        "n_subjects": len(y_true),
        "mae_observed": observed,
        "null_low": np.percentile(null, NULL_BAND_PERCENTILE),
        "null_median": float(np.median(null)),
        "null_max": null.max(),
        "p_value": p_value,
    }


def plot_best_models(stats: list[dict], lang: str):
    fig, axes = plt.subplots(len(stats), 1, figsize=(10, 2.6 * len(stats)))

    for ax, s in zip(np.atleast_1d(axes), stats):
        color = s.get("color", OBSERVED_GREEN)

        ax.axhline(0, color=ROW_RULE_GRAY, lw=1, zorder=0)
        ax.hlines(0, s["null_low"], s["null_max"], color=NULL_GRAY, lw=13, zorder=1)
        ax.plot([s["mae_observed"]], [0], "o", color=color, markersize=12, zorder=3)

        # A dot sitting inside the band would have its label overlap the gray;
        # in that case the value goes above the dot instead of beside it.
        inside_band = s["null_low"] <= s["mae_observed"] <= s["null_max"]
        offset, ha, va = ((0, 16), "center", "bottom") if inside_band else ((12, 0), "left", "center")
        ax.annotate(format_mae(s["mae_observed"], s["decimals"], lang),
                    (s["mae_observed"], 0),
                    textcoords="offset points", xytext=offset,
                    va=va, ha=ha, color=color, fontsize=ANNOT_FS, zorder=4)

        low = min(s["mae_observed"], s["null_low"])
        span = s["null_max"] - low
        ax.set_xlim(low - 0.22 * span, s["null_max"] + 0.06 * span)
        ax.set_ylim(-0.6, 1.0)

        add_two_tone_title(ax, s["prefix"][lang], s["prefix_color"], s["rest"][lang])
        ax.set_xlabel(X_LABEL)

        # p-value in its own column outside the plot area.
        ax.text(1.02, 0, format_p_bare(s["p_value"], lang),
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=ANNOT_FS, clip_on=False)
        ax.text(1.02, 0.95, P_LABEL[lang], transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=ANNOT_FS, clip_on=False,
                color="#555555")

        ax.set_yticks([])
        ax.grid(False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    fig.tight_layout(h_pad=2.5)
    return fig


def save_fig(fig, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    base = os.path.join(FIGURES_DIR, name)
    fig.savefig(f"{base}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    print(f"Saved -> {base}.png  (+ .svg)")


def main(lang: str, n_permutations: int, seed: int):
    stats = [compute_panel_stats(p, n_permutations, seed) for p in PANELS]

    print(f"{n_permutations} permutations, seed {seed}")
    for s in stats:
        print(f"  {s['target']:<9} {s['model']:<22} N={s['n_subjects']:<4} "
              f"MAE={s['mae_observed']:.4f}  nula[p5]={s['null_low']:.4f}  "
              f"mediana={s['null_median']:.4f}  p={s['p_value']:.4f}")

    fig = plot_best_models(stats, lang)
    suffix = "_es" if lang == "es" else ""
    save_fig(fig, f"fig_permutation_best_models{suffix}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Permutation figure for the best model of K_mean, accuracy and c"
    )
    parser.add_argument("--lang", choices=["en", "es"], default="es",
                        help="Idioma de etiquetas (default: es)")
    parser.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS,
                        help=f"Permutaciones (default: {DEFAULT_N_PERMUTATIONS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Semilla del test (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    main(args.lang, args.n_permutations, args.seed)
