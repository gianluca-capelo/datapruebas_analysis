"""Permutation figure for the best model of each reference task.

Same reading as `permutation_dotplot` — a gray band for the model's own null
distribution and a dot at the observed MAE — but one panel per target, because
K_mean, accuracy and SSRT live on different scales and a shared x axis would
squash two of the three into a sliver.

The models are the ones reported in bold in Table 3 of the thesis. SSRT is kept
as a counter-example: its best model does not beat its own null, so its dot
lands inside the band and is drawn in red.

Usage:
    python -m analysis.scripts.figures.permutation_best
    python -m analysis.scripts.figures.permutation_best --lang en
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

from analysis.scripts.figures import permutation_common as pc
from analysis.scripts.figures._style import (
    PRINT_DPI,
    add_lang_argument,
    lang_suffix,
    save_fig,
)
from analysis.scripts.utils import THESIS_RUN, THESIS_RUN_K_MEAN

# Presentation sizes, a step above the demographics slide figures.
LABEL_FS = 24
TICK_FS = 22
ANNOT_FS = 26
TITLE_FS = 26

X_LABEL = "MAE"

pc.use_permutation_style(LABEL_FS, TICK_FS)

# One color per reference task, matching the Okabe-Ito palette of the violin
# figure. Used on the task prefix of each title, not on the dots.
C_CDT = "#E69F00"
C_GONOGO = "#9B59B6"
C_SST = "#009E73"

PANELS = [
    {
        "timestamp": THESIS_RUN_K_MEAN,
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
        "timestamp": THESIS_RUN,
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
        "timestamp": THESIS_RUN_K_MEAN,
        "target": "ssrt",
        "dataset": "tmt_ssrt",
        "model": "SVR",
        "prefix": {"es": "Inhibición (SST)", "en": "Inhibition (SST)"},
        "rest": {"es": " — SSRT (SVR)", "en": " — SSRT (SVR)"},
        "prefix_color": C_SST,
        "decimals": 1,
        "color": pc.NOT_SIGNIFICANT_RED,
    },
]


def compute_panel_stats(panel, n_permutations, seed) -> dict:
    """Observed MAE, null band edges and p-value for one target's best model."""
    y_true, y_pred = pc.load_predictions(
        panel["timestamp"], panel["target"], panel["dataset"], panel["model"])
    observed, p_value, null = pc.run_permutation(y_true, y_pred, n_permutations, seed)
    return {**panel, "n_subjects": len(y_true), "mae_observed": observed,
            "p_value": p_value, **pc.null_summary(null)}


def add_two_tone_title(ax, prefix, prefix_color, rest):
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


def plot_best_models(stats, lang):
    fig, axes = plt.subplots(len(stats), 1, figsize=(10, 2.6 * len(stats)))

    for ax, panel in zip(np.atleast_1d(axes), stats):
        color = panel.get("color", pc.OBSERVED_GREEN)

        ax.axhline(0, color=pc.ROW_RULE_GRAY, lw=1, zorder=0)
        ax.hlines(0, panel["null_low"], panel["null_max"],
                  color=pc.NULL_GRAY, lw=13, zorder=1)
        ax.plot([panel["mae_observed"]], [0], "o", color=color, markersize=12, zorder=3)

        # A dot sitting inside the band would have its label overlap the gray;
        # in that case the value goes above the dot instead of beside it.
        inside_band = panel["null_low"] <= panel["mae_observed"] <= panel["null_max"]
        offset, ha, va = ((0, 16), "center", "bottom") if inside_band else ((12, 0), "left", "center")
        ax.annotate(pc.format_value(panel["mae_observed"], lang, panel["decimals"]),
                    (panel["mae_observed"], 0),
                    textcoords="offset points", xytext=offset,
                    va=va, ha=ha, color=color, fontsize=ANNOT_FS, zorder=4)

        low = min(panel["mae_observed"], panel["null_low"])
        span = panel["null_max"] - low
        ax.set_xlim(low - 0.22 * span, panel["null_max"] + 0.06 * span)
        ax.set_ylim(-0.6, 1.0)

        add_two_tone_title(ax, panel["prefix"][lang], panel["prefix_color"],
                           panel["rest"][lang])
        ax.set_xlabel(X_LABEL)

        # p-value in its own column outside the plot area.
        ax.text(1.02, 0, pc.format_p_bare(panel["p_value"], lang),
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=ANNOT_FS, clip_on=False)
        ax.text(1.02, 0.95, pc.P_LABEL[lang], transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=ANNOT_FS, clip_on=False,
                color="#555555")

        ax.set_yticks([])
        pc.hide_frame(ax)

    fig.tight_layout(h_pad=2.5)
    return fig


def main(lang="es", n_permutations=pc.DEFAULT_N_PERMUTATIONS, seed=pc.DEFAULT_SEED):
    stats = [compute_panel_stats(panel, n_permutations, seed) for panel in PANELS]

    print(f"{n_permutations} permutations, seed {seed}")
    for panel in stats:
        print(f"  {panel['target']:<9} {panel['model']:<22} N={panel['n_subjects']:<4} "
              f"MAE={panel['mae_observed']:.4f}  nula[p5]={panel['null_low']:.4f}  "
              f"mediana={panel['null_median']:.4f}  p={panel['p_value']:.4f}")

    fig = plot_best_models(stats, lang)
    save_fig(fig, f"fig_permutation_best_models{lang_suffix(lang)}",
             formats=("png", "svg"), dpi=PRINT_DPI)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Permutation figure for the best model of K_mean, accuracy and SSRT"
    )
    add_lang_argument(parser)
    pc.add_permutation_arguments(parser)
    args = parser.parse_args()
    main(args.lang, args.n_permutations, args.seed)
