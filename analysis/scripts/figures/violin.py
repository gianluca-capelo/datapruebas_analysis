"""Generate the performance-distributions violin figure (six panels).

Panels (2x3 grid):
    A. cTMT completion time    B. cTMT completion rate   C. Change Detection Task
    D. Stop Signal Task        E. Go/No-Go — c coeff     F. Go/No-Go — Accuracy

Usage:
    python -m analysis.scripts.figures.violin            # castellano
    python -m analysis.scripts.figures.violin --lang en  # inglés
"""
import argparse

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.scripts.figures._style import (
    PANEL_RCPARAMS,
    PRINT_DPI,
    add_lang_argument,
    lang_suffix,
    save_fig,
    use_science_style,
)
from analysis.scripts.figures.demographics_common import load_task_analyses

TITLE_FS = 13.5
LABEL_FS = 16
TICK_FS = 14
LEGEND_FS = 12

use_science_style()
plt.rcParams.update({
    "axes.titlesize": TITLE_FS,
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
    **PANEL_RCPARAMS,
})

# Okabe-Ito task colors, shared with the SHAP figures.
C_TMT = "#0072B2"
C_CDT = "#E69F00"
C_SST = "#009E73"
C_GONOGO = "#9B59B6"
C_GRAY = "#888888"

# The notebook used cut=0 (KDE truncated flat at the data range); the thesis
# figure used seaborn's default cut=2, which lets the violins close with
# tapered tails. The default is kept so the thesis figure is reproduced.
VIOLIN_KW = dict(inner="box", linewidth=0.8, saturation=0.85)

PART_MAP = {"PART_A": "Part A", "PART_B": "Part B"}
SET_SIZE_MAP = {"K_4": "$K_4$", "K_6": "$K_6$"}


def _labels(lang):
    """(titles, ylabels, part names) for the requested language."""
    es = lang == "es"
    titles = {
        "A": "A. Tiempo de completitud (cTMT)" if es else "A. TMT Completion Time",
        "B": "B. Ensayos válidos (cTMT)" if es else "B. TMT Completion Rate",
        "C": "C. Change Detection Task",
        "D": "D. Stop Signal Task",
        "E": "E. Go/No-Go — coeficiente $c$" if es else "E. Go/No-Go — $c$ coefficient",
        "F": "F. Go/No-Go — Accuracy",
    }
    ylabels = {
        "A": "Tiempo medio (s)" if es else "Mean time (s)",
        "B": "Ensayos válidos (%)" if es else "Valid trials (%)",
        "C": "$K$ de Cowan" if es else "Cowan's $K$",
        "D": "SSRT (ms)",
        "E": "coeficiente $c$" if es else "$c$ coefficient",
        "F": "Accuracy",
    }
    parts = ["Parte A", "Parte B"] if es else ["Part A", "Part B"]
    return titles, ylabels, parts


def prepare_data(df_tmt, df_cdt):
    """Derived frames for panels A (mean time), B (valid rate) and C (Cowan's K)."""
    df_part = df_tmt.copy()
    df_part["part"] = df_part["trial_type"].str.extract(r"(PART_[AB])")[0]
    df_part = df_part.dropna(subset=["part"])

    df_valid = df_part[df_part["is_valid"].eq(True)].copy()
    df_valid["time_sec"] = df_valid["non_cut_rt"] / 1000

    # Subjects with at least one valid trial in BOTH Part A and Part B.
    sids_both = (set(df_valid[df_valid["part"] == "PART_A"]["subject_id"])
                 & set(df_valid[df_valid["part"] == "PART_B"]["subject_id"]))
    print(f"TMT N (valid in both parts): {len(sids_both)}")

    tmt_time = (
        df_valid[df_valid["subject_id"].isin(sids_both)]
        .groupby(["subject_id", "part"])["time_sec"].mean().reset_index()
    )
    tmt_time["Part"] = tmt_time["part"].map(PART_MAP)

    tmt_pct = (
        df_part[df_part["subject_id"].isin(sids_both)]
        .groupby(["subject_id", "part"])
        .agg(total=("is_valid", "count"), valid=("is_valid", "sum"))
        .reset_index()
    )
    tmt_pct["pct_valid"] = 100 * tmt_pct["valid"] / tmt_pct["total"]
    tmt_pct["Part"] = tmt_pct["part"].map(PART_MAP)

    cdt_long = pd.melt(
        df_cdt[["subject_id", "K_4", "K_6"]], id_vars="subject_id",
        value_vars=["K_4", "K_6"], var_name="Set Size", value_name="K",
    )
    cdt_long["Set Size"] = cdt_long["Set Size"].map(SET_SIZE_MAP)

    return tmt_time, tmt_pct, cdt_long


def _violin_panel(ax, data, y, color, title, ylabel, n, x=None, xticklabels=None):
    sns.violinplot(data=data, x=x, y=y, ax=ax, color=color, **VIOLIN_KW)
    ax.set_title(f"{title}\n($N$={n})")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    if xticklabels is None:
        ax.set_xticks([])
    else:
        ax.set_xticklabels(xticklabels, fontsize=TICK_FS)


def _apply_style(axes):
    """Re-enforce title and axis label sizes, which seaborn overrides."""
    for ax in np.atleast_1d(axes).flat:
        ax.title.set_fontsize(TITLE_FS)
        ax.xaxis.label.set_fontsize(LABEL_FS)
        ax.yaxis.label.set_fontsize(LABEL_FS)


def main(lang="es"):
    titles, ylabels, parts = _labels(lang)

    print("Loading data...")
    df_tmt, df_sst, df_cdt, df_gonogo = load_task_analyses()
    tmt_time, tmt_pct, cdt_long = prepare_data(df_tmt, df_cdt)

    # An aspect ratio near 2.04 matches the thesis figure proportions.
    fig = plt.figure(figsize=(12, 5.88))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    axes = [fig.add_subplot(gs[row, col]) for row in (0, 1) for col in (0, 1, 2)]
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes

    _violin_panel(ax_a, tmt_time, "time_sec", C_TMT, titles["A"], ylabels["A"],
                  tmt_time["subject_id"].nunique(), x="Part", xticklabels=parts)
    _violin_panel(ax_b, tmt_pct, "pct_valid", C_TMT, titles["B"], ylabels["B"],
                  tmt_pct["subject_id"].nunique(), x="Part", xticklabels=parts)
    ax_b.set_ylim(-5, 105)
    _violin_panel(ax_c, cdt_long, "K", C_CDT, titles["C"], ylabels["C"],
                  df_cdt["subject_id"].nunique(), x="Set Size",
                  xticklabels=list(SET_SIZE_MAP.values()))
    _violin_panel(ax_d, df_sst, "ssrt", C_SST, titles["D"], ylabels["D"], len(df_sst))
    _violin_panel(ax_e, df_gonogo, "c", C_GONOGO, titles["E"], ylabels["E"],
                  len(df_gonogo))
    ax_e.axhline(0, color=C_GRAY, ls="--", lw=0.8, zorder=0)
    _violin_panel(ax_f, df_gonogo, "accuracy", C_GONOGO, titles["F"], ylabels["F"],
                  len(df_gonogo))

    _apply_style(axes)
    fig.tight_layout()
    save_fig(fig, f"fig2_performance_distributions{lang_suffix(lang)}",
             formats=("png",), dpi=PRINT_DPI)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the performance-distributions violin figure")
    add_lang_argument(parser)
    args = parser.parse_args()
    main(args.lang)
