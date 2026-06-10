"""Generate the performance-distributions violin figure (six panels).

Standalone replica of paper_figures.ipynb "Figure 2 — violin plot", extended
with the missing Go/No-Go Accuracy panel (F) and the relabeled c-coefficient
panel (E), so the six-panel thesis figure is reproducible from the repo.

Panels (2x3 grid):
    A. TMT Completion Time   B. TMT Completion Rate   C. Change Detection Task
    D. Stop Signal Task      E. Go/No-Go — c coeff    F. Go/No-Go — Accuracy

Usage:
    python -m analysis.scripts.generate_violin_figures
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scienceplots  # noqa: F401 — registers styles

from src.config import BASE_DIR
from src.loader import (
    get_latest_sst_analysis,
    get_latest_cdt_analysis,
    get_latest_gonogo_analysis,
)

HAND_ANALYSIS_DIR = os.path.join(BASE_DIR, "data", "hand_analysis")

FIGURES_DIR = os.path.join(BASE_DIR, "analysis", "figures")


def _latest_tmt_analysis_path():
    """Return the analysis.csv of the most recent hand_analysis timestamp.

    The notebook hardcoded a specific timestamp; here we resolve the latest
    available one (timestamps sort lexically) so the script stays reproducible
    as new analyses are generated.
    """
    candidates = sorted(glob.glob(os.path.join(HAND_ANALYSIS_DIR, "*", "analysis.csv")))
    if not candidates:
        raise FileNotFoundError(f"No analysis.csv found under {HAND_ANALYSIS_DIR}")
    return candidates[-1]

# ---------------------------------------------------------------------------
# Style — mirrors paper_figures.ipynb Setup cell
# ---------------------------------------------------------------------------
plt.style.use(["science", "no-latex"])

TITLE_FS  = 13.5
LABEL_FS  = 16
TICK_FS   = 14
ANNOT_FS  = 10
LEGEND_FS = 12

plt.rcParams.update({
    "axes.titlesize":  TITLE_FS,
    "axes.labelsize":  LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
    "xtick.top":           False,
    "ytick.right":         False,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
})

# Okabe-Ito task-specific colors (same as notebook)
C_BLUE    = "#0072B2"  # TMT
C_AMBER   = "#E69F00"  # CDT
C_TEAL    = "#009E73"  # SST
C_PURPLE  = "#9B59B6"  # Go/No-Go
C_GRAY    = "#888888"  # neutral reference lines

# Shared violin kwargs.
# NOTE: the notebook used cut=0 (KDE truncated flat at the data range); the
# thesis figure used seaborn's default cut=2, which lets the violins close with
# tapered tails. We keep the default cut to reproduce the thesis figure.
VIOLIN_KW = dict(inner="box", linewidth=0.8, saturation=0.85)

DPI = 300

# One color per task
C_TMT    = C_BLUE    # panels A & B
C_CDT    = C_AMBER   # panel C
C_SST    = C_TEAL    # panel D
C_GONOGO = C_PURPLE  # panels E & F


def save_fig(fig, filename):
    """Save figure to FIGURES_DIR with consistent settings."""
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"Saved -> {path}")


def apply_style(axes):
    """Re-enforce title and axis label font sizes after seaborn calls."""
    for ax in np.atleast_1d(axes).flat:
        ax.title.set_fontsize(TITLE_FS)
        ax.xaxis.label.set_fontsize(LABEL_FS)
        ax.yaxis.label.set_fontsize(LABEL_FS)


def _load_data():
    """Load the task DataFrames used by the figure (subset of notebook cell 5)."""
    tmt_path = _latest_tmt_analysis_path()
    print(f"TMT analysis: {tmt_path}")
    df_tmt = pd.read_csv(tmt_path, on_bad_lines="warn")
    df_sst, _ = get_latest_sst_analysis()
    df_cdt, _ = get_latest_cdt_analysis()
    df_gonogo, _ = get_latest_gonogo_analysis()
    return df_tmt, df_sst, df_cdt, df_gonogo


def _prepare_data(df_tmt, df_cdt):
    """Build derived frames tmt_time, tmt_pct, cdt_long (notebook cell 8)."""
    df_tmt_part = df_tmt.copy()
    df_tmt_part["part"] = df_tmt_part["trial_type"].str.extract(r"(PART_[AB])")[0]
    df_tmt_part = df_tmt_part.dropna(subset=["part"])

    df_tmt_valid = df_tmt_part[df_tmt_part["is_valid"] == True].copy()
    df_tmt_valid["time_sec"] = df_tmt_valid["non_cut_rt"] / 1000

    # N=368: subjects with at least one valid trial in BOTH Part A AND Part B
    sids_a    = set(df_tmt_valid[df_tmt_valid["part"] == "PART_A"]["subject_id"])
    sids_b    = set(df_tmt_valid[df_tmt_valid["part"] == "PART_B"]["subject_id"])
    sids_both = sids_a & sids_b
    print(f"TMT N (valid in both parts): {len(sids_both)}")

    # A. Mean completion time — restricted to sids_both
    tmt_time = (
        df_tmt_valid[df_tmt_valid["subject_id"].isin(sids_both)]
        .groupby(["subject_id", "part"])["time_sec"]
        .mean()
        .reset_index()
    )
    tmt_time["Part"] = tmt_time["part"].map({"PART_A": "Part A", "PART_B": "Part B"})

    # B. Percentage of valid trials — restricted to sids_both
    tmt_pct = (
        df_tmt_part[df_tmt_part["subject_id"].isin(sids_both)]
        .groupby(["subject_id", "part"])
        .agg(total=("is_valid", "count"), valid=("is_valid", "sum"))
        .reset_index()
    )
    tmt_pct["pct_valid"] = 100 * tmt_pct["valid"] / tmt_pct["total"]
    tmt_pct["Part"] = tmt_pct["part"].map({"PART_A": "Part A", "PART_B": "Part B"})

    # C. CDT — long format K_4 / K_6
    cdt_long = pd.melt(
        df_cdt[["subject_id", "K_4", "K_6"]],
        id_vars="subject_id",
        value_vars=["K_4", "K_6"],
        var_name="Set Size",
        value_name="K",
    )
    cdt_long["Set Size"] = cdt_long["Set Size"].map({"K_4": "$K_4$", "K_6": "$K_6$"})

    return tmt_time, tmt_pct, cdt_long


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading data...")
    df_tmt, df_sst, df_cdt, df_gonogo = _load_data()
    tmt_time, tmt_pct, cdt_long = _prepare_data(df_tmt, df_cdt)

    # -----------------------------------------------------------------------
    # Six-panel figure — uniform 2x3 grid
    # -----------------------------------------------------------------------
    # figsize aspect ~2.04 matches the thesis figure proportions
    fig = plt.figure(figsize=(12, 5.88))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    # --- A. TMT Completion Time ---
    sns.violinplot(data=tmt_time, x="Part", y="time_sec", ax=ax_a, **VIOLIN_KW, color=C_TMT)
    ax_a.set_title(f"A. TMT Completion Time\n($N$={tmt_time['subject_id'].nunique()})")
    ax_a.set_ylabel("Mean time (s)")
    ax_a.set_xlabel("")
    ax_a.set_xticklabels(["Part A", "Part B"], fontsize=TICK_FS)

    # --- B. TMT Completion Rate ---
    sns.violinplot(data=tmt_pct, x="Part", y="pct_valid", ax=ax_b, **VIOLIN_KW, color=C_TMT)
    ax_b.set_title(f"B. TMT Completion Rate\n($N$={tmt_pct['subject_id'].nunique()})")
    ax_b.set_ylabel("Valid trials (%)")
    ax_b.set_xlabel("")
    ax_b.set_ylim(-5, 105)
    ax_b.set_xticklabels(["Part A", "Part B"], fontsize=TICK_FS)

    # --- C. Change Detection Task (CDT) ---
    sns.violinplot(data=cdt_long, x="Set Size", y="K", ax=ax_c, **VIOLIN_KW, color=C_CDT)
    ax_c.set_title(f"C. Change Detection Task\n($N$={df_cdt['subject_id'].nunique()})")
    ax_c.set_ylabel("Cowan's $K$")
    ax_c.set_xlabel("")
    ax_c.set_xticklabels(["$K_4$", "$K_6$"], fontsize=TICK_FS)

    # --- D. Stop Signal Task (SST) ---
    sns.violinplot(data=df_sst, y="ssrt", ax=ax_d, **VIOLIN_KW, color=C_SST)
    ax_d.set_title(f"D. Stop Signal Task\n($N$={len(df_sst)})")
    ax_d.set_ylabel("SSRT (ms)")
    ax_d.set_xlabel("")
    ax_d.set_xticks([])

    # --- E. Go/No-Go — c coefficient ---
    sns.violinplot(data=df_gonogo, y="c", ax=ax_e, **VIOLIN_KW, color=C_GONOGO)
    ax_e.axhline(0, color=C_GRAY, ls="--", lw=0.8, zorder=0)
    ax_e.set_title(f"E. Go/No-Go — $c$ coefficient\n($N$={len(df_gonogo)})")
    ax_e.set_ylabel("$c$ coefficient")
    ax_e.set_xlabel("")
    ax_e.set_xticks([])

    # --- F. Go/No-Go — Accuracy ---
    sns.violinplot(data=df_gonogo, y="accuracy", ax=ax_f, **VIOLIN_KW, color=C_GONOGO)
    ax_f.set_title(f"F. Go/No-Go — Accuracy\n($N$={len(df_gonogo)})")
    ax_f.set_ylabel("Accuracy")
    ax_f.set_xlabel("")
    ax_f.set_xticks([])

    apply_style([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f])
    fig.tight_layout()
    save_fig(fig, "fig2_performance_distributions.png")


if __name__ == "__main__":
    main()
