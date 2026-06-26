"""Plot a single cTMT Part B trial with the cursor trail colored by segmentation.

Each cursor point is colored by its movement state (hesitation / search /
travel), reproducing the Methods segmentation figure (thesis Figure 2A).
Standalone replica of paper_figures.ipynb cell 13, with color mapping
(hesitation=orange, search=green, travel=blue) and Spanish labels (no title).

Usage:
    python -m analysis.scripts.generate_segmentation_figure            # inglés
    python -m analysis.scripts.generate_segmentation_figure --lang es  # castellano
"""
import os
import glob

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 — registers styles
import pandas as pd

from src.config import BASE_DIR
from src import config
from src.mapper.datapruebas.datapruebas_mapper import DatapruebasTMTMapper
from neurotask.tmt.segmentation.segmentation import classify_cursor_positions_with_hesitation
from src.visualization.trial_plotting_helpers import draw_trial_targets, configure_trial_axes

FIGURES_DIR = os.path.join(BASE_DIR, "analysis", "figures")
HAND_ANALYSIS_DIR = os.path.join(BASE_DIR, "data", "hand_analysis")
RAW_EXPERIMENT_PATH = os.path.join(
    BASE_DIR, "data", "raw", "tmt", "datapruebas", "subjects", config.EXPERIMENT_FILE_NAME
)

plt.style.use(["science", "no-latex"])
LABEL_FS = 18
TICK_FS = 15
LEGEND_FS = 15
TARGET_FS = 15
DPI = 600

# Segmentation colors (Okabe-Ito)
SEG_COLORS = {
    "Hesitation": "#D55E00",  # naranja
    "Search":     "#2CA02C",  # verde (más vivo, separa mejor del azul)
    "Travel":     "#0072B2",  # azul
}
C_GRAY = "#888888"


def _latest_tmt_analysis_path():
    candidates = sorted(glob.glob(os.path.join(HAND_ANALYSIS_DIR, "*", "analysis.csv")))
    if not candidates:
        raise FileNotFoundError(f"No analysis.csv found under {HAND_ANALYSIS_DIR}")
    return candidates[-1]


def _select_trial(df_tmt, subject_id=None, trial_id=None):
    """Pick a representative valid PART_B trial (deterministic, mirrors cell 13)."""
    if subject_id is not None and trial_id is not None:
        row = df_tmt[(df_tmt["subject_id"] == subject_id) & (df_tmt["trial_id"] == trial_id)].iloc[0]
        return row
    df_cand = df_tmt[
        (df_tmt["is_valid"] == True) &
        (df_tmt["trial_type"].str.contains("PART_B")) &
        (df_tmt["trial_id"].str.startswith("DATAPRUEBAS")) &
        (df_tmt["non_cut_correct_targets_touches"] >= 20) &
        (df_tmt["total_hesitations"] >= 5)
    ].copy()
    return df_cand.sort_values("total_hesitations", ascending=False).iloc[5]


def main(lang="en", subject_id=None, trial_id=None):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df_tmt = pd.read_csv(_latest_tmt_analysis_path(), on_bad_lines="warn")
    row = _select_trial(df_tmt, subject_id, trial_id)
    subject_id = row["subject_id"]
    trial_id = row["trial_id"]
    speed_threshold = row["speed_threshold"]
    print(f"Selected: subject={subject_id}  trial={trial_id}  hesitations={row['total_hesitations']:.0f}")

    experiment = DatapruebasTMTMapper().map(RAW_EXPERIMENT_PATH)
    subject = experiment.subjects[subject_id]
    trial = next(t for t in subject.testing_trials if t.id == trial_id)

    segmentation = classify_cursor_positions_with_hesitation(
        trial, subject.target_radius, config.TARGET_RADIUS_MULTIPLIER, speed_threshold
    )
    labels = [label for (label, _) in segmentation]

    cursor_trail = trial.get_cursor_trail_from_start()
    cursor_x = [p.position.x for p in cursor_trail]
    cursor_y = [p.position.y for p in cursor_trail]
    point_colors = [SEG_COLORS.get(l, C_GRAY) for l in labels]

    es = lang == "es"
    xlabel = "Coordenada X (px)" if es else "X Screen Coordinate (px)"
    ylabel = "Coordenada Y (px)" if es else "Y Screen Coordinate (px)"
    legend_title = "Segmentación" if es else "Segmentation"

    fig, ax = plt.subplots(figsize=(7, 7))
    # faint trajectory line under the colored points (reading order)
    ax.plot(cursor_x, cursor_y, color="#cccccc", lw=1.0, alpha=0.6,
            zorder=2, solid_capstyle="round")
    ax.scatter(cursor_x, cursor_y, c=point_colors, s=20, alpha=0.9,
               linewidths=0, zorder=4)
    # neutral-gray targets so the blue stays exclusive to "hesitation"
    draw_trial_targets(ax, trial, subject.target_radius, circle_color="black",
                       circle_alpha=0.9, circle_linewidth=1.3,
                       text_fontsize=TARGET_FS, text_color="black")
    configure_trial_axes(ax, x=cursor_x, y=cursor_y, show_labels=True,
                         xlabel=xlabel, ylabel=ylabel)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=name,
                   markerfacecolor=SEG_COLORS[name], markersize=11)
        for name in ("Hesitation", "Search", "Travel")
    ]
    ax.legend(handles=handles, title=legend_title, frameon=True,
              fontsize=LEGEND_FS, title_fontsize=LEGEND_FS,
              framealpha=0.9, edgecolor="#cccccc")

    ax.xaxis.label.set_fontsize(LABEL_FS)
    ax.yaxis.label.set_fontsize(LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    # No title (per spec)

    suffix = "_es" if es else ""
    base = os.path.join(FIGURES_DIR, f"fig2a_tmt_segmentation{suffix}")
    fig.savefig(f"{base}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")  # vectorial
    print(f"Saved -> {base}.png  (+ .pdf)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot a cTMT Part B trial colored by segmentation")
    parser.add_argument("--lang", choices=["en", "es"], default="en",
                        help="Idioma de ejes/leyenda (default: en)")
    parser.add_argument("--subject", default=None, help="Override subject_id")
    parser.add_argument("--trial", default=None, help="Override trial_id")
    args = parser.parse_args()
    main(args.lang, args.subject, args.trial)
