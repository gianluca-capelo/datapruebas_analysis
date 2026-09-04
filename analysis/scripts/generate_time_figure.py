"""Plot a single cTMT Part B trial with the cursor trail colored by elapsed time.

Temporal counterpart of generate_speed_figure.py: same trial, same geometry and
styling, but each cursor point is colored by how much time has elapsed since the
first click (sequential colormap + colorbar) instead of by its instantaneous
speed. This makes the reading order of the trajectory explicit: early points are
dark, late points are bright, so the eye can follow how the trial unfolds in time.

Usage:
    python -m analysis.scripts.generate_time_figure                     # inglés, s
    python -m analysis.scripts.generate_time_figure --lang es           # castellano
    python -m analysis.scripts.generate_time_figure --units ms          # ms
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 — registers styles
import pandas as pd

from src import config
from src.mapper.datapruebas.datapruebas_mapper import DatapruebasTMTMapper
from src.visualization.trial_plotting_helpers import draw_trial_targets, configure_trial_axes
# Reuse the segmentation figure's trial selection so all three figures
# (segmentation / speed / time) always show the same trial.
from analysis.scripts.generate_segmentation_figure import (
    FIGURES_DIR,
    RAW_EXPERIMENT_PATH,
    _latest_tmt_analysis_path,
    _select_trial,
)

plt.style.use(["science", "no-latex"])
LABEL_FS = 18
TICK_FS = 15
CBAR_FS = 15
TARGET_FS = 15
DPI = 600

# --big-fonts multiplies every font size by this factor (for slides/presentations).
FONT_SCALE = 1.4

# Cursor timestamps are in milliseconds; divide to reach the requested unit.
UNIT_FACTORS = {"ms": 1.0, "s": 1000.0}


def _point_times(cursor_trail, factor):
    """Elapsed time of each cursor point since the first click, in the given unit.

    Cursor timestamps (``point.time``) are in milliseconds and monotonically
    increasing, so subtracting the first timestamp yields the elapsed time and
    every point has a defined value (including the first, which is 0).
    """
    t0 = cursor_trail[0].time
    x = [p.position.x for p in cursor_trail]
    y = [p.position.y for p in cursor_trail]
    elapsed = [(p.time - t0) / factor for p in cursor_trail]
    return x, y, elapsed


def main(lang="en", subject_id=None, trial_id=None, units="s", cmap="plasma",
         big_fonts=False):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if big_fonts:
        global LABEL_FS, TICK_FS, CBAR_FS, TARGET_FS
        LABEL_FS = round(LABEL_FS * FONT_SCALE)
        TICK_FS = round(TICK_FS * FONT_SCALE)
        CBAR_FS = round(CBAR_FS * FONT_SCALE)
        TARGET_FS = round(TARGET_FS * FONT_SCALE)

    df_tmt = pd.read_csv(_latest_tmt_analysis_path(), on_bad_lines="warn")
    row = _select_trial(df_tmt, subject_id, trial_id)
    subject_id = row["subject_id"]
    trial_id = row["trial_id"]
    print(f"Selected: subject={subject_id}  trial={trial_id}  hesitations={row['total_hesitations']:.0f}")

    experiment = DatapruebasTMTMapper().map(RAW_EXPERIMENT_PATH)
    subject = experiment.subjects[subject_id]
    trial = next(t for t in subject.testing_trials if t.id == trial_id)

    cursor_trail = trial.get_cursor_trail_from_start()

    factor = UNIT_FACTORS[units]
    cursor_x, cursor_y, elapsed = _point_times(cursor_trail, factor)
    print(f"Elapsed ({units}): min={min(elapsed):.3f}  max={max(elapsed):.3f}  "
          f"n_points={len(cursor_trail)}")

    es = lang == "es"
    xlabel = "Coordenada X (px)" if es else "X Screen Coordinate (px)"
    ylabel = "Coordenada Y (px)" if es else "Y Screen Coordinate (px)"
    cbar_label = f"Tiempo transcurrido ({units})" if es else f"Elapsed time ({units})"

    fig, ax = plt.subplots(figsize=(7, 7))
    # faint trajectory line under the colored points (reading order)
    ax.plot(cursor_x, cursor_y, color="#cccccc", lw=1.0, alpha=0.6,
            zorder=2, solid_capstyle="round")
    scatter = ax.scatter(cursor_x, cursor_y, c=elapsed, cmap=cmap,
                         vmin=0, vmax=max(elapsed),
                         s=20, alpha=0.9, linewidths=0, zorder=4)
    draw_trial_targets(ax, trial, subject.target_radius, circle_color="black",
                       circle_alpha=0.9, circle_linewidth=1.3,
                       text_fontsize=TARGET_FS, text_color="black")
    configure_trial_axes(ax, x=cursor_x, y=cursor_y, show_labels=True,
                         xlabel=xlabel, ylabel=ylabel)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    # horizontal label on top of the colorbar (per spec) instead of the
    # default rotated label on the side
    cbar.ax.set_title(cbar_label, fontsize=CBAR_FS, pad=10)
    cbar.ax.tick_params(labelsize=TICK_FS)

    ax.xaxis.label.set_fontsize(LABEL_FS)
    ax.yaxis.label.set_fontsize(LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    # No title (per spec)

    # subject and trial go in the filename so the figure is traceable to its source;
    # the unit too, so ms and s runs don't overwrite each other
    lang_suffix = "_es" if es else ""
    big_suffix = "_big" if big_fonts else ""
    base = os.path.join(
        FIGURES_DIR,
        f"fig_tmt_time_{units}_{subject_id[:8]}_{trial_id}{big_suffix}{lang_suffix}"
    )
    fig.savefig(f"{base}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")  # vectorial
    print(f"Saved -> {base}.png  (+ .pdf)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot a cTMT Part B trial colored by elapsed time")
    parser.add_argument("--lang", choices=["en", "es"], default="en",
                        help="Idioma de ejes/colorbar (default: en)")
    parser.add_argument("--subject", default=None, help="Override subject_id")
    parser.add_argument("--trial", default=None, help="Override trial_id")
    parser.add_argument("--units", choices=["ms", "s"], default="s",
                        help="Unidades de tiempo: ms o s (default: s)")
    parser.add_argument("--cmap", default="plasma",
                        help="Colormap secuencial, distinto al de speed (default: plasma)")
    parser.add_argument("--big-fonts", action="store_true",
                        help="Agranda las fuentes para presentaciones")
    args = parser.parse_args()
    main(args.lang, args.subject, args.trial, args.units, args.cmap, args.big_fonts)
