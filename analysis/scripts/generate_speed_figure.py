"""Plot a single cTMT Part B trial with the cursor trail colored by instantaneous speed.

Continuous counterpart of generate_segmentation_figure.py: same trial, same
geometry and styling, but each cursor point is colored by the speed of the
segment that reaches it (sequential colormap + colorbar) instead of by its
movement state. Plotting both figures side by side shows that hesitations
coincide with the low-speed regions of the trail.

Usage:
    python -m analysis.scripts.generate_speed_figure                     # inglés, px/ms
    python -m analysis.scripts.generate_speed_figure --lang es           # castellano
    python -m analysis.scripts.generate_speed_figure --units s           # px/s
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 — registers styles
import pandas as pd

from src import config
from src.mapper.datapruebas.datapruebas_mapper import DatapruebasTMTMapper
from neurotask.tmt.metrics.speed_metrics import calculate_speeds_with_validity
from src.visualization.trial_plotting_helpers import draw_trial_targets, configure_trial_axes
# Reuse the segmentation figure's trial selection so both figures always show
# the same trial, even if the selection criteria change.
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

C_GRAY = "#888888"
C_START = "#cccccc"

# Speed is computed in px/ms; multiply to reach the requested unit.
UNIT_FACTORS = {"ms": 1.0, "s": 1000.0}


def _point_speeds(cursor_trail, factor):
    """Speed of the segment reaching each point, split into valid and invalid.

    calculate_speeds_with_validity returns len(cursor_trail) - 1 results aligned
    with the segments, so the speed at index i belongs to cursor point i + 1.
    The first point (first click) has no defined speed and is excluded here.

    Returns:
        (valid_x, valid_y, valid_speeds, invalid_x, invalid_y)
    """
    speed_results = calculate_speeds_with_validity(cursor_trail)

    valid_x, valid_y, valid_speeds = [], [], []
    invalid_x, invalid_y = [], []
    for i, result in enumerate(speed_results):
        position = cursor_trail[i + 1].position
        if result.is_valid:
            valid_x.append(position.x)
            valid_y.append(position.y)
            valid_speeds.append(result.value * factor)
        else:
            invalid_x.append(position.x)
            invalid_y.append(position.y)

    return valid_x, valid_y, valid_speeds, invalid_x, invalid_y


def main(lang="en", subject_id=None, trial_id=None, units="s", cmap="viridis",
         vmax_percentile=99.0, big_fonts=False):
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
    cursor_x = [p.position.x for p in cursor_trail]
    cursor_y = [p.position.y for p in cursor_trail]

    factor = UNIT_FACTORS[units]
    valid_x, valid_y, speeds, invalid_x, invalid_y = _point_speeds(cursor_trail, factor)
    if not speeds:
        raise ValueError(f"No valid speeds for subject={subject_id} trial={trial_id}")

    # Saturate the color scale so a single fast jump does not flatten the range.
    vmax = float(np.percentile(speeds, vmax_percentile))
    print(f"Speed ({units}): min={min(speeds):.3f}  max={max(speeds):.3f}  "
          f"p{vmax_percentile:g}={vmax:.3f}  invalid={len(invalid_x)}/{len(cursor_trail) - 1}")

    es = lang == "es"
    xlabel = "Coordenada X (px)" if es else "X Screen Coordinate (px)"
    ylabel = "Coordenada Y (px)" if es else "Y Screen Coordinate (px)"
    cbar_label = f"Velocidad (px/{units})" if es else f"Speed (px/{units})"

    fig, ax = plt.subplots(figsize=(7, 7))
    # faint trajectory line under the colored points (reading order)
    ax.plot(cursor_x, cursor_y, color="#cccccc", lw=1.0, alpha=0.6,
            zorder=2, solid_capstyle="round")
    scatter = ax.scatter(valid_x, valid_y, c=speeds, cmap=cmap, vmin=0, vmax=vmax,
                         s=20, alpha=0.9, linewidths=0, zorder=4)
    # points whose speed could not be computed (non-monotonic time or above
    # INVALID_SPEED_THRESHOLD) stay gray instead of skewing the color scale
    if invalid_x:
        ax.scatter(invalid_x, invalid_y, c=C_GRAY, s=20, alpha=0.9,
                   linewidths=0, zorder=4)
    # first click has no incoming segment, so it has no speed
    ax.scatter([cursor_x[0]], [cursor_y[0]], c=C_START, s=20, alpha=0.9,
               linewidths=0, zorder=4)
    draw_trial_targets(ax, trial, subject.target_radius, circle_color="black",
                       circle_alpha=0.9, circle_linewidth=1.3,
                       text_fontsize=TARGET_FS, text_color="black")
    configure_trial_axes(ax, x=cursor_x, y=cursor_y, show_labels=True,
                         xlabel=xlabel, ylabel=ylabel)

    cbar = fig.colorbar(scatter, ax=ax, extend="max", fraction=0.046, pad=0.04)
    # horizontal label on top of the colorbar (matches the time figure) instead
    # of the default rotated label on the side
    cbar.ax.set_title(cbar_label, fontsize=CBAR_FS, pad=10)
    cbar.ax.tick_params(labelsize=TICK_FS)

    ax.xaxis.label.set_fontsize(LABEL_FS)
    ax.yaxis.label.set_fontsize(LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    # No title (per spec)

    # subject and trial go in the filename so the figure is traceable to its source;
    # the unit too, so px/ms and px/s runs don't overwrite each other
    lang_suffix = "_es" if es else ""
    big_suffix = "_big" if big_fonts else ""
    base = os.path.join(
        FIGURES_DIR,
        f"fig_tmt_speed_px{units}_{subject_id[:8]}_{trial_id}{big_suffix}{lang_suffix}"
    )
    fig.savefig(f"{base}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")  # vectorial
    print(f"Saved -> {base}.png  (+ .pdf)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot a cTMT Part B trial colored by instantaneous speed")
    parser.add_argument("--lang", choices=["en", "es"], default="en",
                        help="Idioma de ejes/colorbar (default: en)")
    parser.add_argument("--subject", default=None, help="Override subject_id")
    parser.add_argument("--trial", default=None, help="Override trial_id")
    parser.add_argument("--units", choices=["ms", "s"], default="s",
                        help="Unidades de velocidad: px/ms o px/s (default: s)")
    parser.add_argument("--cmap", default="viridis", help="Colormap secuencial (default: viridis)")
    parser.add_argument("--vmax-percentile", type=float, default=99.0,
                        help="Percentil de velocidad en que satura la escala (default: 99)")
    parser.add_argument("--big-fonts", action="store_true",
                        help="Agranda las fuentes para presentaciones")
    args = parser.parse_args()
    main(args.lang, args.subject, args.trial, args.units, args.cmap,
         args.vmax_percentile, args.big_fonts)
