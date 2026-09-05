"""Plot a single cTMT Part B trial with the cursor trail colored by instantaneous speed.

Continuous counterpart of `segmentation`: same trial, same geometry, but each
point is colored by the speed of the segment reaching it. Side by side the two
figures show that hesitations coincide with the low-speed stretches.

Usage:
    python -m analysis.scripts.figures.speed              # castellano, px/s
    python -m analysis.scripts.figures.speed --lang en    # inglés
    python -m analysis.scripts.figures.speed --units ms   # px/ms
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from neurotask.tmt.metrics.speed_metrics import calculate_speeds_with_validity

from analysis.scripts.figures import trial_data
from analysis.scripts.figures._style import (
    TRIAL_DPI,
    TRIAL_FONTS,
    add_lang_argument,
    lang_suffix,
    save_fig,
    use_science_style,
)

use_science_style()

C_INVALID = "#888888"
C_START = "#cccccc"

# Speed is computed in px/ms; multiply to reach the requested unit.
UNIT_FACTORS = {"ms": 1.0, "s": 1000.0}

CBAR_LABEL = {"es": "Velocidad (px/{unit})", "en": "Speed (px/{unit})"}


def point_speeds(cursor_trail, factor):
    """Speed of the segment reaching each point, split into valid and invalid.

    calculate_speeds_with_validity returns len(cursor_trail) - 1 results aligned
    with the segments, so the speed at index i belongs to cursor point i + 1.
    The first point (first click) has no incoming segment and is excluded.

    Returns:
        (valid_x, valid_y, valid_speeds, invalid_x, invalid_y)
    """
    valid_x, valid_y, valid_speeds = [], [], []
    invalid_x, invalid_y = [], []
    for i, result in enumerate(calculate_speeds_with_validity(cursor_trail)):
        position = cursor_trail[i + 1].position
        if result.is_valid:
            valid_x.append(position.x)
            valid_y.append(position.y)
            valid_speeds.append(result.value * factor)
        else:
            invalid_x.append(position.x)
            invalid_y.append(position.y)

    return valid_x, valid_y, valid_speeds, invalid_x, invalid_y


def main(lang="es", subject_id=None, trial_id=None, units="s", cmap="viridis",
         vmax_percentile=99.0, big_fonts=False):
    fonts = TRIAL_FONTS.scaled() if big_fonts else TRIAL_FONTS

    row, subject, trial = trial_data.load_trial(subject_id, trial_id)
    cursor_trail, cursor_x, cursor_y = trial_data.cursor_coordinates(trial)

    valid_x, valid_y, speeds, invalid_x, invalid_y = point_speeds(
        cursor_trail, UNIT_FACTORS[units])
    if not speeds:
        raise ValueError(f"No valid speeds for trial {row['trial_id']}")

    # Saturate the color scale so a single fast jump does not flatten the range.
    vmax = float(np.percentile(speeds, vmax_percentile))
    print(f"Speed ({units}): min={min(speeds):.3f}  max={max(speeds):.3f}  "
          f"p{vmax_percentile:g}={vmax:.3f}  invalid={len(invalid_x)}/{len(cursor_trail) - 1}")

    fig, ax = plt.subplots(figsize=(7, 7))
    trial_data.draw_faint_trail(ax, cursor_x, cursor_y)
    scatter = ax.scatter(valid_x, valid_y, c=speeds, cmap=cmap, vmin=0, vmax=vmax,
                         s=20, alpha=0.9, linewidths=0, zorder=4)
    # Points whose speed could not be computed (non-monotonic time or above
    # INVALID_SPEED_THRESHOLD) stay gray instead of skewing the color scale.
    if invalid_x:
        ax.scatter(invalid_x, invalid_y, c=C_INVALID, s=20, alpha=0.9,
                   linewidths=0, zorder=4)
    ax.scatter([cursor_x[0]], [cursor_y[0]], c=C_START, s=20, alpha=0.9,
               linewidths=0, zorder=4)

    trial_data.draw_trial_background(ax, trial, subject, cursor_x, cursor_y, lang, fonts)
    trial_data.horizontal_colorbar_label(
        fig, scatter, ax, CBAR_LABEL[lang].format(unit=units), fonts, extend="max")

    big = "_big" if big_fonts else ""
    save_fig(fig,
             f"fig_tmt_speed_px{units}_{trial_data.trial_slug(row)}{big}{lang_suffix(lang)}",
             dpi=TRIAL_DPI)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a cTMT Part B trial colored by instantaneous speed")
    add_lang_argument(parser)
    parser.add_argument("--subject", default=None, help="Override subject_id")
    parser.add_argument("--trial", default=None, help="Override trial_id")
    parser.add_argument("--units", choices=["ms", "s"], default="s",
                        help="Unidades de velocidad: px/ms o px/s (default: s)")
    parser.add_argument("--cmap", default="viridis",
                        help="Colormap secuencial (default: viridis)")
    parser.add_argument("--vmax-percentile", type=float, default=99.0,
                        help="Percentil de velocidad en que satura la escala (default: 99)")
    parser.add_argument("--big-fonts", action="store_true",
                        help="Agranda las fuentes para presentaciones")
    args = parser.parse_args()
    main(args.lang, args.subject, args.trial, args.units, args.cmap,
         args.vmax_percentile, args.big_fonts)
