"""Plot a single cTMT Part B trial with the cursor trail colored by elapsed time.

Temporal counterpart of `speed`: same trial and geometry, but each point is
colored by the time elapsed since the first click, making the reading order of
the trajectory explicit (early points dark, late points bright).

Usage:
    python -m analysis.scripts.figures.time              # castellano, s
    python -m analysis.scripts.figures.time --lang en    # inglés
    python -m analysis.scripts.figures.time --units ms   # ms
"""
import argparse

import matplotlib.pyplot as plt

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

# Cursor timestamps are in milliseconds; divide to reach the requested unit.
UNIT_FACTORS = {"ms": 1.0, "s": 1000.0}

CBAR_LABEL = {"es": "Tiempo transcurrido ({unit})", "en": "Elapsed time ({unit})"}


def point_times(cursor_trail, factor):
    """Elapsed time of each cursor point since the first click, in the given unit.

    Cursor timestamps are milliseconds and monotonically increasing, so every
    point has a defined value (the first one being 0).
    """
    t0 = cursor_trail[0].time
    x = [p.position.x for p in cursor_trail]
    y = [p.position.y for p in cursor_trail]
    elapsed = [(p.time - t0) / factor for p in cursor_trail]
    return x, y, elapsed


def main(lang="es", subject_id=None, trial_id=None, units="s", cmap="plasma",
         big_fonts=False):
    fonts = TRIAL_FONTS.scaled() if big_fonts else TRIAL_FONTS

    row, subject, trial = trial_data.load_trial(subject_id, trial_id)
    cursor_trail = trial.get_cursor_trail_from_start()
    cursor_x, cursor_y, elapsed = point_times(cursor_trail, UNIT_FACTORS[units])
    print(f"Elapsed ({units}): min={min(elapsed):.3f}  max={max(elapsed):.3f}  "
          f"n_points={len(cursor_trail)}")

    fig, ax = plt.subplots(figsize=(7, 7))
    trial_data.draw_faint_trail(ax, cursor_x, cursor_y)
    scatter = ax.scatter(cursor_x, cursor_y, c=elapsed, cmap=cmap,
                         vmin=0, vmax=max(elapsed),
                         s=20, alpha=0.9, linewidths=0, zorder=4)

    trial_data.draw_trial_background(ax, trial, subject, cursor_x, cursor_y, lang, fonts)
    trial_data.horizontal_colorbar_label(
        fig, scatter, ax, CBAR_LABEL[lang].format(unit=units), fonts)

    big = "_big" if big_fonts else ""
    save_fig(fig,
             f"fig_tmt_time_{units}_{trial_data.trial_slug(row)}{big}{lang_suffix(lang)}",
             dpi=TRIAL_DPI)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a cTMT Part B trial colored by elapsed time")
    add_lang_argument(parser)
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
