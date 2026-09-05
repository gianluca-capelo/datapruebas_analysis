"""Animate a single cTMT Part B trial being completed, cursor colored by time.

Animated counterpart of `time`: same trial and styling, but the cursor points
are revealed one after another at a constant frame rate, with the current
position highlighted. Targets stay fixed in the background.

Outputs an MP4 (ffmpeg comes bundled via imageio-ffmpeg), falling back to a GIF.

Usage:
    python -m analysis.scripts.figures.time_animation                # castellano, s
    python -m analysis.scripts.figures.time_animation --lang en      # inglés
    python -m analysis.scripts.figures.time_animation --seconds 15   # más lenta
"""
import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from analysis.scripts.figures import trial_data
from analysis.scripts.figures._style import (
    ANIMATION_DPI,
    FIGURES_DIR,
    SLIDE_FONT_SCALE,
    TRIAL_FONTS,
    add_lang_argument,
    lang_suffix,
    use_science_style,
)
from analysis.scripts.figures.animation_common import (
    cursor_style,
    frame_point_counts,
    resolve_writer,
)
from analysis.scripts.figures.time import CBAR_LABEL, UNIT_FACTORS, point_times

use_science_style()

# Margins reserved on the fixed canvas so the axis labels and the colorbar's
# horizontal top label are not clipped — the tight-bbox equivalent that the
# static figures get for free at save time.
CANVAS_MARGINS = dict(left=0.11, right=0.82, top=0.9, bottom=0.1)


def main(lang="es", subject_id=None, trial_id=None, units="s", cmap="plasma",
         fps=30, seconds=12.0, hold=1.5, fmt="mp4", cursor="arrow", big_fonts=False):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fonts = TRIAL_FONTS.scaled() if big_fonts else TRIAL_FONTS

    row, subject, trial = trial_data.load_trial(subject_id, trial_id)
    cursor_trail = trial.get_cursor_trail_from_start()
    cursor_x, cursor_y, elapsed = point_times(cursor_trail, UNIT_FACTORS[units])
    cursor_x = np.asarray(cursor_x)
    cursor_y = np.asarray(cursor_y)
    elapsed = np.asarray(elapsed)
    n_points = len(cursor_trail)
    print(f"Elapsed ({units}): min={elapsed.min():.3f}  max={elapsed.max():.3f}  "
          f"n_points={n_points}")

    # Wider than tall: the plot stays square (aspect="equal") and the extra width
    # leaves room for the colorbar plus its horizontal top label, which on a
    # fixed-size animation canvas would otherwise clip.
    fig, ax = plt.subplots(figsize=(8.5, 7))
    trial_data.draw_trial_background(ax, trial, subject, cursor_x, cursor_y, lang, fonts)

    line, = ax.plot([], [], color=trial_data.C_FAINT_TRAIL, lw=1.0, alpha=0.6,
                    zorder=2, solid_capstyle="round")
    scatter = ax.scatter([], [], c=[], cmap=cmap, vmin=0, vmax=float(elapsed.max()),
                         s=20, alpha=0.9, linewidths=0, zorder=4)
    marker, marker_size = cursor_style(cursor, SLIDE_FONT_SCALE if big_fonts else 1.0)
    current, = ax.plot([], [], marker=marker, markersize=marker_size,
                       markerfacecolor="white", markeredgecolor="black",
                       markeredgewidth=1.5, linestyle="None", zorder=6)

    # The colorbar spans the full time range from the first frame, so the scale
    # stays stable while the animation plays.
    trial_data.horizontal_colorbar_label(
        fig, scatter, ax, CBAR_LABEL[lang].format(unit=units), fonts)
    fig.subplots_adjust(**CANVAS_MARGINS)

    counts = frame_point_counts(n_points, max(1, round(fps * seconds)),
                                max(0, round(fps * hold)))

    def update(frame_idx):
        k = counts[frame_idx]
        scatter.set_offsets(np.column_stack([cursor_x[:k], cursor_y[:k]]))
        scatter.set_array(elapsed[:k])
        line.set_data(cursor_x[:k], cursor_y[:k])
        current.set_data([cursor_x[k - 1]], [cursor_y[k - 1]])
        return scatter, line, current

    anim = animation.FuncAnimation(
        fig, update, frames=len(counts), interval=1000.0 / fps, blit=False
    )

    big = "_big" if big_fonts else ""
    # The sweep duration goes in the filename so runs with different --seconds
    # do not overwrite each other (":g" keeps 13.0 -> "13" and 21.5 -> "21.5").
    base = os.path.join(
        FIGURES_DIR,
        f"fig_tmt_time_animation_{units}_{trial_data.trial_slug(row)}"
        f"_{seconds:g}s{big}{lang_suffix(lang)}"
    )
    out_path, writer = resolve_writer(base, fmt, fps)
    anim.save(out_path, writer=writer, dpi=ANIMATION_DPI)
    print(f"Saved -> {out_path}  ({len(counts)} frames @ {fps} fps)")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animate a cTMT Part B trial being completed, colored by elapsed time")
    add_lang_argument(parser)
    parser.add_argument("--subject", default=None, help="Override subject_id")
    parser.add_argument("--trial", default=None, help="Override trial_id")
    parser.add_argument("--units", choices=["ms", "s"], default="s",
                        help="Unidades de tiempo: ms o s (default: s)")
    parser.add_argument("--cmap", default="plasma",
                        help="Colormap secuencial (default: plasma, igual que la figura estática)")
    parser.add_argument("--fps", type=int, default=30, help="Cuadros por segundo (default: 30)")
    parser.add_argument("--seconds", type=float, default=12.0,
                        help="Duración del barrido del trail en segundos (default: 12)")
    parser.add_argument("--hold", type=float, default=1.5,
                        help="Segundos que se mantiene el trail completo al final (default: 1.5)")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4", dest="fmt",
                        help="Formato de salida (default: mp4; cae a gif si no hay ffmpeg)")
    parser.add_argument("--cursor", choices=["arrow", "dot"], default="arrow",
                        help="Marcador de la posición actual (default: arrow)")
    parser.add_argument("--big-fonts", action="store_true",
                        help="Agranda las fuentes para presentaciones")
    args = parser.parse_args()
    main(args.lang, args.subject, args.trial, args.units, args.cmap, args.fps,
         args.seconds, args.hold, args.fmt, args.cursor, args.big_fonts)
