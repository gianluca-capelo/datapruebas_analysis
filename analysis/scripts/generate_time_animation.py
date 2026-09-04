"""Animate a single cTMT Part B trial being completed, cursor colored by time.

Animated counterpart of generate_time_figure.py: same trial, same subject, same
geometry and styling, but instead of drawing the whole cursor trail at once it
reveals the cursor points one after another at a constant frame rate. The growing
trail is colored by elapsed time (plasma colormap, as in the static time figure)
and the current cursor position is highlighted. Targets stay fixed in the
background the whole time (they are not highlighted as they are reached).

The output is an MP4 (needs ffmpeg on the system). If ffmpeg is unavailable it
falls back to a GIF via pillow.

Usage:
    python -m analysis.scripts.generate_time_animation                   # inglés, s
    python -m analysis.scripts.generate_time_animation --lang es         # castellano
    python -m analysis.scripts.generate_time_animation --seconds 15      # más lenta
"""
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
import numpy as np
import scienceplots  # noqa: F401 — registers styles
import pandas as pd

from src import config
from src.mapper.datapruebas.datapruebas_mapper import DatapruebasTMTMapper
from src.visualization.trial_plotting_helpers import draw_trial_targets, configure_trial_axes
# Reuse the segmentation figure's trial selection and the time figure's per-point
# elapsed-time helper so this animation always shows the same trial as the three
# static figures (segmentation / speed / time) and shares their color mapping.
from analysis.scripts.generate_segmentation_figure import (
    FIGURES_DIR,
    RAW_EXPERIMENT_PATH,
    _latest_tmt_analysis_path,
    _select_trial,
)
from analysis.scripts.generate_time_figure import _point_times, UNIT_FACTORS

plt.style.use(["science", "no-latex"])
LABEL_FS = 18
TICK_FS = 15
CBAR_FS = 15
TARGET_FS = 15
DPI = 150  # animations render many frames; 600 DPI would be needlessly heavy

# --big-fonts multiplies every font size by this factor (for slides/presentations).
FONT_SCALE = 1.4

# Classic mouse-pointer outline as a marker path. The hotspot (tip) sits at the
# origin so the marker points exactly at the cursor position; the body extends
# down-right, giving the familiar up-left arrow. Coordinates are normalized
# (~unit height); markersize scales it. Markers are drawn in display space, so
# the arrow keeps its upright orientation regardless of the inverted y axis.
_MOUSE_CURSOR_MARKER = MarkerStyle(Path([
    (0.00, 0.00),    # tip (hotspot)
    (0.00, -1.00),
    (0.25, -0.75),
    (0.44, -1.13),
    (0.56, -1.06),
    (0.38, -0.69),
    (0.69, -0.69),
    (0.00, 0.00),    # back to tip
], closed=True))


def _frame_point_counts(n_points, reveal_frames, hold_frames):
    """Number of cursor points visible at each frame.

    The first ``reveal_frames`` frames grow the trail linearly from 1 point to
    all ``n_points`` points; the trailing ``hold_frames`` keep the full trail on
    screen so the finished trajectory lingers before the loop restarts.
    """
    counts = []
    for frame in range(reveal_frames):
        progress = (frame + 1) / reveal_frames
        counts.append(max(1, round(n_points * progress)))
    counts.extend([n_points] * hold_frames)
    return counts


def main(lang="en", subject_id=None, trial_id=None, units="s", cmap="plasma",
         fps=30, seconds=12.0, hold=1.5, fmt="mp4", cursor="arrow", big_fonts=False):
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
    cursor_x = np.asarray(cursor_x)
    cursor_y = np.asarray(cursor_y)
    elapsed = np.asarray(elapsed)
    n_points = len(cursor_trail)
    print(f"Elapsed ({units}): min={elapsed.min():.3f}  max={elapsed.max():.3f}  "
          f"n_points={n_points}")

    es = lang == "es"
    xlabel = "Coordenada X (px)" if es else "X Screen Coordinate (px)"
    ylabel = "Coordenada Y (px)" if es else "Y Screen Coordinate (px)"
    cbar_label = f"Tiempo transcurrido ({units})" if es else f"Elapsed time ({units})"

    # Slightly wider than tall: the plot stays square (aspect="equal") and the
    # extra width leaves room to the right for the colorbar plus its horizontal
    # top label, which on a fixed-size animation canvas would otherwise clip.
    fig, ax = plt.subplots(figsize=(8.5, 7))

    # Static background: targets and axes are drawn once and never change. The
    # axis limits come from the full trajectory so the view does not rescale as
    # the trail grows.
    draw_trial_targets(ax, trial, subject.target_radius, circle_color="black",
                       circle_alpha=0.9, circle_linewidth=1.3,
                       text_fontsize=TARGET_FS, text_color="black")
    configure_trial_axes(ax, x=cursor_x, y=cursor_y, show_labels=True,
                         xlabel=xlabel, ylabel=ylabel)

    # Animated artists, all starting empty:
    #  - line: faint gray connector under the points (reading order)
    #  - scatter: the revealed cursor points, colored by elapsed time
    #  - current: highlighted marker at the latest revealed cursor position
    line, = ax.plot([], [], color="#cccccc", lw=1.0, alpha=0.6,
                    zorder=2, solid_capstyle="round")
    scatter = ax.scatter([], [], c=[], cmap=cmap, vmin=0, vmax=float(elapsed.max()),
                         s=20, alpha=0.9, linewidths=0, zorder=4)
    # The current cursor position is marked either by a mouse pointer (default,
    # tip on the point) or a plain dot. The arrow needs a bigger markersize
    # because its path spans ~2x the unit radius of the "o" marker.
    if cursor == "arrow":
        cursor_marker, cursor_size = _MOUSE_CURSOR_MARKER, 22
    else:
        cursor_marker, cursor_size = "o", 11
    if big_fonts:
        cursor_size = round(cursor_size * FONT_SCALE)
    current, = ax.plot([], [], marker=cursor_marker, markersize=cursor_size,
                       markerfacecolor="white", markeredgecolor="black",
                       markeredgewidth=1.5, linestyle="None", zorder=6)

    # Colorbar spans the full time range from the first frame so the scale is
    # stable while the animation plays. The label sits horizontally on top of
    # the colorbar (matching the static time figure); the wide figure plus the
    # explicit right margin below keep it inside the fixed animation canvas.
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_title(cbar_label, fontsize=CBAR_FS, pad=10)
    cbar.ax.tick_params(labelsize=TICK_FS)

    ax.xaxis.label.set_fontsize(LABEL_FS)
    ax.yaxis.label.set_fontsize(LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    # Reserve margins on the fixed canvas so axis labels and the colorbar's
    # horizontal top label are not clipped (the tight-bbox equivalent that the
    # static figures get for free at save time). The generous right margin
    # leaves room for the half of the centered top label that extends past the
    # colorbar toward the figure edge.
    fig.subplots_adjust(left=0.11, right=0.82, top=0.9, bottom=0.1)

    reveal_frames = max(1, round(fps * seconds))
    hold_frames = max(0, round(fps * hold))
    counts = _frame_point_counts(n_points, reveal_frames, hold_frames)

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

    lang_suffix = "_es" if es else ""
    big_suffix = "_big" if big_fonts else ""
    # the sweep duration goes in the filename so runs with different --seconds
    # don't overwrite each other (":g" keeps 13.0 -> "13" and 21.5 -> "21.5")
    dur_suffix = f"_{seconds:g}s"
    base = os.path.join(
        FIGURES_DIR,
        f"fig_tmt_time_animation_{units}_{subject_id[:8]}_{trial_id}{dur_suffix}{big_suffix}{lang_suffix}"
    )
    out_path, writer = _resolve_writer(base, fmt, fps)
    anim.save(out_path, writer=writer, dpi=DPI)
    print(f"Saved -> {out_path}  ({len(counts)} frames @ {fps} fps)")
    plt.close(fig)


def _register_bundled_ffmpeg():
    """Point matplotlib at the ffmpeg binary bundled with imageio-ffmpeg.

    Avoids requiring a system-wide ffmpeg install: the binary ships inside the
    venv via the imageio-ffmpeg package. No-op if the package is absent, in
    which case _resolve_writer falls back to a GIF.
    """
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass


def _resolve_writer(base, fmt, fps):
    """Pick an output path + matplotlib writer, falling back mp4 -> gif.

    MP4 needs ffmpeg (provided here by imageio-ffmpeg inside the venv); if it is
    unavailable we transparently fall back to a pillow GIF so the script still
    produces something usable.
    """
    _register_bundled_ffmpeg()
    if fmt == "mp4" and animation.FFMpegWriter.isAvailable():
        return f"{base}.mp4", animation.FFMpegWriter(fps=fps, bitrate=2400)
    if fmt == "mp4":
        print("ffmpeg not available; falling back to GIF (pillow).")
    return f"{base}.gif", animation.PillowWriter(fps=fps)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Animate a cTMT Part B trial being completed, colored by elapsed time")
    parser.add_argument("--lang", choices=["en", "es"], default="en",
                        help="Idioma de ejes/colorbar (default: en)")
    parser.add_argument("--subject", default=None, help="Override subject_id")
    parser.add_argument("--trial", default=None, help="Override trial_id")
    parser.add_argument("--units", choices=["ms", "s"], default="s",
                        help="Unidades de tiempo: ms o s (default: s)")
    parser.add_argument("--cmap", default="plasma",
                        help="Colormap secuencial (default: plasma, igual que la figura de tiempo)")
    parser.add_argument("--fps", type=int, default=30, help="Cuadros por segundo (default: 30)")
    parser.add_argument("--seconds", type=float, default=12.0,
                        help="Duración del barrido del trail en segundos (default: 12)")
    parser.add_argument("--hold", type=float, default=1.5,
                        help="Segundos que se mantiene el trail completo al final (default: 1.5)")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4", dest="fmt",
                        help="Formato de salida (default: mp4; cae a gif si no hay ffmpeg)")
    parser.add_argument("--cursor", choices=["arrow", "dot"], default="arrow",
                        help="Marcador de la posición actual: flecha de mouse o punto (default: arrow)")
    parser.add_argument("--big-fonts", action="store_true",
                        help="Agranda las fuentes para presentaciones")
    args = parser.parse_args()
    main(args.lang, args.subject, args.trial, args.units, args.cmap, args.fps,
         args.seconds, args.hold, args.fmt, args.cursor, args.big_fonts)
