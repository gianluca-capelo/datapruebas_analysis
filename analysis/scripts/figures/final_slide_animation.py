"""Animate a synthetic cTMT-style trail over big labelled targets (closing slide).

Presentation counterpart of `time_animation`: same look and feel (mouse
cursor sweeping a trail colored by elapsed time, mp4 with gif fallback) but with
NO real data involved — neither the targets nor the trajectory come from an
experiment, they are generated here.

The targets carry the storyline of the talk (cTMT, Ingeniería de atributos,
Aprendizaje Automático, ... , Fin), are laid out on a 16:9 canvas and connected by a
synthetic, human-looking trajectory (minimum-jerk speed profile along curved
segments plus smoothed jitter, and a short dwell on each target). As the cursor
reaches a target, that target gets "painted" with the color of the moment it was
reached. The last one reads "Fin" ("End" in English), so the video ends with the
whole path drawn and every target painted — a way to close a talk.

The output is an MP4 (needs ffmpeg, bundled via imageio-ffmpeg). If ffmpeg is
unavailable it falls back to a GIF via pillow.

Usage:
    python -m analysis.scripts.figures.final_slide_animation                 # 7 hitos + "Fin"
    python -m analysis.scripts.figures.final_slide_animation --lang en       # same in English
    python -m analysis.scripts.figures.final_slide_animation --labels "TMT,ML,SHAP,Conclusiones,Fin"
    python -m analysis.scripts.figures.final_slide_animation --layout random --seed 7
"""
import argparse
import os
import textwrap

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa: F401 — registers the crest/mako/flare colormaps
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import Circle

from analysis.scripts.figures._style import (
    ANIMATION_DPI,
    FIGURES_DIR,
    add_lang_argument,
    lang_suffix,
    use_science_style,
)
from analysis.scripts.figures.animation_common import (
    MOUSE_CURSOR_MARKER,
    frame_point_counts,
    resolve_writer,
)

use_science_style()
# Slide typography: the science style is serif (thesis figures); a humanist sans
# reads much better projected. DejaVu Sans is matplotlib's always-available
# fallback when Lato is not installed.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Lato", "DejaVu Sans"]

DPI = ANIMATION_DPI  # 12.8x7.2 in @ 150 dpi -> 1920x1080
# Large flat areas of solid color plus a thin dotted trail are a worst case for
# a low bitrate: the trial animation's 2400 kbps leaves visible ghosting inside
# the targets, so this one encodes fatter.
BITRATE = 12000

# Bigger than in the trial animation: this canvas is a full slide.
CURSOR_STYLES = {"arrow": (MOUSE_CURSOR_MARKER, 36), "dot": ("o", 17)}

# Fixed 16:9 canvas in "screen pixels": axis limits never depend on the targets,
# so the rendered frame is always exactly 16:9 with aspect="equal" (no white
# bands, no rescaling between layouts).
CANVAS_W, CANVAS_H = 1600.0, 900.0
FIG_W_IN = 12.8             # figure width; 1600 canvas px map onto these inches
TARGET_RADIUS = 130.0       # big on purpose: these are slide-sized targets
LABEL_FS = 54               # cap on the label font size (short labels)
FILL_ALPHA = 0.85
TRAIL_COLOR = "#111111"     # the path itself: always black, color lives in the nodes
# Colormaps run all the way to a near-white end; clipping it keeps the first
# points of the trail visible against the white slide (--colors gradient only).
CMAP_RANGE = (0.15, 1.0)

# Default target colors: a qualitative palette (Okabe-Ito, with its unusable
# yellow swapped for a violet) ordered so that consecutive targets never land on
# neighbouring hues. A sequential colormap makes 7 nodes look nearly identical;
# these stay distinguishable, also for color-blind viewers.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
           "#7B4FA8"]

# Curated layouts, one per number of targets: zig-zags that read left to right,
# TMT-like, keeping every pair of targets far apart (y-down screen coordinates).
# Other counts need --layout random.
DEFAULT_LAYOUTS = {
    5: np.array([
        [230.0, 620.0],
        [520.0, 200.0],
        [900.0, 560.0],
        [1320.0, 210.0],
        [1080.0, 790.0],
    ]),
    7: np.array([
        [210.0, 250.0],
        [210.0, 690.0],
        [580.0, 450.0],
        [850.0, 165.0],
        [900.0, 740.0],
        [1270.0, 390.0],
        [1400.0, 735.0],
    ]),
}

# Default label sets: the storyline of the talk, ending on "Fin".
DEFAULT_LABELS_ES = [
    "cTMT",
    "Ingeniería de atributos",
    "Validación cruzada anidada",
    "Predicción de edad",
    "Predicción de Func. Ejec.",
    "Clasificación de DCL",
    "Fin",
]
DEFAULT_LABELS_EN = [
    "cTMT",
    "Feature engineering",
    "Nested cross-val.",
    "Age prediction",
    "Exec. Function prediction",
    "MCI classification",
    "End",
]

# Trajectory generator parameters (all in seconds / pixels).
DT = 0.012                  # sampling period of the synthetic cursor
DWELL = 0.30                # pause on each target once reached
JITTER_PX = 7.0             # amplitude of the smoothed hand tremor
CURVATURE = 0.16            # bow of each segment, as a fraction of its length


def _random_positions(n, rng, radius=TARGET_RADIUS):
    """Sample ``n`` target centers inside the canvas, well separated.

    Rejection sampling with a minimum center-to-center distance so targets never
    overlap and the resulting path is readable. Falls back to the best-effort
    candidate if the constraint cannot be met (very unlikely for n=5).
    """
    margin = radius * 1.5
    min_dist = radius * 3.4
    positions = []
    for _ in range(n):
        for _attempt in range(400):
            candidate = np.array([
                rng.uniform(margin, CANVAS_W - margin),
                rng.uniform(margin, CANVAS_H - margin),
            ])
            if all(np.hypot(*(candidate - p)) >= min_dist for p in positions):
                break
        positions.append(candidate)
    return np.array(positions)


def _smoothed_noise(n, rng, window=9):
    """Zero-mean noise with a moving average applied, scaled to unit amplitude.

    Raw gaussian noise looks like static; smoothing it turns it into the slow
    wobble of a hand holding a mouse.
    """
    if n == 0:
        return np.zeros(0)
    raw = rng.normal(0.0, 1.0, n + window)
    kernel = np.ones(window) / window
    smooth = np.convolve(raw, kernel, mode="valid")[:n]
    peak = np.abs(smooth).max()
    return smooth / peak if peak > 0 else smooth


def _segment_points(start, end, rng):
    """Points of one target-to-target movement (start excluded, end included).

    Shape: a quadratic Bezier bowed perpendicular to the straight line, so the
    path curves like a real hand movement instead of running along a ruler.
    Timing: a minimum-jerk profile (10u^3 - 15u^4 + 6u^5), which accelerates and
    decelerates smoothly and therefore leaves points densely packed near both
    targets and sparse in the middle — exactly what a real cursor trail does.
    Duration grows with distance (a crude Fitts-like law).
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    delta = end - start
    dist = float(np.hypot(*delta))
    n = max(2, int(round((0.30 + dist / 900.0) / DT)))

    # Control point: midpoint pushed along the perpendicular, side alternating at
    # random so consecutive segments do not all bow the same way.
    perpendicular = np.array([-delta[1], delta[0]]) / (dist or 1.0)
    bow = CURVATURE * dist * rng.uniform(0.6, 1.0) * rng.choice([-1.0, 1.0])
    control = (start + end) / 2.0 + perpendicular * bow

    t = np.linspace(0, 1, n + 1)[1:]  # start excluded: it is the previous point
    u = (10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5)[:, None]
    curve = (1 - u) ** 2 * start + 2 * (1 - u) * u * control + u ** 2 * end

    # Tremor tapered by sin(pi*u): zero at both ends, so the trail starts and
    # lands exactly on the target centers.
    taper = np.sin(np.pi * u[:, 0])
    curve[:, 0] += _smoothed_noise(n, rng) * JITTER_PX * taper
    curve[:, 1] += _smoothed_noise(n, rng) * JITTER_PX * taper
    return curve


def _dwell_points(center, rng):
    """Points of the short pause on a target (tiny tremor around its center)."""
    n = max(1, int(round(DWELL / DT)))
    jitter = np.column_stack([_smoothed_noise(n, rng), _smoothed_noise(n, rng)]) * 3.0
    return np.asarray(center, float) + jitter


def _synthetic_trail(positions, rng):
    """Build the whole synthetic trail.

    Returns ``(x, y, elapsed, arrivals, owner)`` where ``arrivals[i]`` is the
    index of the first point that sits on target ``i`` (when it gets painted)
    and ``owner[j]`` is the target each point is heading to (used to color the
    trail by leg). Sampling is at a constant ``DT``, so revealing points linearly
    in time reproduces the simulated speed profile.
    """
    points, owners, arrivals = [], [], []
    for i, center in enumerate(positions):
        if i > 0:
            points.append(_segment_points(positions[i - 1], center, rng))
            owners.append(np.full(len(points[-1]), i))
        arrivals.append(sum(len(chunk) for chunk in points))
        points.append(_dwell_points(center, rng))
        owners.append(np.full(len(points[-1]), i))

    trail = np.vstack(points)
    elapsed = np.arange(len(trail)) * DT
    return trail[:, 0], trail[:, 1], elapsed, arrivals, np.concatenate(owners)


def _fit_label(label, radius=TARGET_RADIUS):
    """Wrap a label and pick the largest font size that keeps it inside a target.

    Tries every line width (never splitting a word) and keeps the wrap that
    allows the biggest font. The text block must fit inside the circle, so its
    half-diagonal — not just its width — is what is bounded: the corners of the
    block have to stay within 0.95 R. Character width and line height are sans
    approximations (0.55 em wide, 1.20 em tall).
    """
    radius_pt = radius * (72 * FIG_W_IN / CANVAS_W)
    best = (0.0, label)
    for width in range(1, len(label) + 1):
        lines = textwrap.wrap(label, width=width, break_long_words=False,
                              break_on_hyphens=False) or [label]
        half_diagonal_per_pt = 0.5 * np.hypot(0.55 * max(len(l) for l in lines),
                                              1.20 * len(lines))
        fontsize = min(LABEL_FS, 0.95 * radius_pt / half_diagonal_per_pt)
        if fontsize > best[0]:
            best = (fontsize, "\n".join(lines))
    return best[1], best[0]


def _clipped_cmap(name, span=CMAP_RANGE):
    """The colormap restricted to ``span``, so neither end blends into white."""
    base = plt.get_cmap(name)
    return ListedColormap(base(np.linspace(*span, 256)), name=f"{name}_clipped")


def _softened(rgba):
    """The color lightened as if drawn with FILL_ALPHA over the white slide.

    Targets are filled opaquely with this precomputed blend instead of using
    alpha, so the black trail disappears behind a painted target (alpha is left
    free for the paint fade-in).
    """
    return tuple(FILL_ALPHA * np.asarray(rgba[:3]) + (1 - FILL_ALPHA)) + (1.0,)


def _text_color(rgba):
    """Black or white, whichever gives more WCAG contrast on top of ``rgba``.

    Relative luminance needs the sRGB gamma expansion — skipping it (a plain
    weighted sum of the channels) misjudges mid greens and teals.
    """
    channels = np.asarray(rgba[:3])
    linear = np.where(channels <= 0.04045, channels / 12.92,
                      ((channels + 0.055) / 1.055) ** 2.4)
    luminance = float(np.dot([0.2126, 0.7152, 0.0722], linear))
    contrast_black = (luminance + 0.05) / 0.05
    contrast_white = 1.05 / (luminance + 0.05)
    return "black" if contrast_black >= contrast_white else "white"


def _resolve_labels(labels, lang):
    if labels:
        return [label.strip() for label in labels.split(",") if label.strip()]
    return DEFAULT_LABELS_ES if lang == "es" else DEFAULT_LABELS_EN


def main(lang="es", labels=None, layout="fixed", seed=7, colors="palette",
         trail="black", cmap="crest", fps=30, seconds=16.0, hold=3.0, fmt="mp4",
         cursor="arrow", dpi=DPI):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    labels = _resolve_labels(labels, lang)
    rng = np.random.default_rng(seed)
    if layout == "random":
        positions = _random_positions(len(labels), rng)
    elif len(labels) in DEFAULT_LAYOUTS:
        positions = DEFAULT_LAYOUTS[len(labels)]
    else:
        raise ValueError(
            f"--layout fixed only has curated layouts for {sorted(DEFAULT_LAYOUTS)} "
            f"targets but {len(labels)} labels were given; use --layout random"
        )

    x, y, elapsed, arrivals, owner = _synthetic_trail(positions, rng)
    total_time = float(elapsed.max())
    print(f"Synthetic trail: {len(x)} points  {total_time:.2f} s simulated  "
          f"targets={labels}")

    # Clean slide look: no axes, no colorbar, white background, everything big.
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_W_IN * CANVAS_H / CANVAS_W))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)  # y-down, as in the real trial figures
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # One color per target: the qualitative palette by default (adjacent nodes
    # stay clearly different), or a sequential colormap sampled over time with
    # --colors gradient. The trail itself is black (--trail node paints each leg
    # with the color of the target it is heading to instead).
    if colors == "palette":
        target_colors = [_softened(to_rgba(PALETTE[i % len(PALETTE)]))
                         for i in range(len(labels))]
    else:
        colormap = _clipped_cmap(cmap)
        target_colors = [_softened(colormap(elapsed[a] / total_time if total_time else 0.0))
                         for a in arrivals]
    if trail == "black":
        point_colors = np.tile(to_rgba(TRAIL_COLOR), (len(owner), 1))
    else:
        point_colors = np.array([target_colors[i] for i in owner])

    # Targets start empty — a white circle with a thin outline and no text — and
    # get painted (fill + name fading in) when the cursor reaches them.
    targets = []
    for label, center, arrival, color in zip(labels, positions, arrivals, target_colors):
        # Targets sit ON TOP of the trail: the path visibly runs behind them
        # instead of crossing their labels.
        circle = Circle(center, TARGET_RADIUS, facecolor="white", edgecolor="black",
                        linewidth=2.5, alpha=1.0, zorder=4)
        ax.add_patch(circle)
        # Long labels are wrapped and shrunk so they stay inside the circle.
        wrapped, fontsize = _fit_label(label)
        text = ax.text(center[0], center[1], wrapped, ha="center", va="center",
                       linespacing=1.15, fontsize=fontsize, fontweight="bold",
                       color=_text_color(color), alpha=0.0, zorder=6)
        targets.append({"circle": circle, "text": text, "arrival": arrival,
                        "color": color, "fontsize": fontsize})

    line, = ax.plot([], [], color=TRAIL_COLOR, lw=1.3, alpha=0.35,
                    zorder=2, solid_capstyle="round")
    scatter = ax.scatter([], [], s=34, alpha=0.95, linewidths=0, zorder=3)
    cursor_marker, cursor_size = CURSOR_STYLES[cursor]
    current, = ax.plot([], [], marker=cursor_marker, markersize=cursor_size,
                       markerfacecolor="white", markeredgecolor="black",
                       markeredgewidth=1.8, linestyle="None", zorder=7)

    reveal_frames = max(1, round(fps * seconds))
    hold_frames = max(0, round(fps * hold))
    counts = frame_point_counts(len(x), reveal_frames, hold_frames)
    # A target finishes being painted well within its dwell, so every target —
    # including the last one, which has no outgoing segment — ends fully opaque
    # and hides the trail running behind it.
    fade_points = max(1, int(round(0.75 * DWELL / DT)))

    def update(frame_idx):
        k = counts[frame_idx]
        scatter.set_offsets(np.column_stack([x[:k], y[:k]]))
        scatter.set_facecolors(point_colors[:k])
        line.set_data(x[:k], y[:k])
        # The cursor leads the trail while it grows, then leaves the screen for
        # the final hold so the closing frame is just the path and the targets.
        if frame_idx < reveal_frames:
            current.set_data([x[k - 1]], [y[k - 1]])
        else:
            current.set_data([], [])

        for target in targets:
            progress = np.clip((k - target["arrival"]) / fade_points, 0.0, 1.0)
            if progress <= 0:
                continue
            eased = progress ** 0.6
            circle = target["circle"]
            circle.set_facecolor(target["color"])
            circle.set_alpha(eased)
            circle.set_linewidth(2.5 + 3.0 * eased)
            # Brief "pop": the target swells and settles back to its size.
            circle.set_radius(TARGET_RADIUS * (1.0 + 0.18 * np.sin(np.pi * progress)))
            # The name is revealed with the paint: before being reached the
            # target is an anonymous empty circle.
            target["text"].set_alpha(eased)

        return ([scatter, line, current]
                + [t["circle"] for t in targets] + [t["text"] for t in targets])

    anim = animation.FuncAnimation(
        fig, update, frames=len(counts), interval=1000.0 / fps, blit=False
    )

    base = os.path.join(
        FIGURES_DIR,
        f"fig_final_slide_animation_{len(labels)}targets_{seconds:g}s{lang_suffix(lang)}"
    )
    out_path, writer = resolve_writer(base, fmt, fps, bitrate=BITRATE)
    anim.save(out_path, writer=writer, dpi=dpi)
    print(f"Saved -> {out_path}  ({len(counts)} frames @ {fps} fps)")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animate a synthetic cTMT-style trail over big labelled targets (closing slide)"
    )
    add_lang_argument(parser)
    parser.add_argument("--labels", default=None,
                        help="Etiquetas separadas por coma; determinan la cantidad de "
                             "targets (default: los 7 hitos de la charla, terminando en Fin)")
    parser.add_argument("--layout", choices=["fixed", "random"], default="fixed",
                        help="Posiciones fijas (curadas) o aleatorias segun --seed (default: fixed)")
    parser.add_argument("--seed", type=int, default=7,
                        help="Semilla del temblor y del layout aleatorio (default: 7)")
    parser.add_argument("--colors", choices=["palette", "gradient"], default="palette",
                        help="Colores de los nodos: paleta cualitativa o degradado temporal (default: palette)")
    parser.add_argument("--trail", choices=["black", "node"], default="black",
                        help="Color del recorrido: negro o el del nodo destino (default: black)")
    parser.add_argument("--cmap", default="crest",
                        help="Colormap para --colors gradient (default: crest; probar mako, flare, plasma)")
    parser.add_argument("--fps", type=int, default=30, help="Cuadros por segundo (default: 30)")
    parser.add_argument("--seconds", type=float, default=16.0,
                        help="Duracion del barrido en segundos (default: 16)")
    parser.add_argument("--hold", type=float, default=3.0,
                        help="Segundos que queda la imagen final completa (default: 3)")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4", dest="fmt",
                        help="Formato de salida (default: mp4; cae a gif si no hay ffmpeg)")
    parser.add_argument("--dpi", type=int, default=DPI,
                        help=f"Resolucion: {DPI} da 1920x1080; bajarla aliviana el gif (default: {DPI})")
    parser.add_argument("--cursor", choices=["arrow", "dot"], default="arrow",
                        help="Marcador de la posicion actual (default: arrow)")
    args = parser.parse_args()
    main(args.lang, args.labels, args.layout, args.seed, args.colors, args.trail,
         args.cmap, args.fps, args.seconds, args.hold, args.fmt, args.cursor,
         args.dpi)
