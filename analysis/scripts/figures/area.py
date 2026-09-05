"""Plot the area_difference_from_ideal metric against the ideal trajectory.

neurotask computes area_difference_from_ideal (area_calculation.py) as
np.trapz(perpendicular_distances, line_positions): the integral of the absolute
perpendicular distance between the real trail and the straight "ideal" line,
taken against the coordinate projected onto that line. It is NOT the geometric
area enclosed between the two curves, and the ideal line joins the first and
last cursor point of each between-targets segment, not the target centers.

Because the projected coordinate is not monotonic (the cursor can move backwards
along the ideal line), np.trapz subtracts area on those stretches. This figure
makes that explicit: the left panel shades the integrated area in screen
coordinates, and --rectified adds a right panel showing the same segment as
np.trapz sees it (position along the ideal line vs. perpendicular distance),
with the subtracting stretches hatched in red.

Usage:
    python -m analysis.scripts.figures.area                  # castellano
    python -m analysis.scripts.figures.area --lang en        # inglés
    python -m analysis.scripts.figures.area --segment D      # destacar el target D
    python -m analysis.scripts.figures.area --rectified      # + panel rectificado
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from neurotask.tmt.metrics.targets_touched import get_all_trails_between_targets
from src import config

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

C_ADD = "#1F77B4"
C_SUB = "#D62728"
C_TRAIL = "#333333"
C_IDEAL = "#000000"
C_FAINT = "#cccccc"

LABELS = {
    "es": {
        "s_label": "Posición sobre la recta ideal (px)",
        "d_label": "Distancia perpendicular (px)",
        "adds": "Área que suma",
        "subtracts": "Área que resta (retroceso)",
        "distance": "Trayectoria real",
        "ideal": "Trayectoria ideal",
        "area_subject": "Área del sujeto",
        "segment": "Segmento",
        "net": "Área neta (métrica)",
        "uncancelled": "Sin cancelación",
        "cancelled": "Cancelada",
    },
    "en": {
        "s_label": "Position along ideal line (px)",
        "d_label": "Perpendicular distance (px)",
        "adds": "Area added",
        "subtracts": "Area subtracted (backtracking)",
        "distance": "Real trail",
        "ideal": "Ideal trail",
        "area_subject": "Subject's area",
        "segment": "Segment",
        "net": "Net area (metric)",
        "uncancelled": "Without cancellation",
        "cancelled": "Cancelled",
    },
}


def _segment_geometry(pts):
    """Project a segment onto its ideal line, mirroring area_calculation.py.

    Args:
        pts: (N, 2) array of real cursor coordinates.

    Returns:
        (projections, s, d, trapezoids) where s is the coordinate along the
        ideal line, d the perpendicular distance, and trapezoids the signed
        per-step contributions whose sum equals np.trapz(d, s) — the metric.
        None for degenerate segments (zero-length ideal line).
    """
    start, end = pts[0], pts[-1]
    ideal_vector = end - start
    ideal_length_sq = float(np.dot(ideal_vector, ideal_vector))
    if ideal_length_sq == 0.0 or len(pts) < 2:
        return None

    projection_factors = (pts - start) @ ideal_vector / ideal_length_sq
    projections = start + projection_factors[:, None] * ideal_vector
    d = np.linalg.norm(pts - projections, axis=1)
    s = projection_factors * np.sqrt(ideal_length_sq)
    trapezoids = (d[:-1] + d[1:]) / 2.0 * np.diff(s)

    return projections, s, d, trapezoids


def _area_quads(pts, projections):
    """Quadrilaterals between each real step and its projection on the ideal line."""
    return [
        [pts[i], pts[i + 1], projections[i + 1], projections[i]]
        for i in range(len(pts) - 1)
    ]


def _rectified_quads(s, d):
    """Trapezoids under d(s) — exactly the terms np.trapz sums."""
    return [
        [(s[i], 0.0), (s[i], d[i]), (s[i + 1], d[i + 1]), (s[i + 1], 0.0)]
        for i in range(len(s) - 1)
    ]


def collect_segments(trial, target_radius):
    """Geometry and areas for every between-targets segment of the trial."""
    trails = get_all_trails_between_targets(
        trial, target_radius, config.TARGET_RADIUS_MULTIPLIER)

    segments = []
    for target, cursor_trail in trails:
        if target is None:
            continue
        pts = np.array([[c.position.x, c.position.y] for c in cursor_trail], dtype=float)
        geometry = _segment_geometry(pts)
        if geometry is None:
            continue
        projections, s, d, trapezoids = geometry
        segments.append({
            "target": target.content,
            "pts": pts,
            "projections": projections,
            "s": s,
            "d": d,
            "trapezoids": trapezoids,
            "net_area": float(trapezoids.sum()),
            "abs_area": float(np.abs(trapezoids).sum()),
        })
    return segments


def _draw_spatial(ax, trial, subject, segments, highlighted, labels, lang, fonts,
                  excluded=(), big_fonts=False):
    """Draw the trial with each segment's integrated area shaded.

    Segments listed in `excluded` keep their faint trail but are not shaded, so
    one oversized segment cannot swamp the rest of the figure.
    """
    _cursor_trail, cursor_x, cursor_y = trial_data.cursor_coordinates(trial)
    ax.plot(cursor_x, cursor_y, color=C_FAINT, lw=1.0, alpha=0.8, zorder=2)

    for segment in segments:
        if segment["target"] in excluded:
            continue
        quads = _area_quads(segment["pts"], segment["projections"])
        colors = [C_SUB if trapezoid < 0 else C_ADD for trapezoid in segment["trapezoids"]]
        ax.add_collection(PolyCollection(quads, facecolors=colors, edgecolors="none",
                                         alpha=0.35, zorder=3))
        # The ideal line joins the first and last cursor point of the segment,
        # not the target centers.
        ax.plot([segment["pts"][0][0], segment["pts"][-1][0]],
                [segment["pts"][0][1], segment["pts"][-1][1]],
                color=C_IDEAL, lw=1.0, ls="--", alpha=0.7, zorder=4)

    ax.plot(highlighted["pts"][:, 0], highlighted["pts"][:, 1],
            color=C_TRAIL, lw=1.6, zorder=5)

    trial_data.draw_trial_background(ax, trial, subject, cursor_x, cursor_y, lang, fonts,
                                     circle_linewidth=1.1, zorder_circle=6, zorder_text=7)

    handles = [
        plt.Line2D([0], [0], color=C_IDEAL, lw=1.0, ls="--", alpha=0.7,
                   label=labels["ideal"]),
        Patch(facecolor=C_ADD, alpha=0.35, edgecolor="none", label=labels["area_subject"]),
    ]
    ax.legend(**trial_data.legend_kwargs(
        fonts, outside=big_fonts, handles=handles,
        **({} if big_fonts else {"loc": "lower left"}),
    ))


def _draw_rectified(ax, segment, labels, fonts, big_fonts=False):
    s, d = segment["s"], segment["d"]
    quads = _rectified_quads(s, d)
    adding = [q for q, t in zip(quads, segment["trapezoids"]) if t >= 0]
    subtracting = [q for q, t in zip(quads, segment["trapezoids"]) if t < 0]

    if adding:
        ax.add_collection(PolyCollection(adding, facecolors=C_ADD, edgecolors="none",
                                         alpha=0.35, zorder=2, label=labels["adds"]))
    if subtracting:
        ax.add_collection(PolyCollection(subtracting, facecolors=C_SUB, edgecolors=C_SUB,
                                         alpha=0.45, hatch="///", lw=0.0, zorder=3,
                                         label=labels["subtracts"]))

    ax.plot(s, d, color=C_TRAIL, lw=1.2, zorder=4, label=labels["distance"])
    ax.axhline(0.0, color=C_IDEAL, lw=1.2, ls="--", zorder=4, label=labels["ideal"])

    ax.set_xlabel(labels["s_label"])
    ax.set_ylabel(labels["d_label"])
    ax.set_xlim(min(s.min(), 0) - 10, s.max() + 10)
    ax.set_ylim(-0.04 * d.max(), d.max() * 1.28)
    ax.legend(**trial_data.legend_kwargs(fonts, outside=big_fonts, loc="upper left"))

    cancelled = segment["abs_area"] - segment["net_area"]
    ax.text(0.98, 0.97,
            f"{labels['segment']}: {segment['target']}\n"
            f"{labels['net']}: {segment['net_area']:,.0f} px²\n"
            f"{labels['uncancelled']}: {segment['abs_area']:,.0f} px²\n"
            f"{labels['cancelled']}: {cancelled:,.0f} px²",
            transform=ax.transAxes, ha="right", va="top", fontsize=fonts.legend,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9))


def _resolve_highlight(segments, shown, segment_target):
    if segment_target is None:
        # Pick among the shaded ones so the highlight matches what is drawn.
        return max(shown, key=lambda s: abs(s["net_area"]))
    match = next((s for s in segments if s["target"] == segment_target), None)
    if match is None:
        available = ", ".join(s["target"] for s in segments)
        raise ValueError(f"Unknown segment '{segment_target}'. Available: {available}")
    return match


def _filter_segments(segments, excluded):
    unknown = [name for name in excluded if not any(s["target"] == name for s in segments)]
    if unknown:
        available = ", ".join(s["target"] for s in segments)
        raise ValueError(f"Unknown segment(s) to exclude: {unknown}. Available: {available}")
    shown = [s for s in segments if s["target"] not in excluded]
    if not shown:
        raise ValueError("All segments were excluded; nothing left to shade")
    return shown


def main(lang="es", subject_id=None, trial_id=None, segment_target=None,
         excluded=(), show_rectified=False, big_fonts=False):
    fonts = TRIAL_FONTS.scaled() if big_fonts else TRIAL_FONTS
    labels = LABELS[lang]
    excluded = tuple(excluded)

    row, subject, trial = trial_data.load_trial(subject_id, trial_id)
    segments = collect_segments(trial, subject.target_radius)
    if not segments:
        raise ValueError(f"No valid between-targets segments for trial {row['trial_id']}")

    shown = _filter_segments(segments, excluded)
    highlighted = _resolve_highlight(segments, shown, segment_target)

    metric = float(np.mean([s["net_area"] for s in segments]))
    n_backtracking = sum(1 for s in segments if (s["trapezoids"] < 0).any())
    print(f"area_difference_from_ideal (mean over {len(segments)} segments) = {metric:,.2f} px²")
    print(f"segments with backtracking along the ideal line: {n_backtracking}/{len(segments)}")
    if excluded:
        # Exclusion only affects what is drawn; the metric above is unchanged.
        print(f"segments hidden from the plot (metric unaffected): {', '.join(excluded)}")
    print(f"highlighted segment -> {highlighted['target']}: "
          f"net={highlighted['net_area']:,.1f}  uncancelled={highlighted['abs_area']:,.1f}")

    if show_rectified:
        fig, (ax_spatial, ax_rectified) = plt.subplots(1, 2, figsize=(14, 7))
        _draw_rectified(ax_rectified, highlighted, labels, fonts, big_fonts)
        trial_data.apply_axis_fonts(ax_rectified, fonts)
    else:
        fig, ax_spatial = plt.subplots(figsize=(7, 7))
    _draw_spatial(ax_spatial, trial, subject, segments, highlighted, labels, lang,
                  fonts, excluded, big_fonts)

    # tight_layout fights with legends anchored outside the axes (--big-fonts):
    # it shrinks the axes to fit them, squishing the trial. bbox_inches="tight"
    # at save time already includes the external legend.
    if not big_fonts:
        fig.tight_layout()

    name = "fig_tmt_area_from_ideal"
    if excluded:
        name += "_excl-" + "-".join(excluded)
    if show_rectified:
        name += "_rect"
    name += f"_{trial_data.trial_slug(row)}"
    if big_fonts:
        name += "_big"
    save_fig(fig, f"{name}{lang_suffix(lang)}", dpi=TRIAL_DPI)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot area_difference_from_ideal against the ideal trajectory")
    add_lang_argument(parser)
    parser.add_argument("--subject", default=None, help="Override subject_id")
    parser.add_argument("--trial", default=None, help="Override trial_id")
    parser.add_argument("--segment", default=None,
                        help="Contenido del target que cierra el segmento a destacar "
                             "(default: el de mayor área entre los dibujados)")
    parser.add_argument("--exclude-segments", nargs="*", default=[], metavar="TARGET",
                        help="Targets cuyo segmento no se sombrea, para que uno muy grande "
                             "no tape al resto (ej: --exclude-segments D). No afecta la métrica.")
    parser.add_argument("--rectified", action="store_true",
                        help="Agrega el panel derecho rectificado (s vs distancia perpendicular)")
    parser.add_argument("--big-fonts", action="store_true",
                        help="Agranda las fuentes para presentaciones")
    args = parser.parse_args()
    main(args.lang, args.subject, args.trial, args.segment,
         args.exclude_segments, args.rectified, args.big_fonts)
