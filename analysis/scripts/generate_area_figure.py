"""Plot the area_difference_from_ideal metric against the ideal trajectory.

neurotask computes area_difference_from_ideal (area_calculation.py) as
np.trapz(perpendicular_distances, line_positions): the integral of the absolute
perpendicular distance between the real trail and the straight "ideal" line,
taken against the coordinate projected onto that line. It is NOT the geometric
area enclosed between the two curves, and the ideal line joins the first and
last cursor point of each between-targets segment, not the target centers.

Because the projected coordinate is not monotonic (the cursor can move backwards
along the ideal line), np.trapz subtracts area on those stretches. This figure
makes that explicit.

Left panel:  the trial in screen coordinates, with each segment's ideal line and
             the integrated area shaded between real trail and ideal line.
Right panel: the highlighted segment rectified — x = position along the ideal
             line, y = perpendicular distance — which is literally what np.trapz
             integrates. Stretches that subtract area are drawn in red hatching.

Usage:
    python -m analysis.scripts.generate_area_figure                  # inglés
    python -m analysis.scripts.generate_area_figure --lang es        # castellano
    python -m analysis.scripts.generate_area_figure --segment D      # destacar target D
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 — registers styles
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from src import config
from src.mapper.datapruebas.datapruebas_mapper import DatapruebasTMTMapper
from neurotask.tmt.metrics.targets_touched import get_all_trails_between_targets
from src.visualization.trial_plotting_helpers import draw_trial_targets, configure_trial_axes
# Reuse the segmentation figure's trial selection so all figures show the same trial.
from analysis.scripts.generate_segmentation_figure import (
    FIGURES_DIR,
    RAW_EXPERIMENT_PATH,
    _latest_tmt_analysis_path,
    _select_trial,
)

plt.style.use(["science", "no-latex"])
# match generate_segmentation_figure / generate_speed_figure so --big-fonts
# yields the same label sizes across the three figures
LABEL_FS = 18
TICK_FS = 15
LEGEND_FS = 15
TARGET_FS = 15
DPI = 600

# --big-fonts multiplies every font size by this factor (for slides/presentations).
FONT_SCALE = 1.4

C_ADD = "#1F77B4"      # tramos que suman area
C_SUB = "#D62728"      # tramos que restan area (retroceso sobre la ideal)
C_TRAIL = "#333333"
C_IDEAL = "#000000"
C_FAINT = "#cccccc"


def _segment_geometry(pts):
    """Project a segment onto its ideal line, mirroring area_calculation.py.

    Args:
        pts: (N, 2) array of real cursor coordinates.

    Returns:
        (projections, s, d, trapezoids) where s is the coordinate along the
        ideal line, d the perpendicular distance, and trapezoids the signed
        per-step contributions whose sum equals np.trapz(d, s) — the metric.
        Returns None for degenerate segments (zero-length ideal line).
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


def _collect_segments(trial, target_radius):
    """Geometry and areas for every between-targets segment of the trial."""
    trails = get_all_trails_between_targets(trial, target_radius, config.TARGET_RADIUS_MULTIPLIER)

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


def _draw_spatial(ax, trial, subject, segments, highlighted, labels, excluded=(),
                  big_fonts=False):
    """Draw the trial with each segment's integrated area shaded.

    Segments listed in `excluded` keep their faint trail but are not shaded,
    so one oversized segment cannot swamp the rest of the figure.
    """
    cursor_trail = trial.get_cursor_trail_from_start()
    cursor_x = [p.position.x for p in cursor_trail]
    cursor_y = [p.position.y for p in cursor_trail]

    ax.plot(cursor_x, cursor_y, color=C_FAINT, lw=1.0, alpha=0.8, zorder=2)

    for segment in segments:
        if segment["target"] in excluded:
            continue
        quads = _area_quads(segment["pts"], segment["projections"])
        colors = [C_SUB if trapezoid < 0 else C_ADD for trapezoid in segment["trapezoids"]]
        ax.add_collection(PolyCollection(quads, facecolors=colors, edgecolors="none",
                                         alpha=0.35, zorder=3))
        # ideal line: first to last cursor point of the segment, not center to center
        ax.plot([segment["pts"][0][0], segment["pts"][-1][0]],
                [segment["pts"][0][1], segment["pts"][-1][1]],
                color=C_IDEAL, lw=1.0, ls="--", alpha=0.7, zorder=4)

    ax.plot(highlighted["pts"][:, 0], highlighted["pts"][:, 1],
            color=C_TRAIL, lw=1.6, zorder=5)

    draw_trial_targets(ax, trial, subject.target_radius, circle_color="black",
                       circle_alpha=0.9, circle_linewidth=1.1,
                       text_fontsize=TARGET_FS, text_color="black",
                       zorder_circle=6, zorder_text=7)
    configure_trial_axes(ax, x=cursor_x, y=cursor_y, show_labels=True,
                         xlabel=labels["xlabel"], ylabel=labels["ylabel"])

    # legend: dashed line = ideal trajectory, light-blue shading = subject's area
    handles = [
        plt.Line2D([0], [0], color=C_IDEAL, lw=1.0, ls="--", alpha=0.7,
                   label=labels["ideal"]),
        Patch(facecolor=C_ADD, alpha=0.35, edgecolor="none",
              label=labels["area_subject"]),
    ]
    legend_kwargs = dict(handles=handles, fontsize=LEGEND_FS,
                         framealpha=0.9, edgecolor="#cccccc")
    if big_fonts:
        # move the legend outside the plot, upper-right, so it doesn't cover data
        legend_kwargs.update(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    else:
        legend_kwargs.update(loc="lower left")
    ax.legend(**legend_kwargs)


def _draw_rectified(ax, segment, labels, big_fonts=False):
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
    legend_kwargs = dict(loc="upper left", fontsize=LEGEND_FS,
                         framealpha=0.9, edgecolor="#cccccc")
    if big_fonts:
        # move the legend outside the plot, upper-right, so it doesn't cover data
        legend_kwargs.update(bbox_to_anchor=(1.02, 1.0))
    ax.legend(**legend_kwargs)

    cancelled = segment["abs_area"] - segment["net_area"]
    ax.text(0.98, 0.97,
            f"{labels['segment']}: {segment['target']}\n"
            f"{labels['net']}: {segment['net_area']:,.0f} px²\n"
            f"{labels['uncancelled']}: {segment['abs_area']:,.0f} px²\n"
            f"{labels['cancelled']}: {cancelled:,.0f} px²",
            transform=ax.transAxes, ha="right", va="top", fontsize=LEGEND_FS,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9))


def main(lang="en", subject_id=None, trial_id=None, segment_target=None,
         excluded=(), show_rectified=False, big_fonts=False):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if big_fonts:
        global LABEL_FS, TICK_FS, LEGEND_FS, TARGET_FS
        LABEL_FS = round(LABEL_FS * FONT_SCALE)
        TICK_FS = round(TICK_FS * FONT_SCALE)
        LEGEND_FS = round(LEGEND_FS * FONT_SCALE)
        TARGET_FS = round(TARGET_FS * FONT_SCALE)

    df_tmt = pd.read_csv(_latest_tmt_analysis_path(), on_bad_lines="warn")
    row = _select_trial(df_tmt, subject_id, trial_id)
    subject_id = row["subject_id"]
    trial_id = row["trial_id"]
    print(f"Selected: subject={subject_id}  trial={trial_id}")

    experiment = DatapruebasTMTMapper().map(RAW_EXPERIMENT_PATH)
    subject = experiment.subjects[subject_id]
    trial = next(t for t in subject.testing_trials if t.id == trial_id)

    segments = _collect_segments(trial, subject.target_radius)
    if not segments:
        raise ValueError(f"No valid between-targets segments for trial {trial_id}")

    excluded = tuple(excluded)
    unknown = [name for name in excluded if not any(s["target"] == name for s in segments)]
    if unknown:
        available = ", ".join(s["target"] for s in segments)
        raise ValueError(f"Unknown segment(s) to exclude: {unknown}. Available: {available}")
    shown = [s for s in segments if s["target"] not in excluded]
    if not shown:
        raise ValueError("All segments were excluded; nothing left to shade")

    if segment_target is not None:
        highlighted = next(s for s in segments if s["target"] == segment_target)
    else:
        # pick among the shaded ones so the highlight matches what is drawn
        highlighted = max(shown, key=lambda s: abs(s["net_area"]))

    metric = float(np.mean([s["net_area"] for s in segments]))
    n_backtracking = sum(1 for s in segments if (s["trapezoids"] < 0).any())
    print(f"area_difference_from_ideal (mean over {len(segments)} segments) = {metric:,.2f} px²")
    print(f"segments with backtracking along the ideal line: {n_backtracking}/{len(segments)}")
    if excluded:
        # the metric above is unchanged: exclusion only affects what is drawn
        print(f"segments hidden from the plot (metric unaffected): {', '.join(excluded)}")
    print(f"highlighted segment -> {highlighted['target']}: "
          f"net={highlighted['net_area']:,.1f}  uncancelled={highlighted['abs_area']:,.1f}")

    es = lang == "es"
    labels = {
        "xlabel": "Coordenada X (px)" if es else "X Screen Coordinate (px)",
        "ylabel": "Coordenada Y (px)" if es else "Y Screen Coordinate (px)",
        "s_label": "Posición sobre la recta ideal (px)" if es else "Position along ideal line (px)",
        "d_label": "Distancia perpendicular (px)" if es else "Perpendicular distance (px)",
        "adds": "Área que suma" if es else "Area added",
        "subtracts": "Área que resta (retroceso)" if es else "Area subtracted (backtracking)",
        "distance": "Trayectoria real" if es else "Real trail",
        "ideal": "Trayectoria ideal" if es else "Ideal trail",
        "area_subject": "Área del sujeto" if es else "Subject's area",
        "segment": "Segmento" if es else "Segment",
        "net": "Área neta (métrica)" if es else "Net area (metric)",
        "uncancelled": "Sin cancelación" if es else "Without cancellation",
        "cancelled": "Cancelada" if es else "Cancelled",
    }

    if show_rectified:
        fig, (ax_spatial, ax_rectified) = plt.subplots(1, 2, figsize=(14, 7))
        _draw_rectified(ax_rectified, highlighted, labels, big_fonts)
        axes = (ax_spatial, ax_rectified)
    else:
        fig, ax_spatial = plt.subplots(figsize=(7, 7))
        axes = (ax_spatial,)
    _draw_spatial(ax_spatial, trial, subject, segments, highlighted, labels, excluded,
                  big_fonts)

    for ax in axes:
        ax.xaxis.label.set_fontsize(LABEL_FS)
        ax.yaxis.label.set_fontsize(LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
    # No title (per spec)

    # tight_layout fights with legends anchored outside the axes (big_fonts): it
    # shrinks the axes to fit them, squishing the trial. bbox_inches="tight" at
    # save time already includes the external legend, so skip it there.
    if not big_fonts:
        fig.tight_layout()
    # subject and trial go in the filename so the figure is traceable to its source;
    # variants too, so they don't overwrite each other
    name = "fig_tmt_area_from_ideal"
    if excluded:
        name += "_excl-" + "-".join(excluded)
    if show_rectified:
        name += "_rect"
    name += f"_{subject_id[:8]}_{trial_id}"
    if big_fonts:
        name += "_big"
    if es:
        name += "_es"
    base = os.path.join(FIGURES_DIR, name)
    fig.savefig(f"{base}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")  # vectorial
    print(f"Saved -> {base}.png  (+ .pdf)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot area_difference_from_ideal against the ideal trajectory")
    parser.add_argument("--lang", choices=["en", "es"], default="en",
                        help="Idioma de ejes/leyenda (default: en)")
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
