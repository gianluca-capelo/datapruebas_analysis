"""Generate the K_mean SHAP importance figures for the talk slides.

Two standalone images for the K_mean model (RandomForestRegressor), tailored for
projection rather than for the thesis: no title, large fonts and feature names
in Spanish.

    slide -> fig3_shap_k_mean_presentacion.png
             Top 3 features, metric names written INSIDE the bars.
    panel -> fig3_shap_k_mean_panel_presentacion.png
             Same layout as `shap_panels` (top 15, names on the y axis) with the
             slide typography.

Usage:
    python -m analysis.scripts.figures.shap_k_mean_slides                  # ambas
    python -m analysis.scripts.figures.shap_k_mean_slides --figure panel
    python -m analysis.scripts.figures.shap_k_mean_slides --figure slide --override
    python -m analysis.scripts.figures.shap_k_mean_slides --from-csv       # sin recomputar
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analysis.scripts.figures import shap_common
from analysis.scripts.figures._style import PRINT_DPI, save_fig, use_slide_style

use_slide_style()
# The feature names on the y axis are the content of this figure, so they run a
# step larger than the shared slide typography.
plt.rcParams["ytick.labelsize"] = 20

TOP_N_SLIDE = 3
TOP_N_PANEL = 15
ANNOT_FS = 20

# Fraction of the longest bar used to inset the metric name from the bar start.
NAME_INSET = 0.02
# x value the finished slide axis is cropped back to (see _slide_figure).
SLIDE_X_CUT = 0.05

# ---------------------------------------------------------------------------
# Spanish metric labels.
#
# Keys are the English base metric names produced by FEATURE_LABELS, i.e. the
# label without its trailing "(Part A)" / "(Part B)" / "(B/A ratio)" qualifier.
# The qualifier is re-attached in Spanish by `to_spanish`.
# ---------------------------------------------------------------------------
ES_METRIC = {
    "Valid sum":                          "Ensayos válidos",
    "Age":                                "Edad",

    "Time in trial":                      "Tiempo de realización",
    "Search time":                        "Tiempo de búsqueda",
    "Travel time":                        "Tiempo de traslado",
    "Hesitation time":                    "Tiempo en vacilación",
    "Inter-target time":                  "Tiempo entre objetivos",
    "Intra-target time":                  "Tiempo dentro del objetivo",

    "Mean speed":                         "Velocidad media",
    "Peak speed":                         "Velocidad máxima",
    "STD speed":                          "Desvío estándar de la velocidad",
    "Search average speed":               "Velocidad media de búsqueda",
    "Travel average speed":               "Velocidad media de traslado",
    "Hesitation average speed":           "Velocidad media en vacilación",

    "Mean acceleration":                  "Aceleración media",
    "Mean absolute acceleration":         "Aceleración absoluta media",
    "Mean negative acceleration":         "Aceleración negativa media",
    "Peak acceleration":                  "Aceleración máxima",
    "Peak absolute acceleration":         "Aceleración absoluta máxima",
    "Peak negative acceleration":         "Aceleración negativa máxima",
    "STD acceleration":                   "Desvío estándar de la aceleración",
    "STD absolute acceleration":          "Desvío estándar de la aceleración absoluta",
    "STD negative acceleration":          "Desvío estándar de la aceleración negativa",

    "Total distance":                     "Distancia total",
    "Search distance":                    "Distancia de búsqueda",
    "Travel distance":                    "Distancia de traslado",
    "Hesitation distance":                "Distancia en vacilación",
    "Area difference from ideal":         "Diferencia de área con la trayectoria ideal",
    "Distance difference from ideal":     "Diferencia de distancia con la trayectoria ideal",

    "Hesitations":                        "Vacilaciones",
    "Average duration":                   "Duración media de la vacilación",
    "Hesitation max duration":            "Duración máxima de la vacilación",

    "State transitions":                  "Transiciones de estado",
    "Number of crosses":                  "Cantidad de cruces",
    "Wrong target touches":               "Toques a objetivos incorrectos",
    "Scale factor":                       "Factor de escala",
    "Sample count":                       "Cantidad de muestras",
    "Valid interval count":               "Cantidad de intervalos válidos",

    # "Complete X" = métricas calculadas sobre el ensayo completo (sin corte),
    # marcadas con "- Compl.". Las transiciones de estado van sin marca: su
    # versión recortada de la Parte B no aparece, así que no hay ambigüedad.
    "Complete state transitions":         "Transiciones de estado",
    "Complete time in trial":             "Tiempo de realización - Compl.",
    "Complete search time":               "Tiempo de búsqueda - Compl.",
    "Complete intra-target time":         "Tiempo dentro del objetivo - Compl.",
    "Complete inter-target time":         "Tiempo entre objetivos - Compl.",
    "Complete mean speed":                "Velocidad media - Compl.",
    "Complete search average speed":      "Velocidad media de búsqueda - Compl.",
    "Complete total distance":            "Distancia total - Compl.",
    "Complete search distance":           "Distancia de búsqueda - Compl.",
    "Complete number of crosses":         "Cantidad de cruces - Compl.",
    "Complete correct touches":           "Toques correctos - Compl.",
}

# The slide figure shows only 3 bars and does not need to distinguish the
# complete trial from the cut one, so the names are shortened to fit inside.
ES_METRIC_SLIDE = {
    "Complete state transitions": "Transiciones de estado",
    "Complete time in trial":     "Tiempo de realización",
}

# Known top-3 mean |SHAP| values from a full run — used with --override to
# iterate on the styling without recomputing SHAP (which takes minutes).
OVERRIDE_TOP3 = {
    "state_transitions_PART_A":         0.046,
    "non_cut_state_transitions_PART_B": 0.035,
    "rt_PART_B":                        0.033,
}

QUALIFIERS = [
    (" (Part A)", " (Parte A)"),
    (" (Part B)", " (Parte B)"),
    (" (B/A ratio)", " (cociente B/A)"),
]


def to_spanish(en_label: str, overrides=None) -> str:
    """Translate an English feature label such as 'State transitions (Part A)'.

    Splits off the trailing qualifier, translates the base metric via
    `overrides` first and ES_METRIC second (falling back to the English base),
    then re-attaches the qualifier in Spanish.
    """
    base, qualifier_es = en_label, ""
    for en_qualifier, es_qualifier in QUALIFIERS:
        if en_label.endswith(en_qualifier):
            base = en_label[: -len(en_qualifier)]
            qualifier_es = es_qualifier
            break

    overrides = overrides or {}
    base_es = overrides.get(base) or ES_METRIC.get(base, base)
    return f"{base_es}{qualifier_es}"


def _top_features(shap_df, top_n, overrides=None):
    return shap_common.top_features(
        shap_df, top_n, relabel=lambda label: to_spanish(label, overrides))


def _slide_figure(shap_df):
    """Top-3 bars with the metric name written inside each bar."""
    df_plot = _top_features(shap_df, TOP_N_SLIDE, overrides=ES_METRIC_SLIDE)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    bars = ax.barh(df_plot.index, df_plot["mean_abs_shap"],
                   color=shap_common.C_AMBER, alpha=0.85, height=0.55)

    max_val = df_plot["mean_abs_shap"].max()
    name_x = max_val * NAME_INSET
    name_texts = []
    for bar, name in zip(bars, df_plot.index):
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        # Fixed point offset so the gap stays constant even when a bar is
        # exactly as long as its label text.
        ax.annotate(f"{width:.3f}", xy=(width, y), xytext=(8, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=ANNOT_FS, color=shap_common.C_AMBER)
        name_texts.append(ax.text(name_x, y, name, va="center", ha="left",
                                  fontsize=ANNOT_FS, color="#1A1A1A"))

    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_xlabel("Media |SHAP|")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    # Settle the layout BEFORE measuring: tight_layout changes the axes width,
    # so measuring after it would invalidate the pixel math below.
    fig.tight_layout()

    # Scale so the SHORTEST bar ends exactly where its label text ends. The
    # label width is fixed (font points) while a bar's length is value/xmax of
    # the axis, so solving (v_min - name_x)/xmax * ax_px == text_px stretches
    # the axis past SLIDE_X_CUT; the crop below trims it back.
    fig.canvas.draw()
    text_px = name_texts[0].get_window_extent(fig.canvas.get_renderer()).width
    ax_px = ax.get_window_extent().width
    v_min = df_plot["mean_abs_shap"].iloc[0]
    xmax = max((v_min - name_x) * ax_px / text_px, max_val * 1.05)
    ax.set_xlim(0, xmax)

    # Keep ticks and the axis line only up to SLIDE_X_CUT and let the tight
    # bounding box trim the empty stretch on the right.
    ax.set_xticks([i / 100 for i in range(0, int(SLIDE_X_CUT * 100) + 1)])
    ax.spines["bottom"].set_bounds(0, SLIDE_X_CUT)

    save_fig(fig, "fig3_shap_k_mean_presentacion", formats=("png",), dpi=PRINT_DPI)
    plt.close(fig)


def _panel_figure(shap_df, top_n=TOP_N_PANEL):
    """Full panel (same layout as `shap_panels`) with the slide typography."""
    df_plot = _top_features(shap_df, top_n)

    # Height grows with the number of bars so the large ticks never collide.
    fig, ax = plt.subplots(figsize=(14, max(3.6, 0.62 * top_n)))
    bars = ax.barh(df_plot.index, df_plot["mean_abs_shap"],
                   color=shap_common.C_AMBER, alpha=0.85)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:.3f}", xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=ANNOT_FS,
                    color=shap_common.C_AMBER)

    ax.set_xlim(0, df_plot["mean_abs_shap"].max() * 1.25)
    ax.set_xlabel("Media |SHAP|")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # The `science` style draws minor ticks on all four edges; without the top
    # and right spines they would float.
    ax.tick_params(which="both", top=False, right=False)

    fig.tight_layout()
    save_fig(fig, "fig3_shap_k_mean_panel_presentacion", formats=("png",), dpi=PRINT_DPI)
    plt.close(fig)


def _load_shap(combo, override, wants_panel, from_csv):
    if override and wants_panel:
        print("--override sólo cubre el top 3: la figura de panel necesita SHAP real.")
    if override and not wants_panel:
        print("Using OVERRIDE_TOP3 values (skipping SHAP computation)...")
        return pd.DataFrame({"mean_abs_shap": OVERRIDE_TOP3})
    if from_csv:
        return shap_common.shap_from_csv(combo)
    print("Loading SHAP data for K_mean...")
    return shap_common.compute_shap(combo)


def main(figure="both", override=False, top_n_panel=TOP_N_PANEL, from_csv=False):
    combo = next(c for c in shap_common.COMBINATIONS if c["dataset"] == "tmt_k_mean")
    wants_panel = figure in ("panel", "both")
    shap_df = _load_shap(combo, override, wants_panel, from_csv)

    if figure in ("slide", "both"):
        _slide_figure(shap_df)
    if wants_panel:
        _panel_figure(shap_df, top_n_panel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the K_mean SHAP figures for the talk slides")
    parser.add_argument("--figure", choices=["slide", "panel", "both"], default="both",
                        help="slide = top 3 con nombres dentro de la barra; "
                             "panel = top 15 estilo shap_panels (default: both)")
    parser.add_argument("--top-n-panel", type=int, default=TOP_N_PANEL,
                        help=f"Cantidad de features en la figura de panel (default: {TOP_N_PANEL})")
    parser.add_argument("--override", action="store_true",
                        help="Usa valores fijos conocidos en vez de recomputar SHAP "
                             "(sólo con --figure slide)")
    parser.add_argument("--from-csv", action="store_true",
                        help="Reusa el shap_values_*.csv ya guardado en results/ "
                             "en vez de recomputar SHAP (minutos)")
    args = parser.parse_args()
    main(args.figure, args.override, args.top_n_panel, args.from_csv)
