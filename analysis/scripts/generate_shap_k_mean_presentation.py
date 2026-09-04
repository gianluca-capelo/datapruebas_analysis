"""Generate the K_mean SHAP importance figures for a PRESENTATION/slide.

Two standalone images for the K_mean model (RandomForestRegressor), tailored for
slides rather than the thesis. Both share the same look:

* No title.
* Large fonts for slide readability (mirrors the rcParams used in
  generate_demographics_figure_368.py).
* Feature names in Spanish (Transiciones de estado, Tiempo de realización, ...).

The two variants:

1. ``slide``  -> fig3_shap_k_mean_presentacion.png
   Only the TOP 3 features, metric names written INSIDE the bars.
2. ``panel``  -> fig3_shap_k_mean_panel_presentacion.png
   Same layout as generate_shap_figures_separate.py (TOP 15 features, names on
   the y axis), but with the presentation typography and no title.

Reuses COMBINATIONS / _compute_shap / FEATURE_LABELS from generate_shap_figures.

Usage:
    python -m analysis.scripts.generate_shap_k_mean_presentation                  # ambas
    python -m analysis.scripts.generate_shap_k_mean_presentation --figure panel
    python -m analysis.scripts.generate_shap_k_mean_presentation --figure slide --override
"""
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

from analysis.scripts.generate_shap_figures import (
    COMBINATIONS, _compute_shap, FEATURE_LABELS, FIGURES_DIR, DPI,
    C_AMBER,
)
from src.config import BASE_DIR

# Features per variant.
TOP_N_SLIDE = 3
TOP_N_PANEL = 15

# Large fonts for presentation/slides (same spirit as generate_demographics_figure_368).
plt.rcParams.update({
    "axes.labelsize":  20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 20,
})
ANNOT_FS = 20  # value labels at the end of each bar

# ---------------------------------------------------------------------------
# Spanish metric labels.
#
# Keys are the English base metric names produced by FEATURE_LABELS (i.e. the
# label without its trailing "(Part A)" / "(Part B)" / "(B/A ratio)" qualifier).
# The qualifier is re-attached in Spanish by _to_spanish.
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

# La figura de slide muestra sólo 3 barras y no necesita distinguir el ensayo
# completo del recortado: se acortan los nombres para que entren dentro de la barra.
ES_METRIC_SLIDE = {
    "Complete state transitions": "Transiciones de estado",
    "Complete time in trial":     "Tiempo de realización",
}

_QUALIFIERS = [
    (" (Part A)",     " (Parte A)"),
    (" (Part B)",     " (Parte B)"),
    (" (B/A ratio)",  " (cociente B/A)"),
]


def _to_spanish(en_label, overrides=None):
    """Translate an English feature label (e.g. 'State transitions (Part A)').

    Splits off the trailing qualifier ('(Part A)'/'(Part B)'/'(B/A ratio)'),
    translates the base metric via `overrides` first and ES_METRIC second
    (falling back to the English base), and re-attaches the qualifier in Spanish.
    """
    base, qualifier_es = en_label, ""
    for en_qualifier, es_qualifier in _QUALIFIERS:
        if en_label.endswith(en_qualifier):
            base = en_label[: -len(en_qualifier)]
            qualifier_es = es_qualifier
            break

    overrides = overrides or {}
    base_es = overrides.get(base) or ES_METRIC.get(base, base)
    return f"{base_es}{qualifier_es}"


# Known top-3 mean |SHAP| values (from a full run) — used with --override to
# iterate on the figure styling without recomputing SHAP (which takes minutes).
OVERRIDE_TOP3 = {
    "state_transitions_PART_A":         0.046,
    "non_cut_state_transitions_PART_B": 0.035,
    "rt_PART_B":                        0.033,
}


def _shap_from_csv(combo):
    """Rebuild mean |SHAP| from the raw per-fold CSV that run_shap already saved.

    Same aggregation as build_mean_shap_df (mean of |value| across the folds where
    the feature was selected; NaN = not selected). Lets us re-style the figures
    without re-fitting every LOO fold, which takes minutes.
    """
    pattern = os.path.join(BASE_DIR, "results", combo["task"], combo["timestamp"],
                           "*", combo["dataset"], f"shap_values_{combo['model']}.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No hay SHAP guardado para {combo['dataset']}/{combo['model']}: {pattern}\n"
            "Corré el script sin --from-csv al menos una vez.")

    print(f"Reusing saved SHAP values -> {matches[0]}")
    df = pd.read_csv(matches[0]).drop(columns=["fold", "base_value"])
    return pd.DataFrame({"mean_abs_shap": df.abs().mean(skipna=True)})


def _top_features(shap_df, top_n, overrides=None):
    """Top-N features sorted ascending (bottom-to-top), with Spanish labels."""
    df_plot = shap_df.sort_values("mean_abs_shap", ascending=True).tail(top_n)
    df_plot.index = df_plot.index.map(lambda x: FEATURE_LABELS.get(x, x))
    df_plot.index = df_plot.index.map(lambda x: _to_spanish(x, overrides))
    return df_plot


def _save(fig, name):
    save_path = os.path.join(FIGURES_DIR, name)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


def _slide_figure(shap_df):
    """TOP-3 bars with the metric name written inside each bar."""
    df_plot = _top_features(shap_df, TOP_N_SLIDE, overrides=ES_METRIC_SLIDE)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    bars = ax.barh(df_plot.index, df_plot["mean_abs_shap"],
                   color=C_AMBER, alpha=0.85, height=0.55)

    max_val = df_plot["mean_abs_shap"].max()
    name_x = max_val * 0.02  # left inset of the metric name inside the bar
    name_texts = []
    for bar, name in zip(bars, df_plot.index):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        # Value just past the end of the bar (fixed point offset so the gap is
        # constant even when a bar is exactly as long as its label text).
        ax.annotate(f"{w:.3f}", xy=(w, y), xytext=(8, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=ANNOT_FS, color=C_AMBER)
        # Metric name inside the bar (left-aligned, dark for contrast on amber).
        t = ax.text(name_x, y, name, va="center", ha="left",
                    fontsize=ANNOT_FS, color="#1A1A1A")
        name_texts.append(t)

    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_xlabel("Media |SHAP|")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Settle the layout BEFORE measuring: tight_layout changes the axes width, so
    # measuring after it would invalidate the pixel math below.
    fig.tight_layout()

    # ----- Scale so the SHORTEST bar ends exactly where its label text ends ----
    # The label width is fixed (font points); a bar's length is value/xmax of the
    # axis. The name is inset by `name_x`, so its right edge sits at name_x +
    # text_width; we want the shortest bar (df_plot ascending -> bars[0]) to reach
    # that edge. Solve: (v_min - name_x)/xmax * ax_px == text_px. This "stretches"
    # the axis (xmax > X_CUT), which we then crop back below.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_px = name_texts[0].get_window_extent(renderer).width
    ax_px = ax.get_window_extent().width
    v_min = df_plot["mean_abs_shap"].iloc[0]
    xmax = (v_min - name_x) * ax_px / text_px
    xmax = max(xmax, max_val * 1.05)  # never clip the longest bar
    ax.set_xlim(0, xmax)

    # ----- Crop the figure back to X_CUT: keep ticks + axis line only up to
    # X_CUT and let bbox_inches="tight" trim the empty stretch on the right -----
    X_CUT = 0.05
    ax.set_xticks([i / 100 for i in range(0, int(X_CUT * 100) + 1)])
    ax.spines["bottom"].set_bounds(0, X_CUT)
    _save(fig, "fig3_shap_k_mean_presentacion.png")


def _panel_figure(shap_df, top_n=TOP_N_PANEL):
    """Full panel (like generate_shap_figures_separate) with slide typography."""
    df_plot = _top_features(shap_df, top_n)

    # Height grows with the number of bars so the large ticks never collide.
    fig, ax = plt.subplots(figsize=(14, max(3.6, 0.62 * top_n)))
    bars = ax.barh(df_plot.index, df_plot["mean_abs_shap"],
                   color=C_AMBER, alpha=0.85)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.3f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=ANNOT_FS, color=C_AMBER)

    ax.set_xlim(0, df_plot["mean_abs_shap"].max() * 1.25)
    ax.set_xlabel("Media |SHAP|")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # El estilo `science` dibuja minor ticks en los cuatro bordes: sin spines
    # arriba/derecha quedan flotando, así que se apagan.
    ax.tick_params(which="both", top=False, right=False)

    fig.tight_layout()
    _save(fig, "fig3_shap_k_mean_panel_presentacion.png")


def main(figure="both", override=False, top_n_panel=TOP_N_PANEL, from_csv=False):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    combo = next(c for c in COMBINATIONS if c["dataset"] == "tmt_k_mean")
    wants_panel = figure in ("panel", "both")

    if override and wants_panel:
        print("--override sólo cubre el top 3: la figura de panel necesita SHAP real.")
    if override and not wants_panel:
        print("Using OVERRIDE_TOP3 values (skipping SHAP computation)...")
        shap_df = pd.DataFrame({"mean_abs_shap": OVERRIDE_TOP3})
    elif from_csv:
        shap_df = _shap_from_csv(combo)
    else:
        print("Loading SHAP data for K_mean...")
        shap_df = _compute_shap(combo["dataset"], combo["model"],
                                combo["timestamp"], combo["task"])

    if figure in ("slide", "both"):
        _slide_figure(shap_df)
    if wants_panel:
        _panel_figure(shap_df, top_n_panel)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate the K_mean SHAP presentation figures")
    parser.add_argument("--figure", choices=["slide", "panel", "both"], default="both",
                        help="slide = top 3 con nombres dentro de la barra; "
                             "panel = top 15 estilo generate_shap_figures_separate "
                             "(default: both)")
    parser.add_argument("--top-n-panel", type=int, default=TOP_N_PANEL,
                        help=f"Cantidad de features en la figura de panel (default: {TOP_N_PANEL})")
    parser.add_argument("--override", action="store_true",
                        help="Usa valores fijos conocidos (0.046, 0.035, 0.033) "
                             "en vez de recomputar SHAP (sólo --figure slide)")
    parser.add_argument("--from-csv", action="store_true",
                        help="Reusa el shap_values_*.csv ya guardado en results/ "
                             "en vez de recomputar SHAP (minutos)")
    args = parser.parse_args()
    main(args.figure, args.override, args.top_n_panel, args.from_csv)
