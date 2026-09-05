"""Prediction loading, formatting and palette shared by the permutation figures.

The null distributions are not stored anywhere — only the p-value survives into
the consolidated CSVs — so every permutation figure recomputes them from the
stored predictions with the same procedure behind the published p-values:
y_true shuffled against fixed y_pred, seed 42, 1000 permutations.
"""
import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import REGRESSION_RESULTS_DIR
from src.model.permutation_tests import permutation_test

from analysis.scripts.figures._style import PANEL_RCPARAMS

NULL_GRAY = "#B0B0B0"
NULL_EDGE = "#5A5A5A"
OBSERVED_GREEN = "#1B5E20"
ROW_RULE_GRAY = "#E5E5E5"

# Reserved for a model that failed to beat its null: the color carries that on
# its own, without an extra annotation.
NOT_SIGNIFICANT_RED = "#B71C1C"

# The p-values in the consolidated results were computed with these; changing
# them makes a prettier figure but a different test.
DEFAULT_N_PERMUTATIONS = 1000
DEFAULT_SEED = 42

# The one-tailed test only cares about the left edge of the null, so the bands
# span this percentile to the maximum.
NULL_BAND_PERCENTILE = 5

# Geometry shared by `permutation_hist --aligned` and
# `permutation_dotplot --aligned`: the same canvas and axes rectangle put the x
# axis on the same pixel in both figures, so the histogram can collapse into the
# dot plot row across two slides. Saving must skip the tight bounding box for
# this to hold — see `save_fig(tight=False)`.
ALIGNED_FIGSIZE = (13, 6)
ALIGNED_AXES_RECT = (0.13, 0.22, 0.74, 0.62)

X_LABELS = {
    "es": {"years": "MAE (años)", "ms": "MAE (ms)", None: "MAE"},
    "en": {"years": "MAE (years)", "ms": "MAE (ms)", None: "MAE"},
}

LEGEND_LABELS = {
    "es": ("MAE bajo etiquetas permutadas", "MAE observado en mejor modelo"),
    "en": ("MAE under permuted labels", "Observed MAE"),
}

P_LABEL = {"es": "p valor", "en": "p-value"}

# Display names: the sklearn class names are too long for a slide.
MODEL_LABELS = {
    "SVR": "SVR",
    "Ridge": "Ridge",
    "ElasticNet": "ElasticNet",
    "Lasso": "Lasso",
    "LinearRegression": {"es": "Regresión lineal", "en": "Linear regression"},
    "RandomForestRegressor": {"es": "Random forest", "en": "Random forest"},
    "XGBRegressor": "XGBoost",
    "DummyRegressor": {"es": "Modelo trivial", "en": "Dummy"},
}


def use_permutation_style(label_fs: int, tick_fs: int):
    """Science style at the sizes these figures are projected at."""
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({
        "axes.labelsize": label_fs,
        "xtick.labelsize": tick_fs,
        "ytick.labelsize": tick_fs,
        **PANEL_RCPARAMS,
    })


def parse_array(array_str) -> np.ndarray:
    """Parse the stringified lists stored in summary.csv."""
    if isinstance(array_str, str):
        return np.array(ast.literal_eval(array_str))
    return np.asarray(array_str)


def read_summary(timestamp: str, target: str, dataset: str) -> pd.DataFrame:
    summary_path = os.path.join(REGRESSION_RESULTS_DIR, timestamp, target, dataset,
                                "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No summary.csv at {summary_path}")
    return pd.read_csv(summary_path)


def load_predictions(timestamp: str, target: str, dataset: str, model: str):
    """(y_true, y_pred) for one model of a regression run."""
    df = read_summary(timestamp, target, dataset)
    row = df[df["model"] == model]
    if row.empty:
        raise ValueError(
            f"Model '{model}' not found for {target}/{dataset} at {timestamp}. "
            f"Available: {', '.join(df['model'])}"
        )
    row = row.iloc[0]
    return parse_array(row["y_true"]), parse_array(row["y_pred"])


def run_permutation(y_true, y_pred, n_permutations: int, seed: int):
    """(observed MAE, p-value, null distribution) for a fixed set of predictions."""
    return permutation_test(y_true, y_pred, n_permutations=n_permutations, seed=seed,
                            metric="mae", return_null_distribution=True)


def null_summary(null: np.ndarray) -> dict:
    """Band edges and median of a null distribution, as the figures need them."""
    return {
        "null_low": float(np.percentile(null, NULL_BAND_PERCENTILE)),
        "null_median": float(np.median(null)),
        "null_min": float(null.min()),
        "null_max": float(null.max()),
    }


def localize(text: str, lang: str) -> str:
    """Spanish uses a comma as the decimal separator."""
    return text.replace(".", ",") if lang == "es" else text


def format_value(value: float, lang: str, decimals: int = 2) -> str:
    return localize(f"{value:.{decimals}f}", lang)


def format_p_value(p_value: float, lang: str) -> str:
    """`p valor < 0.001` below the resolution of the test, `p valor = x.xxx` otherwise.

    With 1000 permutations the smallest attainable p is 1/1001 ~ 0.001, so the
    threshold is the floor of the test rather than an arbitrary cutoff.
    """
    label = P_LABEL[lang]
    if p_value < 0.001:
        return localize(f"{label} < 0.001", lang)
    return localize(f"{label} = {p_value:.3f}", lang)


def format_p_bare(p_value: float, lang: str) -> str:
    """`<0.001` at the resolution floor of the test, `0.00x` otherwise."""
    if p_value < 0.001:
        return localize("<0.001", lang)
    return localize(f"{p_value:.3f}", lang)


def model_label(model: str, lang: str) -> str:
    label = MODEL_LABELS.get(model, model)
    return label[lang] if isinstance(label, dict) else label


def null_xlim(observed: float, null_min: float, null_max: float,
              margin_frac: float = 0.08) -> tuple[float, float]:
    """x limits keeping both the observed value and the whole null in frame.

    The observed MAE normally sits far to the left of the null. Shared with the
    dot plot so a single-model row can reuse the histogram's axis.
    """
    low = min(observed, null_min)
    high = max(observed, null_max)
    margin = margin_frac * (high - low)
    return low - margin, high + margin


def hide_frame(ax, sides=("top", "right", "left")):
    ax.grid(False)
    for side in sides:
        ax.spines[side].set_visible(False)


def add_permutation_arguments(parser):
    parser.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS,
                        help=f"Permutaciones (default: {DEFAULT_N_PERMUTATIONS}, "
                             "el valor con el que se reportó el p-valor)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Semilla del test (default: {DEFAULT_SEED})")
