"""Trial selection, loading and background drawing for the single-trial figures.

`segmentation`, `speed`, `time`, `time_animation` and `area` all render the same
cTMT Part B trial, so the selection criteria live here instead of in whichever
script happened to be written first.
"""
import glob
import os

import pandas as pd

from src import config
from src.config import BASE_DIR
from src.mapper.datapruebas.datapruebas_mapper import DatapruebasTMTMapper
from src.visualization.trial_plotting_helpers import configure_trial_axes, draw_trial_targets

from analysis.scripts.figures._style import FontSizes

HAND_ANALYSIS_DIR = os.path.join(BASE_DIR, "data", "hand_analysis")
RAW_EXPERIMENT_PATH = os.path.join(
    BASE_DIR, "data", "raw", "tmt", "datapruebas", "subjects", config.EXPERIMENT_FILE_NAME
)

MIN_CORRECT_TOUCHES = 20
MIN_HESITATIONS = 5
# Rank within the candidates sorted by hesitation count. Ranking rather than
# taking the top one avoids the extreme outlier while staying deterministic.
CANDIDATE_RANK = 5

C_FAINT_TRAIL = "#cccccc"

AXIS_LABELS = {
    "es": {"x": "Coordenada X (px)", "y": "Coordenada Y (px)"},
    "en": {"x": "X Screen Coordinate (px)", "y": "Y Screen Coordinate (px)"},
}


def latest_tmt_analysis_path() -> str:
    """analysis.csv of the most recent hand_analysis timestamp."""
    candidates = sorted(glob.glob(os.path.join(HAND_ANALYSIS_DIR, "*", "analysis.csv")))
    if not candidates:
        raise FileNotFoundError(f"No analysis.csv found under {HAND_ANALYSIS_DIR}")
    return candidates[-1]


def select_trial(df_tmt, subject_id=None, trial_id=None):
    """Row of a representative valid PART_B trial, or of the requested one."""
    if subject_id is not None and trial_id is not None:
        match = df_tmt[(df_tmt["subject_id"] == subject_id) & (df_tmt["trial_id"] == trial_id)]
        if match.empty:
            raise ValueError(f"No trial {trial_id} for subject {subject_id}")
        return match.iloc[0]

    candidates = df_tmt[
        df_tmt["is_valid"].eq(True)
        & df_tmt["trial_type"].str.contains("PART_B")
        & df_tmt["trial_id"].str.startswith("DATAPRUEBAS")
        & (df_tmt["non_cut_correct_targets_touches"] >= MIN_CORRECT_TOUCHES)
        & (df_tmt["total_hesitations"] >= MIN_HESITATIONS)
    ]
    if len(candidates) <= CANDIDATE_RANK:
        raise ValueError(
            f"Only {len(candidates)} candidate trials; need more than {CANDIDATE_RANK}"
        )
    return candidates.sort_values("total_hesitations", ascending=False).iloc[CANDIDATE_RANK]


def load_trial(subject_id=None, trial_id=None):
    """Return (analysis row, subject, trial) for the figure trial."""
    df_tmt = pd.read_csv(latest_tmt_analysis_path(), on_bad_lines="warn")
    row = select_trial(df_tmt, subject_id, trial_id)

    experiment = DatapruebasTMTMapper().map(RAW_EXPERIMENT_PATH)
    subject = experiment.subjects[row["subject_id"]]
    trial = next(t for t in subject.testing_trials if t.id == row["trial_id"])

    print(f"Selected: subject={row['subject_id']}  trial={row['trial_id']}  "
          f"hesitations={row['total_hesitations']:.0f}")
    return row, subject, trial


def cursor_coordinates(trial):
    cursor_trail = trial.get_cursor_trail_from_start()
    x = [p.position.x for p in cursor_trail]
    y = [p.position.y for p in cursor_trail]
    return cursor_trail, x, y


def draw_faint_trail(ax, x, y):
    """Gray connector under the colored points, so reading order stays visible."""
    ax.plot(x, y, color=C_FAINT_TRAIL, lw=1.0, alpha=0.6, zorder=2,
            solid_capstyle="round")


def draw_trial_background(ax, trial, subject, x, y, lang: str, fonts: FontSizes,
                          circle_linewidth: float = 1.3, zorder_circle: int = 5,
                          zorder_text: int = 6):
    """Targets, axis limits and axis labels — shared by the trial figures.

    `area` draws shaded polygons under the targets, so it needs to lift them
    above its own artists and thin their stroke.
    """
    draw_trial_targets(ax, trial, subject.target_radius, circle_color="black",
                       circle_alpha=0.9, circle_linewidth=circle_linewidth,
                       text_fontsize=fonts.target, text_color="black",
                       zorder_circle=zorder_circle, zorder_text=zorder_text)
    configure_trial_axes(ax, x=x, y=y, show_labels=True,
                         xlabel=AXIS_LABELS[lang]["x"], ylabel=AXIS_LABELS[lang]["y"])
    apply_axis_fonts(ax, fonts)


def apply_axis_fonts(ax, fonts: FontSizes):
    ax.xaxis.label.set_fontsize(fonts.label)
    ax.yaxis.label.set_fontsize(fonts.label)
    ax.tick_params(axis="both", labelsize=fonts.tick)


def horizontal_colorbar_label(fig, mappable, ax, label: str, fonts: FontSizes,
                              extend: str = "neither"):
    """Colorbar with its label laid out horizontally on top instead of rotated.

    `extend="max"` adds the pointed cap that says the scale is clipped, which is
    the only cue that values above the color range exist.
    """
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, extend=extend)
    cbar.ax.set_title(label, fontsize=fonts.legend, pad=10)
    cbar.ax.tick_params(labelsize=fonts.tick)
    return cbar


def trial_slug(row) -> str:
    """Filename fragment tying a figure back to the trial it was drawn from."""
    return f"{row['subject_id'][:8]}_{row['trial_id']}"


def legend_kwargs(fonts: FontSizes, outside: bool, **overrides):
    """Legend styling; `outside` parks it to the right so it cannot cover data."""
    kwargs = dict(fontsize=fonts.legend, framealpha=0.9, edgecolor="#cccccc")
    if outside:
        kwargs.update(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    kwargs.update(overrides)
    return kwargs
