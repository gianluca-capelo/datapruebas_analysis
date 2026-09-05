"""Generate the sample-demographics figures for the MODELING sample.

Variant of `demographics` restricted to the subjects that actually feed the ML
models: those with a valid TMT (at least one valid PART_A and one valid PART_B
trial), as defined by `DatasetBuilder._load_tmt_aggregated()`. The full-sample
script covers every subject seen across the four tasks.

Outputs four separate PNGs — age, gender, education level, nationality — with
the slide typography.

Usage:
    python -m analysis.scripts.figures.demographics_modeling            # castellano
    python -m analysis.scripts.figures.demographics_modeling --lang en  # inglés
    python -m analysis.scripts.figures.demographics_modeling --stats    # + media/DE en edad
"""
import argparse

import matplotlib.pyplot as plt

from analysis.scripts.figures import demographics_common as dc
from analysis.scripts.figures._style import (
    PRINT_DPI,
    add_lang_argument,
    lang_suffix,
    save_fig,
)

dc.use_demographics_style(slide=True)

# One color per panel, kept in sync with `demographics` so both sets of slides
# read as one system; blue distinguishes this sample's age panel from the teal
# of the full-sample one.
PANEL_COLORS = {
    "age": "#2E5C8A",
    "gender": "#E1812C",
    "education": "#3A923A",
    "nationality": "#8A5CA8",
}
C_STAT = "#1A1A1A"

N_FS = 22


def _restrict_to_modeling_sample(metadata):
    modeling_ids = dc.modeling_subject_ids()
    restricted = metadata[metadata["subject_id"].isin(modeling_ids)].copy()
    print(f"Restricted to modeling sample: {len(restricted)} / {len(metadata)} subjects "
          f"(expected {len(modeling_ids)})")
    if len(restricted) != len(modeling_ids):
        raise ValueError(
            f"Expected {len(modeling_ids)} modeling subjects, matched {len(restricted)}"
        )
    return restricted


def main(lang="es", stats=False):
    labels = dc.display_labels(lang)

    print("Loading data...")
    metadata = _restrict_to_modeling_sample(dc.build_metadata(*dc.load_task_analyses()))
    suffix = lang_suffix(lang)

    panels = [
        ("age", metadata["age"].dropna(), None),
        ("gender", dc.ordered_counts(metadata, "gender_desc", dc.GENDER_ORDER),
         labels["gender"]),
        ("education", dc.ordered_counts(metadata, "education_level", dc.EDU_ORDER),
         labels["education"]),
        ("nationality", dc.ordered_counts(metadata, "nationality_clean"),
         labels["nationality"]),
    ]

    for key, data, display_map in panels:
        fig, ax = plt.subplots(figsize=dc.PANEL_FIGSIZE)
        name = f"fig1_demographics_modeling_{key}"

        if key == "age":
            dc.draw_age_panel(ax, data, labels, PANEL_COLORS[key])
            # N reports the full modeling sample; the histogram itself omits the
            # few subjects whose age is missing or corrupt after cleaning.
            dc.annotate_n(ax, len(metadata), N_FS)
            if stats:
                dc.annotate_age_stats(ax, data, labels, C_STAT, N_FS)
                name += "_stats"
        else:
            dc.draw_category_panel(ax, data, display_map, labels, PANEL_COLORS[key])

        fig.tight_layout()
        save_fig(fig, f"{name}{suffix}", formats=("png",), dpi=PRINT_DPI)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the demographics figures for the modeling sample")
    add_lang_argument(parser)
    parser.add_argument("--stats", action="store_true",
                        help="Superpone media y ±1 DE en el panel de edad")
    args = parser.parse_args()
    main(args.lang, args.stats)
