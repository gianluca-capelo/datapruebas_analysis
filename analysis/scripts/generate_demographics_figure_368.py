"""Generate the sample-demographics figures (one file per panel) for the MODELING sample.

Variant of `generate_demographics_figure.py` restricted to the 368 subjects
that actually feed the ML models: those with a *valid* TMT (at least one valid
PART_A and one valid PART_B trial), as defined by
`DatasetBuilder._load_tmt_aggregated()`.

The full-sample script builds metadata for every subject seen across the four
tasks (484). Here we reuse that exact metadata-building machinery and then keep
only the 368 modeling subjects, so the demographics match the sample the models
are trained on.

Outputs four separate PNGs (age, gender, education, nationality):
    A. Age distribution   B. Gender
    C. Education level     D. Nationality

Usage:
    python -m analysis.scripts.generate_demographics_figure_368              # inglés
    python -m analysis.scripts.generate_demographics_figure_368 --lang es    # castellano
    python -m analysis.scripts.generate_demographics_figure_368 --stats      # + media/DE en edad
"""
import matplotlib.pyplot as plt

from src.model.datasetbuilder.dataset_builder import DatasetBuilder

# Reuse everything from the full-sample script (style + rcParams are applied on
# import; helpers/constants are imported by reference to avoid duplication).
from analysis.scripts.generate_demographics_figure import (
    _labels,
    _load_data,
    _build_metadata,
    save_fig,
    GENDER_ORDER,
    EDU_ORDER,
)

# Larger fonts for presentation/slides (overrides rcParams applied on import of
# the full-sample script).
plt.rcParams.update({
    "axes.labelsize":  20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})
N_FS = 22  # font size of the in-plot N annotation

# One color per demographic panel.
C_AGE = "#2E5C8A"          # azul
C_GENDER = "#E1812C"       # naranja
C_EDUCATION = "#3A923A"    # verde
C_NATIONALITY = "#8A5CA8"  # violeta
C_STAT = "#1A1A1A"         # gris casi negro (media/DE, alto contraste sobre las barras)


def _annotate_age_stats(ax, ages, lang):
    """Overlay mean + ±1 SD on the age histogram for slide readability.

    Three elements: a thick vertical line at the mean (anchored to the age
    axis), a horizontal ±1 SD whisker floating near the top (shows the spread
    as a visible width without muddying the bars), and a large text label with
    the exact values.
    """
    m, s = ages.mean(), ages.std()
    mean_lbl = "Media" if lang == "es" else "Mean"
    sd_lbl = "DE" if lang == "es" else "SD"

    # Vertical mean line (full height, on the age axis) + a tag at its top so
    # it's clear the line is the mean.
    ax.axvline(m, color=C_STAT, lw=2.0, zorder=5)
    ymax = ax.get_ylim()[1]
    ax.text(m, ymax, f" {mean_lbl}", color=C_STAT, ha="left", va="top",
            fontsize=N_FS - 4)

    # Numeric summary (right side, below the N annotation).
    ax.text(0.97, 0.78, f"{mean_lbl} = {m:.0f}\n{sd_lbl} = {s:.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=N_FS)


def _modeling_subject_ids():
    """Return the set of subject_ids with a valid TMT (the 368 modeling sample)."""
    builder = DatasetBuilder()
    return set(builder._load_tmt_aggregated()["subject_id"])


def main(lang="en", stats=False):
    _titles, age_lbl, count_lbl, gender_map, edu_map, nat_map = _labels(lang)

    print("Loading data...")
    metadata = _build_metadata(*_load_data())

    # ---- Restrict to the modeling sample (368 subjects with valid TMT) ----
    modeling_ids = _modeling_subject_ids()
    before = len(metadata)
    metadata = metadata[metadata["subject_id"].isin(modeling_ids)].copy()
    print(f"Restricted to modeling sample: {len(metadata)} / {before} subjects "
          f"(expected {len(modeling_ids)})")
    assert len(metadata) == len(modeling_ids), (
        f"Expected {len(modeling_ids)} subjects, got {len(metadata)}"
    )

    # -----------------------------------------------------------------------
    # Four separate figures (one file per panel)
    # -----------------------------------------------------------------------
    suffix = "_es" if lang == "es" else ""

    # --- A. Age distribution ---
    fig_a, ax_a = plt.subplots(figsize=(4.5, 3.2))
    ages = metadata["age"].dropna()
    ax_a.hist(ages, bins=20, color=C_AGE, edgecolor="white", linewidth=0.4)
    ax_a.set_xlabel(age_lbl)
    ax_a.set_ylabel(count_lbl)
    # N reports the full modeling sample (368); the histogram itself omits the
    # few subjects whose age is missing/corrupt after cleaning.
    ax_a.text(0.97, 0.95, f"$N$ = {len(metadata)}",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=N_FS)
    if stats:
        _annotate_age_stats(ax_a, ages, lang)
    fig_a.tight_layout()
    stat_suffix = "_stats" if stats else ""
    save_fig(fig_a, f"fig1_demographics_368_age{stat_suffix}{suffix}.png")

    # --- B. Gender ---
    fig_b, ax_b = plt.subplots(figsize=(4.5, 3.2))
    present_genders = [g for g in GENDER_ORDER if g in metadata["gender_desc"].values]
    gender_counts = metadata["gender_desc"].value_counts().reindex(present_genders)
    gender_labels = [gender_map.get(x, x) for x in gender_counts.index[::-1]]
    ax_b.barh(gender_labels, gender_counts.values[::-1], color=C_GENDER)
    ax_b.set_xlabel(count_lbl)
    fig_b.tight_layout()
    save_fig(fig_b, f"fig1_demographics_368_gender{suffix}.png")

    # --- C. Education level (ordered P->R) ---
    fig_c, ax_c = plt.subplots(figsize=(4.5, 3.2))
    present_edu = [e for e in EDU_ORDER if e in metadata["education_level"].values]
    edu_counts = metadata["education_level"].value_counts().reindex(present_edu)
    edu_labels = [edu_map[e] for e in edu_counts.index]
    ax_c.barh(edu_labels[::-1], edu_counts.values[::-1], color=C_EDUCATION)
    ax_c.set_xlabel(count_lbl)
    fig_c.tight_layout()
    save_fig(fig_c, f"fig1_demographics_368_education{suffix}.png")

    # --- D. Nationality ---
    fig_d, ax_d = plt.subplots(figsize=(4.5, 3.2))
    nat_counts = metadata["nationality_clean"].value_counts()
    nat_labels = [nat_map.get(x, x) for x in nat_counts.index[::-1]]
    ax_d.barh(nat_labels, nat_counts.values[::-1], color=C_NATIONALITY)
    ax_d.set_xlabel(count_lbl)
    fig_d.tight_layout()
    save_fig(fig_d, f"fig1_demographics_368_nationality{suffix}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the demographics figure for the 368-subject modeling sample"
    )
    parser.add_argument("--lang", choices=["en", "es"], default="en",
                        help="Idioma de títulos/ejes/categorías (default: en)")
    parser.add_argument("--stats", action="store_true",
                        help="Superpone media y ±1 DE en el panel de edad")
    args = parser.parse_args()
    main(args.lang, args.stats)
