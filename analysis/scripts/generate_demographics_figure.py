"""Generate the sample-demographics figure (2x2 panel).

Standalone replica of paper_figures.ipynb "Figure 1 — 2x2 demographics panel"
(plus the metadata-building cell it depends on), so the thesis demographics
figure is reproducible from the repo.

Panels:
    A. Age distribution   B. Gender
    C. Education level     D. Nationality

Usage:
    python -m analysis.scripts.generate_demographics_figure                  # 2x2 panel, inglés
    python -m analysis.scripts.generate_demographics_figure --lang es        # 2x2 panel, castellano
    python -m analysis.scripts.generate_demographics_figure --presentation   # 4 PNGs separados, fuentes grandes, sin títulos
    python -m analysis.scripts.generate_demographics_figure --presentation --stats  # + media/DE en edad
"""
import os
import re
import glob
import difflib

import numpy as np  # noqa: F401 — kept for parity with notebook setup
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 — registers styles

from src.config import BASE_DIR
from src.loader import (
    get_latest_sst_analysis,
    get_latest_cdt_analysis,
    get_latest_gonogo_analysis,
)

HAND_ANALYSIS_DIR = os.path.join(BASE_DIR, "data", "hand_analysis")
META_BASE = os.path.join(BASE_DIR, "data", "raw", "tmt")
FIGURES_DIR = os.path.join(BASE_DIR, "analysis", "figures")

# ---------------------------------------------------------------------------
# Style — mirrors paper_figures.ipynb Setup cell
# ---------------------------------------------------------------------------
plt.style.use(["science", "no-latex"])

TITLE_FS = 13.5
LABEL_FS = 16
TICK_FS  = 14

plt.rcParams.update({
    "axes.titlesize":  TITLE_FS,
    "axes.labelsize":  LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "xtick.top":           False,
    "ytick.right":         False,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
})

C_DEMO = "#777777"  # demographic data
DPI = 300

# ---------------------------------------------------------------------------
# Presentation mode — large fonts + one figure per panel, no titles
# (mirrors generate_demographics_figure_368.py, but for the FULL sample)
# ---------------------------------------------------------------------------
# Applied inside the presentation routine so the paper 2x2 panel keeps the
# smaller rcParams set on import.
PRESENTATION_RCPARAMS = {
    "axes.labelsize":  20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
}
PRES_FIGSIZE = (4.5, 3.2)
N_FS = 22  # font size of the in-plot N annotation

# One color per demographic panel (kept in sync with the _368 variant so both
# sets of slides look like one system).
C_AGE = "#2A9D8F"          # teal (distinto del azul de la variante _368)
C_GENDER = "#E1812C"       # naranja
C_EDUCATION = "#3A923A"    # verde
C_NATIONALITY = "#8A5CA8"  # violeta
C_STAT = "#1A1A1A"         # gris casi negro (media/DE, alto contraste sobre las barras)

# ---------------------------------------------------------------------------
# Metadata processing constants (paper_figures.ipynb cell 5)
# ---------------------------------------------------------------------------
REFERENCE_YEAR = 2022
METADATA_COLS = ["age", "gender", "education_level", "nationality"]
_TASK_RE = re.compile(
    r"^(.+@.+)-(tmt-plugin|change-detection-task|new-go-nogo|stop-signal-task)_\d+.*$"
)

EDU_MAP = {
    "P": "P", "Primaria completa": "P",
    "S": "S", "Secundaria completa": "S",
    "T": "T", "Terciaria completa": "T",
    "U": "U", "Universitaria completa": "U",
    "R": "R", "Posgrado completa": "R", "Posgrado completo": "R",
}
CLEAN_COUNTRIES = [
    "Argentina", "Uruguay", "Colombia", "Venezuela",
    "Mexico", "Peru", "Chile", "Ecuador", "Spain", "Bolivia",
]
GENDER_MAP = {"M": "Male", "F": "Female", "N": "Non-binary",
              "G": "Gender fluid", "O": "Other", "D": "Other"}

# Display ordering for the figure (cell 6)
EDU_ORDER = ["P", "S", "T", "U", "R"]
EDU_LABELS = {
    "P": "Primary", "S": "Secondary", "T": "Tertiary",
    "U": "University", "R": "Postgraduate",
}
GENDER_ORDER = ["Female", "Male", "Non-binary", "Gender fluid", "Other"]


def _labels(lang):
    """Return display labels for the requested language.

    Only display strings are translated; the underlying data (canonical English
    categories used for matching/ordering) is untouched.
    """
    es = lang == "es"
    titles = {
        "A": "A. Distribución de la edad" if es else "A. Age distribution",
        "B": "B. Género"                  if es else "B. Gender",
        "C": "C. Nivel educativo"         if es else "C. Education level",
        "D": "D. Nacionalidad"            if es else "D. Nationality",
    }
    age_lbl   = "Edad (años)" if es else "Age (years)"
    count_lbl = "Cantidad"    if es else "Count"
    # canonical -> display maps (identity in English)
    gender = {
        "Female": "Femenino", "Male": "Masculino", "Non-binary": "No binario",
        "Gender fluid": "Género fluido", "Other": "Otro",
    } if es else {k: k for k in GENDER_ORDER}
    edu = {
        "P": "Primario", "S": "Secundario", "T": "Terciario",
        "U": "Universitario", "R": "Posgrado",
    } if es else dict(EDU_LABELS)
    nat = {"Mexico": "México", "Peru": "Perú", "Spain": "España", "Other": "Otros"} if es else {}
    return titles, age_lbl, count_lbl, gender, edu, nat


def _latest_tmt_analysis_path():
    """Return the analysis.csv of the most recent hand_analysis timestamp.

    The notebook hardcoded a specific timestamp; here we resolve the latest
    available one (timestamps sort lexically).
    """
    candidates = sorted(glob.glob(os.path.join(HAND_ANALYSIS_DIR, "*", "analysis.csv")))
    if not candidates:
        raise FileNotFoundError(f"No analysis.csv found under {HAND_ANALYSIS_DIR}")
    return candidates[-1]


def save_fig(fig, filename):
    """Save figure to FIGURES_DIR with consistent settings."""
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"Saved -> {path}")


def _load_data():
    """Load the task DataFrames used to enumerate/deduplicate subjects."""
    tmt_path = _latest_tmt_analysis_path()
    print(f"TMT analysis: {tmt_path}")
    df_tmt = pd.read_csv(tmt_path, on_bad_lines="warn")
    df_sst, _ = get_latest_sst_analysis()
    df_cdt, _ = get_latest_cdt_analysis()
    df_gonogo, _ = get_latest_gonogo_analysis()
    return df_tmt, df_sst, df_cdt, df_gonogo


def _build_metadata(df_tmt, df_sst, df_cdt, df_gonogo):
    """Build the cleaned `metadata` DataFrame (paper_figures.ipynb cell 5)."""
    # ---- Raw metadata sources ----
    raw_dp = pd.read_csv(
        os.path.join(META_BASE, "datapruebas", "metadata", "metadata.csv"), sep=";"
    ).drop_duplicates("id")
    raw_np = pd.read_csv(
        os.path.join(META_BASE, "neuropruebas", "metadata", "metadata.csv"), sep=";"
    ).drop_duplicates("id")
    raw_legacy = pd.read_csv(
        os.path.join(
            META_BASE, "neuropruebas", "metadata",
            "Sujetxs TMT Nacho - participantes_con_genero_inferido.csv",
        )
    )

    # ---- Normalize ----
    raw_dp["age"] = REFERENCE_YEAR - pd.to_datetime(raw_dp["birthdate"], errors="coerce").dt.year
    dp_norm = raw_dp[["id", "email", "age", "gender", "level_of_education", "nationality"]].rename(
        columns={"level_of_education": "education_level"}
    )

    raw_np["age"] = REFERENCE_YEAR - raw_np["año_de_nacimiento"]
    np_norm = raw_np[["id", "email", "age", "genero", "nivel_educativo", "nacionalidad"]].rename(
        columns={"genero": "gender", "nivel_educativo": "education_level", "nacionalidad": "nationality"}
    )

    # ---- Lookup dicts ----
    def _email_lookup(df):
        lkp = {}
        for _, r in df.iterrows():
            e = str(r.get("email", "")).strip().lower()
            if e and e != "nan":
                lkp[e] = {c: r[c] for c in METADATA_COLS}
        return lkp

    lookup_dp_id    = dp_norm.set_index("id")[METADATA_COLS].to_dict("index")
    lookup_np_id    = np_norm.set_index("id")[METADATA_COLS].to_dict("index")
    lookup_dp_email = _email_lookup(dp_norm)
    lookup_np_email = _email_lookup(np_norm)
    lookup_legacy   = {
        str(r.get("Mail", "")).strip().lower(): {
            "age": None, "gender": r.get("genero"),
            "education_level": r.get("nivel_educativo"), "nationality": r.get("pais"),
        }
        for _, r in raw_legacy.iterrows()
    }

    # ---- Deduplicate subjects across tasks ----
    all_sids_raw = (
        set(df_tmt["subject_id"]) | set(df_sst["subject_id"])
        | set(df_cdt["subject_id"]) | set(df_gonogo["subject_id"])
    )
    email_to_sids = {}
    non_email_sids = set()
    for sid in all_sids_raw:
        m = _TASK_RE.match(str(sid))
        if m:
            email_to_sids.setdefault(m.group(1).strip().lower(), []).append(sid)
        else:
            non_email_sids.add(sid)

    email_canonical = {
        e: next((s for s in sids if "tmt-plugin" in s), sorted(sids)[0])
        for e, sids in email_to_sids.items()
    }
    all_subjects = non_email_sids | set(email_canonical.values())

    # ---- TMT metadata (most reliable — uses exact birth dates) ----
    tmt_meta_lkp = (
        df_tmt[["subject_id"] + METADATA_COLS]
        .groupby("subject_id").first()
        .query("age > 0")
        [METADATA_COLS].to_dict("index")
    )

    # ---- Match metadata for every subject (priority order) ----
    def _find_meta(sid):
        if sid in tmt_meta_lkp:    return tmt_meta_lkp[sid]
        if sid in lookup_dp_id:    return lookup_dp_id[sid]
        if sid in lookup_np_id:    return lookup_np_id[sid]
        m = _TASK_RE.match(str(sid))
        email = m.group(1).strip().lower() if m else str(sid).strip().lower()
        return (
            lookup_np_email.get(email)
            or lookup_dp_email.get(email)
            or lookup_legacy.get(email)
        )

    rows = []
    for sid in sorted(all_subjects):
        meta = _find_meta(sid) or {}
        rows.append({"subject_id": sid, **{c: meta.get(c) for c in METADATA_COLS}})
    metadata = pd.DataFrame(rows)

    # ---- Clean ----
    metadata.loc[(metadata["age"] <= 0) | (metadata["age"] >= 100), "age"] = None
    metadata["education_level"] = metadata["education_level"].astype(str).str.strip().map(EDU_MAP)

    def _match_country(raw):
        if pd.isna(raw): return None
        m = difflib.get_close_matches(str(raw), CLEAN_COUNTRIES, n=1, cutoff=0.6)
        return m[0] if m else "Other"
    metadata["nationality_clean"] = metadata["nationality"].apply(_match_country)
    metadata["gender_desc"] = metadata["gender"].map(GENDER_MAP)

    print(f"Total subjects: {len(metadata)}")
    return metadata


def _annotate_age_stats(ax, ages, lang):
    """Overlay mean + ±1 SD on the age histogram for slide readability.

    A thick vertical line at the mean (on the age axis) with a tag at its top,
    plus a large text label with the exact mean/SD values.
    """
    m, s = ages.mean(), ages.std()
    mean_lbl = "Media" if lang == "es" else "Mean"
    sd_lbl = "DE" if lang == "es" else "SD"

    ax.axvline(m, color=C_STAT, lw=2.0, zorder=5)
    ymax = ax.get_ylim()[1]
    ax.text(m, ymax, f" {mean_lbl}", color=C_STAT, ha="left", va="top",
            fontsize=N_FS - 4)
    ax.text(0.97, 0.78, f"{mean_lbl} = {m:.0f}\n{sd_lbl} = {s:.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=N_FS)


def _generate_presentation_figures(metadata, lang, stats):
    """Save four separate, title-less, large-font PNGs (one per panel).

    Presentation variant for the FULL sample: large fonts, no titles, one file
    per demographic, sample size reported as an in-plot ``N`` annotation.
    """
    plt.rcParams.update(PRESENTATION_RCPARAMS)
    _titles, age_lbl, count_lbl, gender_map, edu_map, nat_map = _labels(lang)
    suffix = "_es" if lang == "es" else ""

    # --- A. Age distribution ---
    fig_a, ax_a = plt.subplots(figsize=PRES_FIGSIZE)
    ages = metadata["age"].dropna()
    ax_a.hist(ages, bins=20, color=C_AGE, edgecolor="white", linewidth=0.4)
    ax_a.set_xlabel(age_lbl)
    ax_a.set_ylabel(count_lbl)
    # N reports the subjects actually plotted (those with a valid age), which is
    # fewer than the full sample after dropping missing/corrupt ages.
    ax_a.text(0.97, 0.95, f"$N$ = {len(ages)}",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=N_FS)
    if stats:
        _annotate_age_stats(ax_a, ages, lang)
    fig_a.tight_layout()
    stat_suffix = "_stats" if stats else ""
    save_fig(fig_a, f"fig1_demographics_age{stat_suffix}{suffix}.png")

    # --- B. Gender ---
    fig_b, ax_b = plt.subplots(figsize=PRES_FIGSIZE)
    present_genders = [g for g in GENDER_ORDER if g in metadata["gender_desc"].values]
    gender_counts = metadata["gender_desc"].value_counts().reindex(present_genders)
    gender_labels = [gender_map.get(x, x) for x in gender_counts.index[::-1]]
    ax_b.barh(gender_labels, gender_counts.values[::-1], color=C_GENDER)
    ax_b.set_xlabel(count_lbl)
    fig_b.tight_layout()
    save_fig(fig_b, f"fig1_demographics_gender{suffix}.png")

    # --- C. Education level (ordered P->R) ---
    fig_c, ax_c = plt.subplots(figsize=PRES_FIGSIZE)
    present_edu = [e for e in EDU_ORDER if e in metadata["education_level"].values]
    edu_counts = metadata["education_level"].value_counts().reindex(present_edu)
    edu_labels = [edu_map[e] for e in edu_counts.index]
    ax_c.barh(edu_labels[::-1], edu_counts.values[::-1], color=C_EDUCATION)
    ax_c.set_xlabel(count_lbl)
    fig_c.tight_layout()
    save_fig(fig_c, f"fig1_demographics_education{suffix}.png")

    # --- D. Nationality ---
    fig_d, ax_d = plt.subplots(figsize=PRES_FIGSIZE)
    nat_counts = metadata["nationality_clean"].value_counts()
    nat_labels = [nat_map.get(x, x) for x in nat_counts.index[::-1]]
    ax_d.barh(nat_labels, nat_counts.values[::-1], color=C_NATIONALITY)
    ax_d.set_xlabel(count_lbl)
    fig_d.tight_layout()
    save_fig(fig_d, f"fig1_demographics_nationality{suffix}.png")


def main(lang="en", presentation=False, stats=False):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    titles, age_lbl, count_lbl, gender_map, edu_map, nat_map = _labels(lang)

    print("Loading data...")
    df_tmt, df_sst, df_cdt, df_gonogo = _load_data()
    metadata = _build_metadata(df_tmt, df_sst, df_cdt, df_gonogo)

    if presentation:
        _generate_presentation_figures(metadata, lang, stats)
        return

    # -----------------------------------------------------------------------
    # Figure 1 — 2x2 demographics panel (notebook cell 6)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # --- A. Age distribution ---
    ages = metadata["age"].dropna()
    ax_a.hist(ages, bins=20, color=C_DEMO, edgecolor="white", linewidth=0.4)
    ax_a.set_xlabel(age_lbl)
    ax_a.set_ylabel(count_lbl)
    ax_a.set_title(f"{titles['A']} ($N$={len(ages)})")

    # --- B. Gender ---
    present_genders = [g for g in GENDER_ORDER if g in metadata["gender_desc"].values]
    gender_counts = metadata["gender_desc"].value_counts().reindex(present_genders)
    gender_labels = [gender_map.get(x, x) for x in gender_counts.index[::-1]]
    ax_b.barh(gender_labels, gender_counts.values[::-1], color=C_DEMO)
    ax_b.set_xlabel(count_lbl)
    ax_b.set_title(f"{titles['B']} ($N$={metadata['gender_desc'].notna().sum()})")

    # --- C. Education level (ordered P->R) ---
    present_edu = [e for e in EDU_ORDER if e in metadata["education_level"].values]
    edu_counts  = metadata["education_level"].value_counts().reindex(present_edu)
    edu_labels  = [edu_map[e] for e in edu_counts.index]
    ax_c.barh(edu_labels[::-1], edu_counts.values[::-1], color=C_DEMO)
    ax_c.set_xlabel(count_lbl)
    ax_c.set_title(f"{titles['C']} ($N$={metadata['education_level'].notna().sum()})")

    # --- D. Nationality ---
    nat_counts = metadata["nationality_clean"].value_counts()
    nat_labels = [nat_map.get(x, x) for x in nat_counts.index[::-1]]
    ax_d.barh(nat_labels, nat_counts.values[::-1], color=C_DEMO)
    ax_d.set_xlabel(count_lbl)
    ax_d.set_title(f"{titles['D']} ($N$={metadata['nationality_clean'].notna().sum()})")

    fig.tight_layout()
    suffix = "_es" if lang == "es" else ""
    save_fig(fig, f"fig1_demographics{suffix}.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate the demographics figure")
    parser.add_argument("--lang", choices=["en", "es"], default="en",
                        help="Idioma de títulos/ejes/categorías (default: en)")
    parser.add_argument("--presentation", action="store_true",
                        help="Figuras separadas, sin títulos y con fuentes grandes "
                             "para presentación (todos los sujetos)")
    parser.add_argument("--stats", action="store_true",
                        help="Superpone media y ±1 DE en el panel de edad "
                             "(solo con --presentation)")
    args = parser.parse_args()
    main(args.lang, args.presentation, args.stats)
