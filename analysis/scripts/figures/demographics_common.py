"""Sample metadata and demographic panels shared by the demographics figures.

`demographics` (full sample) and `demographics_modeling` (the subjects that feed
the models) build the same metadata table from the same raw sources and draw the
same four panels; only the sample filter, the palette and the titles differ.

Replica of the paper_figures.ipynb metadata cell, so the thesis demographics are
reproducible from the repo.
"""
import difflib
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from src.config import BASE_DIR
from src.loader import (
    get_latest_cdt_analysis,
    get_latest_gonogo_analysis,
    get_latest_sst_analysis,
)
from src.model.datasetbuilder.dataset_builder import DatasetBuilder

from analysis.scripts.figures._style import (
    PANEL_RCPARAMS,
    SLIDE_RCPARAMS,
    use_science_style,
)
from analysis.scripts.figures.trial_data import latest_tmt_analysis_path

META_BASE = os.path.join(BASE_DIR, "data", "raw", "tmt")

PANEL_FIGSIZE = (4.5, 3.2)

PAPER_RCPARAMS = {
    "axes.titlesize": 13.5,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}


def use_demographics_style(slide: bool = False):
    """Style for the demographic panels.

    PANEL_RCPARAMS matters in both variants: scienceplots turns on top/right and
    minor ticks, which these bar panels are not drawn for.
    """
    use_science_style()
    plt.rcParams.update({
        **(SLIDE_RCPARAMS if slide else PAPER_RCPARAMS),
        **PANEL_RCPARAMS,
    })

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

def build_metadata(df_tmt, df_sst, df_cdt, df_gonogo):
    """Cleaned per-subject metadata table for every subject seen across tasks."""
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


def display_labels(lang):
    """Display labels for the requested language.

    Only display strings are translated; the underlying data (canonical English
    categories used for matching and ordering) is untouched.
    """
    es = lang == "es"
    titles = {
        "A": "A. Distribución de la edad" if es else "A. Age distribution",
        "B": "B. Género" if es else "B. Gender",
        "C": "C. Nivel educativo" if es else "C. Education level",
        "D": "D. Nacionalidad" if es else "D. Nationality",
    }
    gender = {
        "Female": "Femenino", "Male": "Masculino", "Non-binary": "No binario",
        "Gender fluid": "Género fluido", "Other": "Otro",
    } if es else {name: name for name in GENDER_ORDER}
    edu = {
        "P": "Primario", "S": "Secundario", "T": "Terciario",
        "U": "Universitario", "R": "Posgrado",
    } if es else dict(EDU_LABELS)
    nat = {"Mexico": "México", "Peru": "Perú", "Spain": "España",
           "Other": "Otros"} if es else {}
    return {
        "titles": titles,
        "age": "Edad (años)" if es else "Age (years)",
        "count": "Cantidad" if es else "Count",
        "mean": "Media" if es else "Mean",
        "sd": "DE" if es else "SD",
        "gender": gender,
        "education": edu,
        "nationality": nat,
    }


def load_task_analyses():
    """The four task DataFrames used to enumerate and deduplicate subjects."""
    tmt_path = latest_tmt_analysis_path()
    print(f"TMT analysis: {tmt_path}")
    df_tmt = pd.read_csv(tmt_path, on_bad_lines="warn")
    df_sst, _ = get_latest_sst_analysis()
    df_cdt, _ = get_latest_cdt_analysis()
    df_gonogo, _ = get_latest_gonogo_analysis()
    return df_tmt, df_sst, df_cdt, df_gonogo


def modeling_subject_ids() -> set:
    """subject_ids with a valid TMT — the sample the models are trained on."""
    return set(DatasetBuilder()._load_tmt_aggregated()["subject_id"])


def ordered_counts(metadata, column: str, order=None) -> pd.Series:
    """Value counts of a categorical column, in display order."""
    counts = metadata[column].value_counts()
    if order is None:
        return counts
    present = [value for value in order if value in metadata[column].values]
    return counts.reindex(present)


def draw_age_panel(ax, ages, labels, color, bins: int = 20):
    ax.hist(ages, bins=bins, color=color, edgecolor="white", linewidth=0.4)
    ax.set_xlabel(labels["age"])
    ax.set_ylabel(labels["count"])


def draw_category_panel(ax, counts, display_map, labels, color):
    """Horizontal bars, largest at the top (barh draws bottom-up)."""
    names = [display_map.get(value, value) for value in counts.index[::-1]]
    ax.barh(names, counts.values[::-1], color=color)
    ax.set_xlabel(labels["count"])


def annotate_n(ax, n: int, fontsize: int):
    ax.text(0.97, 0.95, f"$N$ = {n}", transform=ax.transAxes,
            ha="right", va="top", fontsize=fontsize)


def annotate_age_stats(ax, ages, labels, color, fontsize: int):
    """Overlay the mean and ±1 SD on the age histogram, for slide readability.

    A thick vertical line at the mean with a tag on top, plus a text block with
    the exact values.
    """
    mean, sd = ages.mean(), ages.std()
    ax.axvline(mean, color=color, lw=2.0, zorder=5)
    ax.text(mean, ax.get_ylim()[1], f" {labels['mean']}", color=color,
            ha="left", va="top", fontsize=fontsize - 4)
    ax.text(0.97, 0.78, f"{labels['mean']} = {mean:.0f}\n{labels['sd']} = {sd:.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=fontsize)
