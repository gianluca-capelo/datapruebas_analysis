"""Generate the sample-demographics figure (2x2 panel).

Standalone replica of paper_figures.ipynb "Figure 1 — 2x2 demographics panel"
(plus the metadata-building cell it depends on), so the thesis demographics
figure is reproducible from the repo.

Panels:
    A. Age distribution   B. Gender
    C. Education level     D. Nationality

Usage:
    python -m analysis.scripts.generate_demographics_figure
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


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading data...")
    df_tmt, df_sst, df_cdt, df_gonogo = _load_data()
    metadata = _build_metadata(df_tmt, df_sst, df_cdt, df_gonogo)

    # -----------------------------------------------------------------------
    # Figure 1 — 2x2 demographics panel (notebook cell 6)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # --- A. Age distribution ---
    ages = metadata["age"].dropna()
    ax_a.hist(ages, bins=20, color=C_DEMO, edgecolor="white", linewidth=0.4)
    ax_a.set_xlabel("Age (years)")
    ax_a.set_ylabel("Count")
    ax_a.set_title(f"A. Age distribution ($N$={len(ages)})")

    # --- B. Gender ---
    present_genders = [g for g in GENDER_ORDER if g in metadata["gender_desc"].values]
    gender_counts = metadata["gender_desc"].value_counts().reindex(present_genders)
    ax_b.barh(gender_counts.index[::-1], gender_counts.values[::-1], color=C_DEMO)
    ax_b.set_xlabel("Count")
    ax_b.set_title(f"B. Gender ($N$={metadata['gender_desc'].notna().sum()})")

    # --- C. Education level (ordered P->R) ---
    present_edu = [e for e in EDU_ORDER if e in metadata["education_level"].values]
    edu_counts  = metadata["education_level"].value_counts().reindex(present_edu)
    edu_labels  = [EDU_LABELS[e] for e in edu_counts.index]
    ax_c.barh(edu_labels[::-1], edu_counts.values[::-1], color=C_DEMO)
    ax_c.set_xlabel("Count")
    ax_c.set_title(f"C. Education level ($N$={metadata['education_level'].notna().sum()})")

    # --- D. Nationality ---
    nat_counts = metadata["nationality_clean"].value_counts()
    ax_d.barh(nat_counts.index[::-1], nat_counts.values[::-1], color=C_DEMO)
    ax_d.set_xlabel("Count")
    ax_d.set_title(f"D. Nationality ($N$={metadata['nationality_clean'].notna().sum()})")

    fig.tight_layout()
    save_fig(fig, "fig1_demographics.png")


if __name__ == "__main__":
    main()
