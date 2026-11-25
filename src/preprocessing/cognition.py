"""
Preprocess cognitive assessment data from COBRE and BrainGluSchi.

Merge both datasets using their shared neuropsychological features,
retain each subject's first visit, and prepare the resulting feature
table for model training.

Missing-value imputation and feature standardization should be fitted
within each training fold to prevent data leakage.

Flow (offline, no imputation or scaling):
    build_cognition_table
        load_raw_cognition
            load_clinical_csv
        find_shared_columns
        select_feature_columns
            is_non_feature
        merge_studies
            prepare_study_table
                retain_first_visit
                    find_visit_order_column
            coerce_features_numeric
            resolve_cross_study_duplicates
                find_cross_study_duplicates
        make_code_indicators            (on original values, before replacement)
        log_code_indicator_rates
        replace_missing_codes           (9999 and 8888 -> NaN)
        summarize_missingness
        flag_unusable_features
        validate_cognition_table
        export_cognition_table

Flow (per CV fold, fit on training rows only):
    transform_fold
        make_fold_preprocessor

Separate utility (works on the raw files, counts only):
    profile_cognition_inputs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants (verified against the real headers of both neuropsych files)
# --------------------------------------------------------------------------
ID_COL = "Anonymized ID"
STUDY_COL = "study"  # should come from the manifest / enrollment, not the filename

COBRE_FILE = "1139_Cobre_Neuropsych_V2_20261123.csv"
BGS_FILE = "2079_Neurospsych_20261123.csv"  # sic

# Exact names; none of these may ever be a model feature.
METADATA_COLS = {
    "Anonymized ID",
    "Subject Type",                    # diagnosis label
    "Sub Study Label",                 # COBRE only
    "Days since sub study enrollment", # COBRE only
    "Days since first enrollment",     # COBRE only
    "Days since enrollment",           # BrainGluSchi only
    "Visit",
    "Age", "Gender", "Ethnicity", "Education",  # demographics = confounds
    "Administration Type",
}
VISIT_ORDER_CANDIDATES = ["Days since first enrollment", "Days since enrollment"]

# Missing-value codes (profile of the real files + COBRE data dictionary).
MISSING_CODES = [9999]      # data dictionary: "Not Applicable". Used instead of blanks
                            # (~0.5% of feature cells), usually whole tests at a time.
UNRESOLVED_CODES = [8888]   # NOT in the data dictionary, but common: in the INDI NP sheet 13 of
                            # 135 subjects (~10%) have at least one. It marks whole tests
                            # (raw and T-score both 8888), and 8888 in NAB_Mazes carries
                            # into the derived MATRICS Reasoning/Composite scores. Likely
                            # "attempted but not completed/invalid", which may be
                            # informative. Never let it reach a scaler as a number.
                            # Plan: convert to NaN AND keep a per-test indicator
                            # (see make_code_indicators); confirm the meaning with the source.

# Coding from the COBRE data dictionary (CNP_02, CNP_03, CNP_105).
# BrainGluSchi uses free text instead (m/f; hispanic, caucasian, ...), so these
# maps are for the confound module, not for the features.
GENDER_CODES = {1: "Male", 2: "Female"}
ETHNICITY_CODES = {1: "Caucasian", 2: "African-American", 3: "Hispanic"}
ADMINISTRATION_TYPE_CODES = {0: "Incorrect", 1: "Correct"}  # a QC flag, not a feature

EXPECTED_N_FEATURES = 52        # score columns present in both files
EXPECTED_SHARED_BY_NAME = 60    # 52 features + 8 shared metadata columns
EXPECTED_COBRE_SUBJECTS = 189
EXPECTED_BGS_SUBJECTS = 175

# Test groups used for sentinel-code indicators (prefix of the readable column name).
TEST_GROUP_PREFIXES = [
    ("WTAR", "WTAR_"), ("TMT_A", "TMT_A_"), ("BACS_SC", "BACS_SC_"),
    ("HVLT", "HVLT_"), ("WMS-III_SS", "WMS-III_SS_"), ("LNS", "LNS_"),
    ("NAB_Mazes", "NAB_Mazes_"), ("BVMT", "BVMT_"), ("FAS", "FAS_"),
    ("Animal_Fluency", "Animal_Fluency_"), ("MSCEIT", "MSCEIT_"), ("CPT-IP", "CPT-IP_"),
]  # each MatricsDomain_* column is its own group


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_clinical_csv(path: Path) -> pd.DataFrame:
    """Load a clinical CSV with two header rows (names, then codes).

    Only blank cells are read as missing, so strings such as "N/A" survive to
    `coerce_features_numeric`, which reports them instead of hiding them.
    """
    df = pd.read_csv(path, header=0, skiprows=[1], keep_default_na=False, na_values=[""])
    df.columns = df.columns.str.strip()
    return df


def load_raw_cognition(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw COBRE and BrainGluSchi neuropsych tables.

    Check that 'Anonymized ID' exists in both and log row, column and
    unique-ID counts for each.
    """
    cobre = load_clinical_csv(data_dir / "cobre_clinical" / COBRE_FILE)
    bgs = load_clinical_csv(data_dir / "braingluschi_clinical" / BGS_FILE)
    for name, df in (("COBRE", cobre), ("BrainGluSchi", bgs)):
        if ID_COL not in df.columns:
            raise KeyError(f"{name}: missing '{ID_COL}'. First columns: {list(df.columns[:6])}")
        log.info("%s: %d rows, %d columns, %d unique IDs",
                 name, len(df), df.shape[1], df[ID_COL].nunique())
    return cobre, bgs


# --------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------
def is_non_feature(col: str) -> bool:
    """True if col is in METADATA_COLS (exact-name match, not regex)."""
    return col in METADATA_COLS


def find_shared_columns(cobre: pd.DataFrame, bgs: pd.DataFrame) -> list[str]:
    """Columns present in both files, matched on readable name (not code:
    BrainGluSchi uses NEU_xx, COBRE uses CNP_xx), in BrainGluSchi order.

    Expected: 60. 'Days since enrollment' exists only in BrainGluSchi, so the
    Data Guide's "all 61 also in COBRE" is off by one (harmless: it is metadata).
    Warn on any other BrainGluSchi column missing from COBRE.
    """
    cobre_cols = set(cobre.columns)
    missing_in_cobre = [c for c in bgs.columns if c not in cobre_cols]
    unexpected = [c for c in missing_in_cobre if c != "Days since enrollment"]
    if unexpected:
        log.warning("BrainGluSchi columns absent from COBRE: %s", unexpected)
    shared = [c for c in bgs.columns if c in cobre_cols]
    if len(shared) != EXPECTED_SHARED_BY_NAME:
        log.warning("Expected %d shared columns, found %d.", EXPECTED_SHARED_BY_NAME, len(shared))
    return shared


def select_feature_columns(
    shared_columns: list[str], expected_n: int | None = EXPECTED_N_FEATURES
) -> list[str]:
    """Remove METADATA_COLS from the shared set.

    Expect EXPECTED_N_FEATURES (52); pass expected_n=None to skip that check.
    Raise if Age, Gender, Ethnicity, Education, Subject Type or Administration
    Type survive into the result. Log dropped columns.
    """
    features = [c for c in shared_columns if not is_non_feature(c)]
    dropped = [c for c in shared_columns if is_non_feature(c)]
    forbidden = {"Age", "Gender", "Ethnicity", "Education", "Subject Type", "Administration Type"}
    if leaked := forbidden & set(features):
        raise ValueError(f"Demographic/label columns in features: {sorted(leaked)}")
    if expected_n is not None and len(features) != expected_n:
        raise ValueError(f"Expected {expected_n} features, found {len(features)}. "
                         f"Dropped as metadata: {dropped}")
    log.info("Shared columns: %d | metadata dropped: %d %s | features: %d",
             len(shared_columns), len(dropped), dropped, len(features))
    return features


# --------------------------------------------------------------------------
# Visit handling
# --------------------------------------------------------------------------
def find_visit_order_column(df: pd.DataFrame) -> str | None:
    """Return the column that orders visits within a subject, or None.

    Candidates: VISIT_ORDER_CANDIDATES (COBRE has the first, BrainGluSchi the
    second). Both files also have 'Visit', which is not used for ordering.
    """
    for col in VISIT_ORDER_CANDIDATES:
        if col in df.columns:
            return col
    return None


def retain_first_visit(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per 'Anonymized ID': the earliest visit.

    Expected to be a no-op: each file has one row per subject (COBRE 189,
    BrainGluSchi 175) and 'Visit' is constant (COBRE 'n', BrainGluSchi
    'bsgBaseline'). Warn if any row is dropped, and check that the retained
    row's 'Visit' is the first visit.
    If no ordering column exists, fall back to file order and warn.
    """
    order_col = find_visit_order_column(df)
    if order_col is None:
        log.warning("No visit-ordering column found; using first row in file order.")
        first = df.drop_duplicates(ID_COL, keep="first").copy()
    else:
        ordered = df.assign(_order=pd.to_numeric(df[order_col], errors="coerce")).sort_values(
            [ID_COL, "_order"], na_position="last", kind="stable"
        )
        first = ordered.drop_duplicates(ID_COL, keep="first").drop(columns="_order")

    n_dropped = len(df) - len(first)
    if n_dropped:
        log.warning("First-visit filter dropped %d rows (ordered by %s).", n_dropped, order_col)
    if "Visit" in first.columns and first["Visit"].nunique(dropna=False) > 1:
        log.warning("Retained rows have several 'Visit' values: %s",
                    first["Visit"].value_counts(dropna=False).to_dict())
    return first.sort_index()


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------
def prepare_study_table(df: pd.DataFrame, study: str, feature_cols: list[str]) -> pd.DataFrame:
    """First visit only; columns restricted to ID + features; add 'study' column."""
    first = retain_first_visit(df)
    out = first[[ID_COL] + feature_cols].copy()
    out.insert(1, STUDY_COL, study)
    return out


def coerce_features_numeric(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Convert features to numeric and warn about values that became NaN."""
    out = df.copy()
    for col in feature_cols:
        converted = pd.to_numeric(out[col], errors="coerce")
        newly_nan = int(converted.isna().sum() - out[col].isna().sum())
        if newly_nan:
            bad = out.loc[converted.isna() & out[col].notna(), col].astype(str)
            log.warning("Column '%s': %d non-numeric values became NaN (e.g. %s)",
                        col, newly_nan, bad.unique()[:3].tolist())
        out[col] = converted.astype(float)
    return out


def _all_codes() -> list[int]:
    return list(MISSING_CODES) + list(UNRESOLVED_CODES)


def replace_missing_codes(
    df: pd.DataFrame, feature_cols: list[str], codes: list[int] | None = None
) -> pd.DataFrame:
    """Replace sentinel codes with NaN in feature columns, before anything else
    touches the values (scaling a 9999 or 8888 would wreck a column).

    `codes` defaults to MISSING_CODES + UNRESOLVED_CODES, so no raw sentinel
    survives into the exported table. Log the cells replaced per column, code
    and study. Call make_code_indicators on the ORIGINAL values first so the
    distinction between 8888 and 9999 is not lost.
    """
    codes = _all_codes() if codes is None else list(codes)
    out = df.copy()
    for code in codes:
        mask = out[feature_cols] == code
        n_cells = int(mask.to_numpy().sum())
        if not n_cells:
            continue
        by_study = mask.any(axis=1).groupby(out[STUDY_COL]).sum().astype(int).to_dict()
        per_col = mask.sum()
        log.info("Code %s: %d cells in %d columns; subjects affected by study: %s",
                 code, n_cells, int((per_col > 0).sum()), by_study)
        out[feature_cols] = out[feature_cols].mask(mask)
    return out


def _group_of(col: str) -> str:
    """Test group of a feature column (see TEST_GROUP_PREFIXES)."""
    if col.startswith("MatricsDomain_"):
        return col.replace("_T-Score", "")
    for group, prefix in TEST_GROUP_PREFIXES:
        if col.startswith(prefix):
            return group
    return col  # unknown test: its own group


def make_code_indicators(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Binary flags, one per test group and code, computed before replacement.

    Groups (by column prefix): WTAR, TMT_A, BACS_SC, HVLT, WMS-III_SS, LNS,
    NAB_Mazes, BVMT, FAS, Animal_Fluency, MSCEIT, CPT-IP, MATRICS domains.
    Output columns like 'TMT_A__code8888', 'CPT-IP__code9999'. Constant (all-0)
    flags are dropped. They use only the raw values (no labels, no cross-subject
    statistics), so computing them offline cannot leak.

    How well the flags predict diagnosis needs labels, so that check belongs in
    the evaluation module, not here.
    """
    groups: dict[str, list[str]] = {}
    for col in feature_cols:
        groups.setdefault(_group_of(col), []).append(col)

    flags = {}
    for group, cols in groups.items():
        for code in _all_codes():
            flag = (df[cols] == code).any(axis=1).astype(int)
            if flag.any():
                flags[f"{group}__code{code}"] = flag
    return pd.DataFrame(flags, index=df.index)


def log_code_indicator_rates(indicators: pd.DataFrame, study: pd.Series) -> pd.DataFrame:
    """Log and return the share of subjects with each flag, per study (percent)."""
    if indicators.empty:
        log.info("No sentinel-code indicators were triggered.")
        return pd.DataFrame()
    rates = indicators.groupby(study).mean().T * 100
    log.info("Sentinel-code rates (%% of subjects) by study:\n%s", rates.round(1).to_string())
    return rates


def find_cross_study_duplicates(df: pd.DataFrame) -> list[str]:
    """IDs that appear in both studies' cognition files.

    Profile result: 0. Keep as an assertion/safeguard.
    """
    n_studies = df.groupby(ID_COL)[STUDY_COL].nunique()
    return n_studies[n_studies > 1].index.tolist()


def resolve_cross_study_duplicates(
    df: pd.DataFrame, feature_cols: list[str], policy: str = "fewest_missing"
) -> pd.DataFrame:
    """Reduce people present in both studies to a single row.

    Expected to be a no-op: the profile found no ID in both neuropsych files.

    PLACEHOLDER: the final rule should come from the master manifest
    (src/data/) so all three modalities agree.
    Policies: 'fewest_missing', 'prefer_cobre', 'prefer_braingluschi'.
    """
    dup_ids = find_cross_study_duplicates(df)
    if not dup_ids:
        return df
    log.warning("%d subjects appear in both studies; resolving with policy '%s'",
                len(dup_ids), policy)

    def rank(frame: pd.DataFrame) -> pd.Series:
        if policy == "fewest_missing":
            unobserved = frame[feature_cols].isna() | frame[feature_cols].isin(_all_codes())
            return unobserved.sum(axis=1)
        if policy == "prefer_cobre":
            return (frame[STUDY_COL] != "COBRE").astype(int)
        if policy == "prefer_braingluschi":
            return (frame[STUDY_COL] != "BrainGluSchi").astype(int)
        raise ValueError(f"Unknown policy: {policy}")

    ranked = df.assign(_rank=rank(df)).sort_values([ID_COL, "_rank"], kind="stable")
    return ranked.drop_duplicates(ID_COL, keep="first").drop(columns="_rank")


def merge_studies(
    cobre: pd.DataFrame,
    bgs: pd.DataFrame,
    feature_cols: list[str],
    dedup_policy: str = "fewest_missing",
) -> pd.DataFrame:
    """Stack both studies on the shared features; one row per person.

    Sentinel codes are still present in the result (see replace_missing_codes).
    """
    stacked = pd.concat(
        [prepare_study_table(cobre, "COBRE", feature_cols),
         prepare_study_table(bgs, "BrainGluSchi", feature_cols)],
        ignore_index=True,
    )
    stacked = coerce_features_numeric(stacked, feature_cols)
    merged = resolve_cross_study_duplicates(stacked, feature_cols, dedup_policy)
    return merged.sort_values(ID_COL).reset_index(drop=True)


# --------------------------------------------------------------------------
# Quality checks
# --------------------------------------------------------------------------
def summarize_missingness(
    df: pd.DataFrame, feature_cols: list[str]
) -> dict[str, pd.DataFrame | pd.Series]:
    """Descriptive missingness: per feature, per subject, per study.

    Must not modify the data.
    """
    missing = df[feature_cols].isna()
    per_feature = pd.DataFrame({
        "n_missing": missing.sum(),
        "pct_missing": missing.mean() * 100,
    }).sort_values("pct_missing", ascending=False)
    per_subject = missing.mean(axis=1) * 100
    per_study = missing.groupby(df[STUDY_COL]).mean().T * 100
    return {"per_feature": per_feature, "per_subject": per_subject, "per_study": per_study}


def flag_unusable_features(
    df: pd.DataFrame, feature_cols: list[str], max_missing_pct: float = 50.0
) -> list[str]:
    """Report (do not drop) features that are constant or mostly missing."""
    flagged = [
        col for col in feature_cols
        if df[col].isna().mean() * 100 > max_missing_pct or df[col].nunique(dropna=True) <= 1
    ]
    if flagged:
        log.warning("Flagged features (constant or >%.0f%% missing): %s", max_missing_pct, flagged)
    return flagged


def validate_cognition_table(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Raise on: null or duplicate IDs, non-numeric features, unexpected
    study labels, or entirely-missing features."""
    if absent := [c for c in feature_cols if c not in df.columns]:
        raise KeyError(f"Feature columns missing from table: {absent}")
    if metadata := [c for c in feature_cols if is_non_feature(c)]:
        raise ValueError(f"Metadata columns listed as features: {metadata}")
    if df[ID_COL].isna().any():
        raise ValueError("Null subject IDs present.")
    if df[ID_COL].duplicated().any():
        raise ValueError("Duplicate subject IDs after first-visit/dedup step.")
    if df[feature_cols].select_dtypes(exclude="number").shape[1]:
        raise TypeError("Non-numeric feature columns remain.")
    if not set(df[STUDY_COL]).issubset({"COBRE", "BrainGluSchi"}):
        raise ValueError(f"Unexpected study labels: {df[STUDY_COL].unique()}")
    if leftovers := [c for c in feature_cols if df[c].isin(_all_codes()).any()]:
        raise ValueError(f"Sentinel codes remain in: {leftovers}")
    if all_missing := [c for c in feature_cols if df[c].isna().all()]:
        raise ValueError(f"Entirely-missing features: {all_missing}")
    log.info("Validation passed: %d subjects x %d features", len(df), len(feature_cols))


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export_cognition_table(
    df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    indicators: pd.DataFrame | None = None,
) -> None:
    """Write cognition.csv (ID + features), cognition_study.csv (ID + study),
    cognition_features.csv (feature names) and, if given, cognition_indicators.csv
    (ID + sentinel-code flags).

    No labels or demographics are written; join them from labels.csv
    on 'Anonymized ID'.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df[[ID_COL] + feature_cols].to_csv(out_dir / "cognition.csv", index=False)
    df[[ID_COL, STUDY_COL]].to_csv(out_dir / "cognition_study.csv", index=False)
    pd.Series(feature_cols, name="feature").to_csv(out_dir / "cognition_features.csv", index=False)
    if indicators is not None and not indicators.empty:
        pd.concat([df[[ID_COL]], indicators], axis=1).to_csv(
            out_dir / "cognition_indicators.csv", index=False)
    log.info("Exported cognition tables to %s", out_dir)


# --------------------------------------------------------------------------
# Offline pipeline (no imputation or scaling)
# --------------------------------------------------------------------------
def build_cognition_table(
    data_dir: Path, out_dir: Path, dedup_policy: str = "fewest_missing"
) -> tuple[pd.DataFrame, list[str]]:
    """Raw CSVs -> validated, unimputed, unscaled subject-level feature table.

    Sentinel codes (9999, 8888) become NaN; per-test indicator flags preserve
    where they occurred. Everything data-dependent (imputation, scaling) is
    left to `transform_fold`.
    """
    cobre, bgs = load_raw_cognition(data_dir)
    feature_cols = select_feature_columns(find_shared_columns(cobre, bgs))
    table = merge_studies(cobre, bgs, feature_cols, dedup_policy)

    counts = table[STUDY_COL].value_counts().to_dict()
    for study, expected in (("COBRE", EXPECTED_COBRE_SUBJECTS), ("BrainGluSchi", EXPECTED_BGS_SUBJECTS)):
        if counts.get(study) != expected:
            log.warning("%s: expected %d subjects, found %s", study, expected, counts.get(study))

    indicators = make_code_indicators(table, feature_cols)   # must precede replacement
    log_code_indicator_rates(indicators, table[STUDY_COL])
    table = replace_missing_codes(table, feature_cols)

    missingness = summarize_missingness(table, feature_cols)
    log.info("Overall missing after code replacement: %.2f%%",
             table[feature_cols].isna().mean().mean() * 100)
    flag_unusable_features(table, feature_cols)
    validate_cognition_table(table, feature_cols)

    export_cognition_table(table, feature_cols, out_dir, indicators)
    missingness["per_feature"].to_csv(out_dir / "cognition_missingness.csv")  # aggregate only
    return table, feature_cols


# --------------------------------------------------------------------------
# Fold-level preprocessing (fit on training rows only)
# --------------------------------------------------------------------------
def make_fold_preprocessor(strategy: str = "median") -> Pipeline:
    """Return a fresh, unfitted imputer + scaler pipeline for one fold."""
    return Pipeline([
        ("impute", SimpleImputer(strategy=strategy, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])


def transform_fold(
    X: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    strategy: str = "median",
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    """Fit imputer + scaler on X.iloc[train_idx] only; transform both partitions.

    Raise if train_idx and val_idx overlap. Call again inside each inner
    tuning split using only the outer-training rows.
    """
    if set(np.asarray(train_idx).tolist()) & set(np.asarray(val_idx).tolist()):
        raise ValueError("Train and validation indices overlap.")
    prep = make_fold_preprocessor(strategy)
    X_train = prep.fit_transform(X.iloc[train_idx])
    X_val = prep.transform(X.iloc[val_idx])
    return X_train, X_val, prep


# --------------------------------------------------------------------------
# Input profiling (counts only; safe to share)
# --------------------------------------------------------------------------
def profile_cognition_inputs(data_dir: Path, out_path: Path | None = None) -> dict:
    """Summarize the raw cognition inputs without exposing any subject IDs.

    Self-contained on purpose (reads the CSVs itself), so it works while the
    rest of this module is still stubs. Reports:
      * rows / columns / unique IDs / max rows per ID, per file
      * value counts of Subject Type, Sub Study Label, Visit, Administration
        Type, Gender, Ethnicity
      * per-column missingness, non-numeric strings (top 5), min/max, and
        counts of common sentinel codes (-9, -99, -999, -9999, 999, 9999,
        8888, 88888, 99999), and the 5 largest / smallest distinct values
      * Enrolled Substudies: raw-string counts, token counts, and family
        combinations (COBRE / BrainGluSchi / both)
      * overlap: number of IDs in both neuropsych files, and how each file's
        IDs split by enrollment family
    Prints JSON and optionally writes it to `out_path`.
    """
    import json

    categorical_cols = ["Subject Type", "Sub Study Label", "Visit",
                        "Administration Type", "Gender", "Ethnicity"]
    sentinels = [-9, -99, -999, -9999, 999, 9999, 8888, 88888, 99999]

    def read_two_header(path: Path) -> pd.DataFrame:
        # Only blank cells count as missing here, so strings like "N/A" or "NA"
        # stay visible in the non-numeric report instead of being hidden.
        return pd.read_csv(path, header=0, skiprows=[1],
                           keep_default_na=False, na_values=[""])

    def value_counts(series: pd.Series) -> dict[str, int]:
        return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}

    def substudy_family(raw: object) -> str:
        families = set()
        for token in str(raw).split(","):
            token = token.strip().lower()
            if not token:
                continue
            if token.startswith("cobre"):
                families.add("COBRE")
            elif "braingluschi" in token:
                families.add("BrainGluSchi")
            else:
                families.add(f"OTHER:{token}")
        return "+".join(sorted(families)) or "NONE"

    def profile_file(df: pd.DataFrame) -> dict:
        return {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "n_unique_ids": int(df[ID_COL].nunique()),
            "max_rows_per_id": int(df[ID_COL].value_counts().max()),
            "categorical": {c: value_counts(df[c]) for c in categorical_cols if c in df.columns},
        }

    def profile_columns(df: pd.DataFrame) -> dict:
        out = {}
        for col in df.columns:
            if col == ID_COL or col in categorical_cols:
                continue
            raw = df[col]
            num = pd.to_numeric(raw, errors="coerce")
            nonnumeric = raw[num.isna() & raw.notna()].astype(str)
            has_values = num.notna().any()
            out[col] = {
                "n_missing": int(raw.isna().sum()),
                "n_nonnumeric": int(len(nonnumeric)),
                "top_nonnumeric": {k: int(v) for k, v in nonnumeric.value_counts().head(5).items()},
                "min": float(num.min()) if has_values else None,
                "max": float(num.max()) if has_values else None,
                "largest_values": {str(k): int(v) for k, v in
                                   num.value_counts().sort_index(ascending=False).head(5).items()},
                "smallest_values": {str(k): int(v) for k, v in
                                    num.value_counts().sort_index().head(5).items()},
                "sentinel_counts": {str(v): int((num == v).sum()) for v in sentinels if (num == v).any()},
            }
        return out

    def profile_enrollment(enroll: pd.DataFrame) -> dict:
        col = "Enrolled Substudies"
        tokens = (enroll[col].astype(str).str.split(",").explode().str.strip().str.lower())
        return {
            "n_rows": int(len(enroll)),
            "n_unique_ids": int(enroll[ID_COL].nunique()),
            "raw_string_counts": value_counts(enroll[col]),
            "token_counts": value_counts(tokens),
            "family_counts": value_counts(enroll[col].map(substudy_family)),
        }

    cobre = read_two_header(data_dir / "cobre_clinical" / COBRE_FILE)
    bgs = read_two_header(data_dir / "braingluschi_clinical" / BGS_FILE)
    enroll = pd.read_csv(data_dir / "cobre_clinical" / "enrollment_summary.csv")  # single header row

    cobre_ids, bgs_ids = set(cobre[ID_COL]), set(bgs[ID_COL])
    family_by_id = dict(zip(enroll[ID_COL], enroll["Enrolled Substudies"].map(substudy_family)))

    def split_by_family(ids: set) -> dict[str, int]:
        return value_counts(pd.Series([family_by_id.get(i, "NOT_IN_ENROLLMENT") for i in ids]))

    report = {
        "cobre": profile_file(cobre),
        "braingluschi": profile_file(bgs),
        "enrollment": profile_enrollment(enroll),
        "overlap": {
            "ids_in_both_neuropsych_files": len(cobre_ids & bgs_ids),
            "cobre_file_ids_by_enrollment_family": split_by_family(cobre_ids),
            "braingluschi_file_ids_by_enrollment_family": split_by_family(bgs_ids),
        },
        "columns": {"cobre": profile_columns(cobre), "braingluschi": profile_columns(bgs)},
    }

    text = json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
    print(text)
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Arguments: --data-dir, --out-dir, --dedup-policy, --profile, --profile-out."""
    parser = argparse.ArgumentParser(description="Cognition preprocessing (COBRE + BrainGluSchi).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Folder containing cobre_clinical/ and braingluschi_clinical/")
    parser.add_argument("--out-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--dedup-policy", default="fewest_missing",
                        choices=["fewest_missing", "prefer_cobre", "prefer_braingluschi"])
    parser.add_argument("--profile", action="store_true",
                        help="Run profile_cognition_inputs (counts only) instead of the pipeline.")
    parser.add_argument("--profile-out", type=Path, default=Path("results/cognition_profile.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run build_cognition_table (or the profiler)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    if args.profile:
        profile_cognition_inputs(args.data_dir, args.profile_out)
    else:
        build_cognition_table(args.data_dir, args.out_dir, args.dedup_policy)


if __name__ == "__main__":
    main()