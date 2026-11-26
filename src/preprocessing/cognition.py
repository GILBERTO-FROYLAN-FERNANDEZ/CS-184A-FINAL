"""
Preprocess cognitive assessment data from COBRE and BrainGluSchi.

This module reads the neuropsychological CSVs, validates subject IDs and
reviewed feature names, retains one assessment per subject, and exports an
unimputed, unscaled cognition table for downstream modeling and evaluation.

The exported table is intentionally tabular-only:
- no MRI processing
- no scan-session selection
- no cross-modal fusion
- no global imputation or scaling

Those responsibilities belong to other modules in the project.

Flow (offline, no imputation or scaling):
    build_cognition_table
        load_approved_features
        load_raw_cognition
            load_clinical_csv
                validate_source_ids
        validate_approved_features
        load_cognition_manifest            (optional)
        merge_studies
            prepare_study_table
                retain_first_visit
                    find_visit_order_column
            select_manifest_cognition_rows (if manifest supplied)
        coerce_features_numeric
        make_code_indicators               (before replacement)
        log_code_indicator_rates
        replace_missing_codes              (9999 and 8888 -> NaN)
        summarize_missingness
        flag_unusable_features
        validate_cognition_table
        export_cognition_table

Flow (per CV fold, fit on training rows only):
    transform_fold
        make_fold_preprocessor

Separate utility (works on the raw files, private QC counts):
    profile_cognition_inputs

Notes from the reviewed COBRE workbooks:
- COBRE Data Dictionary maps the neuropsych columns to CNP codes and documents
  Administration Type as CNP_105, which is excluded from model features.
- COBRE Data Dictionary and COBRE INDI Additional data repeatedly document
  codes such as 9999, DK, and MD across many instruments.
- COBRE INDI Additional data also shows additional neuropsych fields beyond the
  reviewed shared set (for example WASI/WAIS/CPT-II fields), so this module
  uses an explicit whitelist instead of selecting all cognition-like columns.
- COBRE_scan_key shows repeated MRI sessions and series names such as
  mprage_5e, mprage_5e_RMS, RSTpre_V01_R01, dti_35dir, and GATE_V01_R01,
  confirming that scan/session handling should remain outside cognition.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ID_COL = "Anonymized ID"
STUDY_COL = "study"
SOURCE_STUDY_COL = "source_study"

COBRE_FILE = "1139_Cobre_Neuropsych_V2_20261123.csv"
BGS_FILE = "2079_Neurospsych_20261123.csv"

VALID_STUDIES = {"COBRE", "BrainGluSchi"}
ID_PATTERN = re.compile(r"^A\d{8}$")

# Additional safeguards. The actual feature contract is the external whitelist.
METADATA_COLS = {
    ID_COL,
    "Subject Type",
    "Sub Study Label",
    "Days since sub study enrollment",
    "Days since first enrollment",
    "Days since enrollment",
    "Visit",
    "Age",
    "Gender",
    "Ethnicity",
    "Education",
    "Administration Type",
    STUDY_COL,
    SOURCE_STUDY_COL,
}

VISIT_ORDER_CANDIDATES = (
    "Days since first enrollment",
    "Days since enrollment",
)

# Reviewed numeric sentinel handling for the approved cognitive score columns.
# 9999 is documented broadly in the COBRE dictionary as Not Applicable.
# 8888 appears in uploaded workbook excerpts for some non-cognitive fields
# (for example QOL money items as "Not applicable 8888"), so it is treated
# conservatively as a non-valid numeric score if encountered in approved
# cognition columns.
MISSING_CODES = (9999,)
UNRESOLVED_CODES = (8888,)

EXPECTED_N_FEATURES = 52

TEST_GROUP_PREFIXES = (
    ("WTAR", "WTAR_"),
    ("TMT_A", "TMT_A_"),
    ("BACS_SC", "BACS_SC_"),
    ("HVLT", "HVLT_"),
    ("WMS-III_SS", "WMS-III_SS_"),
    ("LNS", "LNS_"),
    ("NAB_Mazes", "NAB_Mazes_"),
    ("BVMT", "BVMT_"),
    ("FAS", "FAS_"),
    ("Animal_Fluency", "Animal_Fluency_"),
    ("MSCEIT", "MSCEIT_"),
    ("CPT-IP", "CPT-IP_"),
)


def all_missing_codes() -> tuple[int, ...]:
    """Return the reviewed numeric sentinel codes for approved score columns."""
    return MISSING_CODES + UNRESOLVED_CODES


# --------------------------------------------------------------------------
# Loading and subject IDs
# --------------------------------------------------------------------------
def validate_source_ids(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate and normalize subject IDs.

    Parameters
    ----------
    df
        DataFrame containing an Anonymized ID column.
    source
        Human-readable source name used in error messages.

    Returns
    -------
    pd.DataFrame
        Copy of the input with stripped string IDs.

    Raises
    ------
    KeyError
        If the ID column is missing.
    ValueError
        If any ID is null, blank, or malformed.

    Notes
    -----
    The uploaded scan-key workbook repeatedly shows subject IDs in the form
    A00028303, A00014522, etc., which is consistent with the expected
    A + 8 digits pattern enforced here.
    """
    if ID_COL not in df.columns:
        raise KeyError(f"{source}: missing {ID_COL!r}")

    out = df.copy()
    out[ID_COL] = out[ID_COL].astype("string").str.strip()

    invalid = out[ID_COL].isna() | ~out[ID_COL].str.fullmatch(
        ID_PATTERN.pattern
    ).fillna(False)
    if invalid.any():
        raise ValueError(
            f"{source}: {int(invalid.sum())} null, blank, or malformed IDs"
        )

    return out


def load_clinical_csv(path: Path) -> pd.DataFrame:
    """Load a clinical CSV with a readable first header row and coded second row.

    The neuropsych CSVs use:
    - row 1: readable names
    - row 2: coded names such as CNP_32

    This function keeps only blank cells as missing. Unexpected strings like
    DK or MD are preserved so numeric coercion can fail loudly if they appear
    in approved score columns.

    Parameters
    ----------
    path
        CSV path.

    Returns
    -------
    pd.DataFrame
        Loaded and ID-validated table.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If duplicate trimmed column names are detected.
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        header=0,
        skiprows=[1],
        dtype={ID_COL: "string"},
        keep_default_na=False,
        na_values=[""],
    )
    df.columns = df.columns.str.strip()

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"{path.name}: duplicate column names: {duplicates}")

    return validate_source_ids(df, path.name)


def load_raw_cognition(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the COBRE and BrainGluSchi raw neuropsych tables.

    Parameters
    ----------
    data_dir
        Directory containing cobre_clinical/ and braingluschi_clinical/.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (cobre, braingluschi)

    Notes
    -----
    This function does not load MRI-related workbooks or files. The uploaded
    COBRE scan key confirms that imaging sessions and series are tracked
    separately and should not be mixed into cognition preprocessing.
    """
    cobre = load_clinical_csv(data_dir / "cobre_clinical" / COBRE_FILE)
    bgs = load_clinical_csv(data_dir / "braingluschi_clinical" / BGS_FILE)

    for name, df in (("COBRE", cobre), ("BrainGluSchi", bgs)):
        log.info(
            "%s: %d rows, %d columns, %d unique IDs",
            name,
            len(df),
            df.shape[1],
            df[ID_COL].nunique(),
        )

    return cobre, bgs


# --------------------------------------------------------------------------
# Reviewed feature whitelist
# --------------------------------------------------------------------------
def is_non_feature(column: str) -> bool:
    """Return True for known metadata/confound/label columns.

    This is a safeguard only. The authoritative feature definition is the
    external whitelist loaded by load_approved_features().
    """
    return column in METADATA_COLS


def load_approved_features(path: Path) -> list[str]:
    """Load the reviewed cognition feature whitelist.

    The file must contain a column named 'feature'. Additional columns are
    ignored.

    Parameters
    ----------
    path
        Path to the whitelist CSV, e.g. data/whitelisted-features.csv.

    Returns
    -------
    list[str]
        Approved feature names in file order.

    Raises
    ------
    FileNotFoundError
        If the whitelist file does not exist.
    ValueError
        If the whitelist is malformed, contains blanks, duplicates, metadata,
        or the wrong number of features.

    Notes
    -----
    The uploaded COBRE Data Dictionary and COBRE INDI Additional data show many
    additional neuropsych or clinical variables beyond the 52 approved shared
    columns. That is why this script uses an explicit whitelist rather than
    inferring features from all available columns.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Feature whitelist not found: {path}")

    whitelist = pd.read_csv(path, dtype="string")

    if "feature" not in whitelist.columns:
        raise ValueError(f"{path}: must contain a 'feature' column")

    values = whitelist["feature"].str.strip()
    if values.isna().any() or values.eq("").any():
        raise ValueError(f"{path}: contains blank feature names")

    features = values.tolist()

    if len(features) != len(set(features)):
        raise ValueError(f"{path}: contains duplicate feature names")

    if len(features) != EXPECTED_N_FEATURES:
        raise ValueError(
            f"{path}: expected {EXPECTED_N_FEATURES} features, found {len(features)}"
        )

    forbidden = [name for name in features if is_non_feature(name)]
    if forbidden:
        raise ValueError(
            f"{path}: contains known metadata columns: {forbidden}"
        )

    return features


def validate_approved_features(
    cobre: pd.DataFrame,
    bgs: pd.DataFrame,
    features: list[str],
) -> None:
    """Validate that every approved feature exists in both cognition tables.

    Parameters
    ----------
    cobre, bgs
        Raw cognition DataFrames.
    features
        Reviewed feature list.

    Raises
    ------
    ValueError
        If any approved feature is absent from either file.

    Notes
    -----
    The uploaded COBRE dictionary excerpts include CNP mappings for the current
    neuropsych score columns, such as:
    - TMT_A_RawScore / TMT_A_T-Score
    - HVLT scores
    - BVMT scores
    - CPT-IP scores
    - MatricsDomain_* T-scores
    and Administration Type as CNP_105, which remains excluded.
    """
    for name, df in (("COBRE", cobre), ("BrainGluSchi", bgs)):
        missing = [feature for feature in features if feature not in df.columns]
        if missing:
            raise ValueError(
                f"{name}: approved features absent from file: {missing}"
            )

    log.info(
        "Validated %d whitelisted features in both source files",
        len(features),
    )


# --------------------------------------------------------------------------
# Visit selection and study rows
# --------------------------------------------------------------------------
def find_visit_order_column(df: pd.DataFrame) -> str | None:
    """Return the enrollment-day column used to order repeated visits.

    Returns
    -------
    str | None
        First available column among VISIT_ORDER_CANDIDATES, or None.
    """
    for column in VISIT_ORDER_CANDIDATES:
        if column in df.columns:
            return column
    return None


def retain_first_visit(df: pd.DataFrame) -> pd.DataFrame:
    """Retain one cognition row per subject, choosing the earliest visit.

    Subjects with a single row are passed through unchanged. Subjects with
    multiple rows must have a usable numeric visit-order column.

    Parameters
    ----------
    df
        Source cognition table.

    Returns
    -------
    pd.DataFrame
        One row per subject.

    Raises
    ------
    ValueError
        If repeated subjects exist but visit ordering is ambiguous.

    Notes
    -----
    The uploaded COBRE scan key clearly shows that some participants can have
    multiple sessions on the imaging side. This function handles repeated rows
    only within the cognition CSVs and does not attempt MRI session alignment.
    """
    if ID_COL not in df.columns:
        raise KeyError(f"Missing {ID_COL!r}")

    repeated = df[ID_COL].duplicated(keep=False)
    if not repeated.any():
        return df.copy()

    order_column = find_visit_order_column(df)
    if order_column is None:
        raise ValueError(
            "Repeated subjects exist, but no visit-order column is available"
        )

    out = df.copy()
    out["_visit_order"] = pd.to_numeric(out[order_column], errors="coerce")

    repeated_rows = out.loc[repeated].copy()
    unique_rows = out.loc[~repeated].copy()

    if repeated_rows["_visit_order"].isna().any():
        raise ValueError(
            "Repeated subjects have missing or nonnumeric visit-order values"
        )
    if np.isinf(repeated_rows["_visit_order"].to_numpy(dtype=float)).any():
        raise ValueError(
            "Repeated subjects have infinite visit-order values"
        )

    earliest_day = repeated_rows.groupby(ID_COL)["_visit_order"].transform("min")
    earliest = repeated_rows.loc[
        repeated_rows["_visit_order"].eq(earliest_day)
    ].copy()

    if earliest[ID_COL].duplicated().any():
        raise ValueError(
            "Some repeated subjects have tied earliest visits"
        )

    combined = pd.concat([unique_rows, earliest], axis=0).sort_index()
    dropped = len(df) - len(combined)
    if dropped:
        log.info(
            "First-visit selection dropped %d later rows using %s",
            dropped,
            order_column,
        )

    return combined.drop(columns="_visit_order")


def prepare_study_table(
    df: pd.DataFrame,
    source_study: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Create a study-specific cognition table.

    Parameters
    ----------
    df
        Raw study table.
    source_study
        COBRE or BrainGluSchi.
    feature_cols
        Approved cognition feature names.

    Returns
    -------
    pd.DataFrame
        ID + approved features + source_study.
    """
    if source_study not in VALID_STUDIES:
        raise ValueError(f"Unknown study: {source_study}")

    first = retain_first_visit(df)
    selected = first[[ID_COL] + feature_cols].copy()
    selected[SOURCE_STUDY_COL] = source_study
    return selected


# --------------------------------------------------------------------------
# Optional shared-manifest integration
# --------------------------------------------------------------------------
def load_cognition_manifest(path: Path) -> pd.DataFrame:
    """Load a shared manifest coordinating cognition with other modalities.

    Required columns
    ----------------
    Anonymized ID
        Unique subject ID.
    eligible
        Boolean-like indicator: true/false or 1/0.
    study
        Canonical study assignment.
    cognition_source_study
        Which cognition file supplies this subject's retained cognition row.

    Parameters
    ----------
    path
        Manifest CSV path.

    Returns
    -------
    pd.DataFrame
        Validated manifest.

    Notes
    -----
    This module does not build the manifest itself. If omitted, cognition.py
    exports all cognition rows and warns that cohort eligibility and canonical
    study must be enforced elsewhere.
    """
    required = {ID_COL, "eligible", STUDY_COL, "cognition_source_study"}

    manifest = pd.read_csv(path, dtype={ID_COL: "string"})
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    manifest = validate_source_ids(manifest, "manifest")

    if manifest[ID_COL].duplicated().any():
        raise ValueError("Manifest contains duplicate subject IDs")

    eligibility = manifest["eligible"].astype("string").str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}

    if not eligibility.isin(mapping).all():
        raise ValueError(
            "Manifest 'eligible' must contain only true/false or 1/0"
        )

    manifest["eligible"] = eligibility.map(mapping).astype(bool)

    eligible = manifest.loc[manifest["eligible"]]
    for column in (STUDY_COL, "cognition_source_study"):
        if eligible[column].isna().any():
            raise ValueError(f"Eligible subjects have missing {column}")
        if not eligible[column].isin(VALID_STUDIES).all():
            raise ValueError(f"Eligible subjects have invalid {column}")

    return manifest


def select_manifest_cognition_rows(
    stacked: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Filter stacked cognition rows according to a shared manifest.

    Parameters
    ----------
    stacked
        Combined COBRE + BrainGluSchi cognition rows with source_study.
    manifest
        Shared subject manifest.

    Returns
    -------
    pd.DataFrame
        Eligible cognition rows with canonical study assignment.
    """
    if stacked.duplicated([ID_COL, SOURCE_STUDY_COL]).any():
        raise ValueError("Duplicate cognition rows within a source study")

    joined = stacked.merge(
        manifest[[ID_COL, "eligible", STUDY_COL, "cognition_source_study"]],
        on=ID_COL,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    absent = joined["_merge"].eq("left_only")
    if absent.any():
        raise ValueError(
            f"{joined.loc[absent, ID_COL].nunique()} cognition IDs are absent from the manifest"
        )

    joined = joined.drop(columns="_merge")

    selected = joined.loc[
        joined["eligible"]
        & joined[SOURCE_STUDY_COL].eq(joined["cognition_source_study"])
    ].copy()

    if selected[ID_COL].duplicated().any():
        raise ValueError("Manifest selection produced duplicate cognition IDs")

    expected = set(manifest.loc[manifest["eligible"], ID_COL])
    unavailable = expected - set(selected[ID_COL])
    if unavailable:
        log.warning(
            "%d eligible manifest subjects have no cognition row in their designated source",
            len(unavailable),
        )

    return (
        selected.drop(columns=["eligible", "cognition_source_study"])
        .sort_values(ID_COL)
        .reset_index(drop=True)
    )


def merge_studies(
    cobre: pd.DataFrame,
    bgs: pd.DataFrame,
    feature_cols: list[str],
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge COBRE and BrainGluSchi cognition rows.

    Parameters
    ----------
    cobre, bgs
        Raw cognition tables.
    feature_cols
        Approved cognition features.
    manifest
        Optional shared manifest for subject selection.

    Returns
    -------
    pd.DataFrame
        Subject-level cognition table.

    Notes
    -----
    Without a manifest, source_study is copied to study and all cognition rows
    are exported. That is acceptable for preprocessing output but not sufficient
    to define the final multimodal cohort.
    """
    stacked = pd.concat(
        [
            prepare_study_table(cobre, "COBRE", feature_cols),
            prepare_study_table(bgs, "BrainGluSchi", feature_cols),
        ],
        ignore_index=True,
    )

    if manifest is not None:
        return select_manifest_cognition_rows(stacked, manifest)

    if stacked[ID_COL].duplicated().any():
        raise ValueError(
            "An ID occurs in both cognition files. Supply a shared --manifest; "
            "cognition.py must not choose a source independently."
        )

    log.warning(
        "No manifest supplied: exporting all cognition subjects. "
        "Cohort eligibility and canonical study must be checked downstream."
    )

    stacked[STUDY_COL] = stacked[SOURCE_STUDY_COL]
    return stacked.sort_values(ID_COL).reset_index(drop=True)


# --------------------------------------------------------------------------
# Numeric conversion and sentinel handling
# --------------------------------------------------------------------------
def coerce_features_numeric(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Convert approved feature columns to numeric.

    Parameters
    ----------
    df
        Cognition table.
    feature_cols
        Approved score columns.

    Returns
    -------
    pd.DataFrame
        Copy with numeric feature columns.

    Raises
    ------
    ValueError
        If unexpected nonnumeric values or infinities are found.

    Notes
    -----
    The uploaded COBRE dictionary documents textual codes like DK and MD in
    many instruments. If those appear in approved numeric cognition columns,
    this function fails explicitly rather than silently coercing them away.
    """
    out = df.copy()

    for column in feature_cols:
        raw = out[column]
        converted = pd.to_numeric(raw, errors="coerce")

        unexpected = raw.notna() & converted.isna()
        if unexpected.any():
            examples = raw.loc[unexpected].astype(str).unique()[:3].tolist()
            raise ValueError(
                f"{column}: {int(unexpected.sum())} unexpected nonnumeric "
                f"values (e.g. {examples})"
            )

        converted = converted.astype(float)
        if np.isinf(converted.to_numpy()).any():
            raise ValueError(f"{column}: infinite values present")

        out[column] = converted

    return out


def group_of(column: str) -> str:
    """Map an approved feature to its indicator test group.

    Parameters
    ----------
    column
        Approved feature name.

    Returns
    -------
    str
        Indicator group name.

    Raises
    ------
    ValueError
        If the feature is not mapped.
    """
    if column.startswith("MatricsDomain_"):
        return column.removesuffix("_T-Score")

    for group, prefix in TEST_GROUP_PREFIXES:
        if column.startswith(prefix):
            return group

    raise ValueError(
        f"Approved feature has no reviewed test-group mapping: {column}"
    )


def make_code_indicators(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Create fixed-schema sentinel indicators keyed by subject ID.

    Parameters
    ----------
    df
        Cognition table with original numeric values still present.
    feature_cols
        Approved cognition score columns.

    Returns
    -------
    pd.DataFrame
        ID + binary indicator columns for each group x sentinel-code pair.

    Notes
    -----
    The schema is fixed regardless of whether a given flag is all zero in the
    current dataset. This makes downstream joins and testing more stable.
    """
    groups: dict[str, list[str]] = {}
    for column in feature_cols:
        groups.setdefault(group_of(column), []).append(column)

    arrays: dict[str, np.ndarray] = {}
    for group, cols in groups.items():
        for code in all_missing_codes():
            name = f"{group}__code{code}"
            arrays[name] = (
                df[cols].eq(code).any(axis=1).to_numpy().astype("int8")
            )

    indicators = pd.DataFrame(arrays)
    indicators.insert(0, ID_COL, df[ID_COL].to_numpy())
    return indicators


def log_code_indicator_rates(
    indicators: pd.DataFrame,
    study: pd.Series,
) -> pd.DataFrame:
    """Log per-study prevalence of sentinel indicators.

    Parameters
    ----------
    indicators
        Indicator table returned by make_code_indicators().
    study
        Canonical study labels aligned row-wise with indicators.

    Returns
    -------
    pd.DataFrame
        Percent prevalence by study for all indicator columns.
    """
    flag_columns = indicators.columns.drop(ID_COL)
    rates = (
        indicators[flag_columns]
        .groupby(study.to_numpy())
        .mean()
        .T
        * 100
    )

    nonzero = rates.loc[rates.sum(axis=1) > 0]
    if nonzero.empty:
        log.info("No sentinel-code indicators triggered.")
    else:
        log.info(
            "Sentinel-code rates (%% of subjects, nonzero rows):\n%s",
            nonzero.round(1).to_string(),
        )

    return rates


def replace_missing_codes(
    df: pd.DataFrame,
    feature_cols: list[str],
    codes: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Replace reviewed numeric sentinel codes with NaN.

    Parameters
    ----------
    df
        Cognition table.
    feature_cols
        Approved score columns.
    codes
        Optional override; defaults to reviewed missing-code tuple.

    Returns
    -------
    pd.DataFrame
        Copy with sentinel values replaced by NaN.

    Notes
    -----
    This replacement happens after indicator creation so that the original
    distinction between 9999 and 8888 is preserved in cognition_indicators.csv.
    """
    codes = all_missing_codes() if codes is None else codes
    out = df.copy()

    for code in codes:
        mask = out[feature_cols].eq(code)
        count = int(mask.to_numpy().sum())
        if not count:
            continue

        by_study = (
            mask.any(axis=1)
            .groupby(out[STUDY_COL])
            .sum()
            .astype(int)
            .to_dict()
        )

        log.info(
            "Code %s: %d cells; subjects affected by study: %s",
            code,
            count,
            by_study,
        )
        out[feature_cols] = out[feature_cols].mask(mask)

    return out


# --------------------------------------------------------------------------
# Quality checks and export
# --------------------------------------------------------------------------
def summarize_missingness(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, pd.DataFrame | pd.Series]:
    """Summarize post-replacement missingness.

    Returns
    -------
    dict[str, pd.DataFrame | pd.Series]
        per_feature, per_subject, per_study summaries.
    """
    missing = df[feature_cols].isna()

    per_feature = pd.DataFrame(
        {
            "n_missing": missing.sum(),
            "pct_missing": missing.mean() * 100,
        }
    ).sort_values("pct_missing", ascending=False)

    return {
        "per_feature": per_feature,
        "per_subject": missing.mean(axis=1) * 100,
        "per_study": missing.groupby(df[STUDY_COL]).mean().T * 100,
    }


def flag_unusable_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    max_missing_pct: float = 50.0,
) -> list[str]:
    """Report features needing manual review.

    Parameters
    ----------
    df
        Cognition table.
    feature_cols
        Approved features.
    max_missing_pct
        Threshold for warning on high missingness.

    Returns
    -------
    list[str]
        Features that are constant or exceed the missingness threshold.
    """
    flagged = [
        column
        for column in feature_cols
        if (
            df[column].isna().mean() * 100 > max_missing_pct
            or df[column].nunique(dropna=True) <= 1
        )
    ]

    if flagged:
        log.warning("Features needing review: %s", flagged)

    return flagged


def validate_cognition_table(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """Validate the final unimputed, unscaled cognition table.

    Checks
    ------
    - non-empty output
    - approved feature presence
    - required study/source columns
    - valid subject IDs
    - no duplicate IDs
    - valid study/source labels
    - numeric features only
    - no infinities
    - no remaining sentinel codes
    - no entirely missing features
    """
    if df.empty:
        raise ValueError("Cognition table is empty")

    missing = [column for column in feature_cols if column not in df.columns]
    if missing:
        raise KeyError(f"Missing approved features: {missing}")

    for required in (STUDY_COL, SOURCE_STUDY_COL):
        if required not in df.columns:
            raise KeyError(f"Missing required column: {required}")

    validate_source_ids(df, "cognition output")

    if df[ID_COL].duplicated().any():
        raise ValueError("Duplicate cognition IDs")

    for column in (STUDY_COL, SOURCE_STUDY_COL):
        if df[column].isna().any():
            raise ValueError(f"Missing values in {column}")
        if not df[column].isin(VALID_STUDIES).all():
            raise ValueError(f"Unexpected values in {column}")

    nonnumeric = df[feature_cols].select_dtypes(exclude="number").columns
    if len(nonnumeric):
        raise TypeError(f"Nonnumeric cognitive features remain: {list(nonnumeric)}")

    values = df[feature_cols].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite cognitive values remain")

    remaining_codes = [
        column for column in feature_cols
        if df[column].isin(all_missing_codes()).any()
    ]
    if remaining_codes:
        raise ValueError(f"Sentinel codes remain in: {remaining_codes}")

    all_missing = [
        column for column in feature_cols if df[column].isna().all()
    ]
    if all_missing:
        raise ValueError(f"Entirely missing cognitive features: {all_missing}")

    log.info(
        "Validation passed: %d subjects x %d features",
        len(df),
        len(feature_cols),
    )


def export_cognition_table(
    df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    indicators: pd.DataFrame,
) -> None:
    """Export cognition preprocessing outputs.

    Files written
    -------------
    cognition.csv
        ID + approved cognition scores.
    cognition_study.csv
        ID + canonical study + source_study.
    cognition_features.csv
        The approved feature list used for this run.
    cognition_indicators.csv
        ID + fixed-schema sentinel indicators.
    cognition_missingness.csv
        Written separately by build_cognition_table().

    Notes
    -----
    No labels, demographics, or MRI/session fields are exported here.
    """
    if indicators[ID_COL].duplicated().any():
        raise ValueError("Duplicate IDs in indicator table")

    if set(indicators[ID_COL]) != set(df[ID_COL]):
        raise ValueError("Indicator IDs do not match cognition table IDs")

    flag_columns = indicators.columns.drop(ID_COL)
    if not indicators[flag_columns].isin([0, 1]).all().all():
        raise ValueError("Indicators must be binary 0/1 values")

    flags = df[[ID_COL]].merge(
        indicators,
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    if flags[flag_columns].isna().any().any():
        raise ValueError("Indicator alignment introduced missing flags")

    out_dir.mkdir(parents=True, exist_ok=True)

    df[[ID_COL] + feature_cols].to_csv(out_dir / "cognition.csv", index=False)
    df[[ID_COL, STUDY_COL, SOURCE_STUDY_COL]].to_csv(
        out_dir / "cognition_study.csv", index=False
    )
    pd.DataFrame({"feature": feature_cols}).to_csv(
        out_dir / "cognition_features.csv", index=False
    )
    flags.to_csv(out_dir / "cognition_indicators.csv", index=False)

    log.info("Exported cognition tables to %s", out_dir)


# --------------------------------------------------------------------------
# Offline pipeline
# --------------------------------------------------------------------------
def build_cognition_table(
    data_dir: Path,
    out_dir: Path,
    approved_features_path: Path | None = None,
    manifest_path: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the final unimputed, unscaled cognition table.

    Parameters
    ----------
    data_dir
        Root data directory.
    out_dir
        Output directory for exported cognition files.
    approved_features_path
        Optional whitelist CSV path. Defaults to data_dir/whitelisted-features.csv.
    manifest_path
        Optional shared manifest path.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Final cognition table and approved feature list.

    Notes
    -----
    This function intentionally does not produce a globally imputed or scaled
    table. Imputation and scaling must be fitted inside each training fold.
    """
    if approved_features_path is None:
        approved_features_path = data_dir / "whitelisted-features.csv"

    feature_cols = load_approved_features(approved_features_path)
    cobre, bgs = load_raw_cognition(data_dir)
    validate_approved_features(cobre, bgs, feature_cols)

    manifest = (
        load_cognition_manifest(manifest_path)
        if manifest_path is not None
        else None
    )

    table = merge_studies(cobre, bgs, feature_cols, manifest)
    table = coerce_features_numeric(table, feature_cols)

    log.info(
        "Cognition rows by source study: %s",
        table[SOURCE_STUDY_COL].value_counts().to_dict(),
    )

    indicators = make_code_indicators(table, feature_cols)
    log_code_indicator_rates(indicators, table[STUDY_COL])

    table = replace_missing_codes(table, feature_cols)
    missingness = summarize_missingness(table, feature_cols)

    log.info(
        "Overall missing after code replacement: %.2f%%",
        table[feature_cols].isna().mean().mean() * 100,
    )

    flag_unusable_features(table, feature_cols)
    validate_cognition_table(table, feature_cols)
    export_cognition_table(table, feature_cols, out_dir, indicators)

    missingness["per_feature"].to_csv(out_dir / "cognition_missingness.csv")

    return table, feature_cols


# --------------------------------------------------------------------------
# Fold-level preprocessing
# --------------------------------------------------------------------------
def make_fold_preprocessor(strategy: str = "median") -> Pipeline:
    """Create a fresh imputer+scaler pipeline for one fold.

    Parameters
    ----------
    strategy
        Imputation strategy for sklearn SimpleImputer.

    Returns
    -------
    Pipeline
        Unfitted sklearn pipeline.

    Notes
    -----
    This is used inside cross-validation only. The offline exports remain
    unimputed and unscaled by design.
    """
    return Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy=strategy, keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
        ]
    )


def transform_fold(
    X: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    strategy: str = "median",
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    """Fit preprocessing on training rows only and transform both splits.

    Parameters
    ----------
    X
        Numeric feature matrix.
    train_idx, val_idx
        Positional row indices for train and validation partitions.
    strategy
        Imputation strategy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, Pipeline]
        Transformed train array, transformed validation array, fitted pipeline.

    Raises
    ------
    ValueError
        If indices are invalid or output contains nonfinite values.

    Notes
    -----
    This function exists so evaluation can avoid leakage. It should be called
    inside each outer and inner CV split as needed.
    """
    train = np.asarray(train_idx)
    val = np.asarray(val_idx)

    for name, positions in (("train", train), ("validation", val)):
        if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
            raise ValueError(f"{name} indices must be a one-dimensional integer array")
        if not len(positions) or len(np.unique(positions)) != len(positions):
            raise ValueError(f"{name} indices must be nonempty and unique")
        if (positions < 0).any() or (positions >= len(X)).any():
            raise ValueError(f"{name} indices are out of range")

    if np.intersect1d(train, val).size:
        raise ValueError("Train and validation indices overlap")

    if X.empty or not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
        raise ValueError("X must contain numeric model features only")

    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("X contains infinite values")

    prep = make_fold_preprocessor(strategy)
    X_train = prep.fit_transform(X.iloc[train])
    X_val = prep.transform(X.iloc[val])

    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Fold preprocessing produced nonfinite values")

    return X_train, X_val, prep


# --------------------------------------------------------------------------
# Private input profiling
# --------------------------------------------------------------------------
def profile_cognition_inputs(
    data_dir: Path,
    out_path: Path | None = None,
) -> dict:
    """Generate private QC counts for the raw cognition inputs.

    Parameters
    ----------
    data_dir
        Root data directory.
    out_path
        Optional private JSON output path.

    Returns
    -------
    dict
        Aggregate QC report.

    Notes
    -----
    This utility is intentionally conservative:
    - no subject-level rows
    - no raw free-text dumps
    - no MRI/session joins

    Even aggregate outputs should be treated as private under the data-use
    agreement unless explicitly approved for sharing.
    """
    cobre, bgs = load_raw_cognition(data_dir)
    report: dict = {"files": {}, "overlap": {}}

    for name, df in (("COBRE", cobre), ("BrainGluSchi", bgs)):
        scores = df.drop(
            columns=[column for column in METADATA_COLS if column in df.columns],
            errors="ignore",
        )

        repeated_ids = (
            df[ID_COL]
            .duplicated(keep=False)
            .groupby(df[ID_COL])
            .any()
            .sum()
        )

        report["files"][name] = {
            "rows": len(df),
            "columns": df.shape[1],
            "unique_ids": int(df[ID_COL].nunique()),
            "subjects_with_multiple_rows": int(repeated_ids),
            "blank_cells_by_column": {
                column: int(count)
                for column, count in scores.isna().sum().items()
            },
        }

    report["overlap"]["ids_in_both_cognition_files"] = len(
        set(cobre[ID_COL]) & set(bgs[ID_COL])
    )

    text = json.dumps(report, indent=2)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

    print(text)
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for cognition preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess COBRE and BrainGluSchi cognition."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Contains cobre_clinical/ and braingluschi_clinical/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/features"),
        help="Output directory for exported cognition tables.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Optional whitelist CSV. Defaults to DATA_DIR/whitelisted-features.csv.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional shared subject manifest.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run private QC profiling instead of exporting cognition tables.",
    )
    parser.add_argument(
        "--profile-out",
        type=Path,
        default=Path("data/private/cognition_profile.json"),
        help="Private QC JSON output path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the cognition preprocessing CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    if args.profile:
        profile_cognition_inputs(args.data_dir, args.profile_out)
    else:
        build_cognition_table(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            approved_features_path=args.features,
            manifest_path=args.manifest,
        )


if __name__ == "__main__":
    main()