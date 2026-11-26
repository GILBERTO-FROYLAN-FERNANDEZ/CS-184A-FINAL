"""
Preprocess cognitive assessment data from COBRE and BrainGluSchi.

Select reviewed neuropsychological features, retain one assessment per
subject, and export an unimputed, unscaled table for model training.

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

Changes from the earlier version:
- Uses the reviewed feature whitelist instead of automatic column selection.
- Validates IDs and repeated visits, and supports a shared subject manifest.
- Exports fixed-schema sentinel indicators aligned by subject ID.
- Validates fold indices and keeps profiling output in a private location.
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

ID_COL = "Anonymized ID"
STUDY_COL = "study"
SOURCE_STUDY_COL = "source_study"

COBRE_FILE = "1139_Cobre_Neuropsych_V2_20261123.csv"
BGS_FILE = "2079_Neurospsych_20261123.csv"

VALID_STUDIES = {"COBRE", "BrainGluSchi"}
ID_PATTERN = re.compile(r"^A\d{8}$")

# Additional safeguards; the whitelist, not this set, determines features.
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
    return MISSING_CODES + UNRESOLVED_CODES


# --------------------------------------------------------------------------
# Loading and subject IDs
# --------------------------------------------------------------------------
def validate_source_ids(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate every ID before any visit selection can discard a row."""
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
    """Read readable headers and skip the second, coded-header row."""
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
    """Load the two neuropsychological files."""
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
    """Check known metadata names; not a substitute for the whitelist."""
    return column in METADATA_COLS


def load_approved_features(path: Path) -> list[str]:
    """Load the reviewed score-column whitelist.

    The CSV must contain a 'feature' column with the approved names.
    Additional columns (notes, groupings, ...) are ignored.
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
            f"{path}: expected {EXPECTED_N_FEATURES} features, found "
            f"{len(features)}"
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
    """Require every whitelisted feature to occur once in each file."""
    for name, df in (("COBRE", cobre), ("BrainGluSchi", bgs)):
        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            raise ValueError(f"{name}: duplicate headers: {duplicates}")

        missing = [feature for feature in features if feature not in df]
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
    """Return the available enrollment-day column, if any."""
    for column in VISIT_ORDER_CANDIDATES:
        if column in df.columns:
            return column
    return None


def retain_first_visit(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest recorded row per ID; refuse ambiguous repeats.

    Subjects with a single row are passed through unchanged, even if their
    visit-order value is missing or nonnumeric. Subjects with multiple rows
    must have numeric, non-missing visit-order values and a unique minimum.
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

    if out.loc[repeated, "_visit_order"].isna().any():
        raise ValueError(
            "Repeated subjects have missing or nonnumeric visit-order values"
        )

    repeated_rows = out[repeated]
    unique_rows = out[~repeated]

    earliest_day = repeated_rows.groupby(ID_COL)["_visit_order"].transform(
        "min"
    )
    earliest = repeated_rows.loc[
        repeated_rows["_visit_order"].eq(earliest_day)
    ]

    if earliest[ID_COL].duplicated().any():
        raise ValueError(
            "Some subjects have multiple rows tied for their earliest visit"
        )

    combined = pd.concat([unique_rows, earliest])
    dropped = len(df) - len(combined)
    if dropped:
        log.info(
            "First-visit selection dropped %d later rows using %s",
            dropped,
            order_column,
        )

    return combined.drop(columns="_visit_order").sort_index()


def prepare_study_table(
    df: pd.DataFrame,
    source_study: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Select ID and scores from one source after visit selection."""
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
    """Load a private manifest shared with the MRI pipelines.

    Required columns:
      'Anonymized ID': one row per subject
      'eligible': true/false or 1/0
      'study': canonical study assignment
      'cognition_source_study': clinical file supplying cognition scores
    """
    required = {ID_COL, "eligible", STUDY_COL, "cognition_source_study"}

    manifest = pd.read_csv(path, dtype={ID_COL: "string"})
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    manifest = validate_source_ids(manifest, "manifest")

    if manifest[ID_COL].duplicated().any():
        raise ValueError("Manifest contains duplicate subject IDs")

    eligibility = (
        manifest["eligible"].astype("string").str.strip().str.lower()
    )
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
    """Select eligible rows from the source specified by the shared manifest."""
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
            f"{joined.loc[absent, ID_COL].nunique()} cognition IDs "
            "are absent from the manifest"
        )

    joined = joined.drop(columns="_merge")
    selected = joined.loc[
        joined["eligible"]
        & joined[SOURCE_STUDY_COL].eq(joined["cognition_source_study"])
    ].copy()

    if selected[ID_COL].duplicated().any():
        raise ValueError(
            "Manifest selection produced duplicate cognition IDs"
        )

    expected = set(manifest.loc[manifest["eligible"], ID_COL])
    unavailable = expected - set(selected[ID_COL])
    if unavailable:
        log.warning(
            "%d eligible manifest subjects have no cognition row in "
            "their designated source",
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
    """Stack cognition rows, using a manifest if one is supplied.

    Without a manifest, source study is used provisionally as canonical
    study. That is acceptable for exporting cognition but does not
    establish cross-modal eligibility or canonical study assignment.
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
            "An ID occurs in both cognition files. Supply a shared "
            "--manifest; cognition.py must not choose a source independently."
        )

    log.warning(
        "No manifest supplied: exporting all cognition subjects. Cohort "
        "eligibility and canonical study must be checked downstream."
    )

    stacked[STUDY_COL] = stacked[SOURCE_STUDY_COL]
    return stacked.sort_values(ID_COL).reset_index(drop=True)


# --------------------------------------------------------------------------
# Numeric values and sentinel codes
# --------------------------------------------------------------------------
def coerce_features_numeric(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Convert reviewed score columns to numbers; reject unexpected text."""
    out = df.copy()

    for column in feature_cols:
        raw = out[column]
        converted = pd.to_numeric(raw, errors="coerce")

        unexpected = raw.notna() & converted.isna()
        if unexpected.any():
            examples = (
                raw.loc[unexpected].astype(str).unique()[:3].tolist()
            )
            raise ValueError(
                f"{column}: {int(unexpected.sum())} unexpected nonnumeric "
                f"values (e.g. {examples}); review the source data locally"
            )

        converted = converted.astype(float)
        if np.isinf(converted.to_numpy()).any():
            raise ValueError(f"{column}: infinite values present")

        out[column] = converted

    return out


def group_of(column: str) -> str:
    """Return a fixed test-group name for an approved score column."""
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
    """Return ID-keyed sentinel flags with a schema independent of values.

    Compute on original numeric scores, before sentinel replacement.
    Build via a dict of arrays so the DataFrame is constructed in a
    single step and does not trigger pandas fragmentation.
    """
    groups: dict[str, list[str]] = {}
    for column in feature_cols:
        groups.setdefault(group_of(column), []).append(column)

    flag_arrays: dict[str, np.ndarray] = {}
    for group, columns in groups.items():
        for code in all_missing_codes():
            name = f"{group}__code{code}"
            flag_arrays[name] = (
                df[columns].eq(code).any(axis=1).to_numpy().astype("int8")
            )

    flags = pd.DataFrame(flag_arrays)
    flags.insert(0, ID_COL, df[ID_COL].to_numpy())
    return flags


def log_code_indicator_rates(
    indicators: pd.DataFrame,
    study: pd.Series,
) -> pd.DataFrame:
    """Log per-study rates; return the full (possibly sparse) rate table."""
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
    """Replace approved sentinel codes with NaN in score columns only."""
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
    """Descriptive summaries; do not modify the input."""
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
    """Report features to investigate; do not select or drop them here."""
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
    """Validate the unimputed, unscaled subject-level score table."""
    if df.empty:
        raise ValueError("Cognition table is empty")

    missing = [column for column in feature_cols if column not in df]
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
        raise TypeError(
            f"Nonnumeric cognitive features remain: {list(nonnumeric)}"
        )

    values = df[feature_cols].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite cognitive values remain")

    remaining_codes = [
        column
        for column in feature_cols
        if df[column].isin(all_missing_codes()).any()
    ]
    if remaining_codes:
        raise ValueError(f"Sentinel codes remain in: {remaining_codes}")

    all_missing = [
        column for column in feature_cols if df[column].isna().all()
    ]
    if all_missing:
        raise ValueError(
            f"Entirely missing cognitive features: {all_missing}"
        )

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
    """Export private, ID-keyed feature tables; never export labels here."""
    if indicators[ID_COL].duplicated().any():
        raise ValueError("Duplicate IDs in indicator table")

    if set(indicators[ID_COL]) != set(df[ID_COL]):
        raise ValueError(
            "Indicator IDs do not match cognition table IDs"
        )

    flags = df[[ID_COL]].merge(
        indicators,
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    flag_columns = flags.columns.drop(ID_COL)
    if flags[flag_columns].isna().any().any():
        raise ValueError("Indicator alignment introduced missing flags")

    out_dir.mkdir(parents=True, exist_ok=True)

    df[[ID_COL] + feature_cols].to_csv(
        out_dir / "cognition.csv", index=False
    )
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
    """Build and export approved cognitive scores without fitting a model."""
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
    """Return a new, unfitted preprocessing pipeline for one fold."""
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
    """Fit on training positions only; transform train and validation."""
    train = np.asarray(train_idx)
    val = np.asarray(val_idx)

    for name, positions in (("train", train), ("validation", val)):
        if positions.ndim != 1 or not np.issubdtype(
            positions.dtype, np.integer
        ):
            raise ValueError(
                f"{name} indices must be a one-dimensional integer array"
            )
        if not len(positions) or len(np.unique(positions)) != len(positions):
            raise ValueError(f"{name} indices must be nonempty and unique")
        if (positions < 0).any() or (positions >= len(X)).any():
            raise ValueError(f"{name} indices are out of range")

    if np.intersect1d(train, val).size:
        raise ValueError("Train and validation indices overlap")

    if X.empty or not all(
        pd.api.types.is_numeric_dtype(t) for t in X.dtypes
    ):
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
    """Produce local QC counts; do not commit detailed profiles.

    Omits IDs, raw free-text values, and individual score extremes.
    Check the COINS agreement before sharing even aggregate counts.
    """
    cobre, bgs = load_raw_cognition(data_dir)
    report: dict = {"files": {}, "overlap": {}}

    for name, df in (("COBRE", cobre), ("BrainGluSchi", bgs)):
        scores = df.drop(
            columns=[column for column in METADATA_COLS if column in df],
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
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Defaults to DATA_DIR/whitelisted-features.csv.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional private shared subject manifest.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run private input QC instead of exporting features.",
    )
    parser.add_argument(
        "--profile-out",
        type=Path,
        default=Path("data/private/cognition_profile.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
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