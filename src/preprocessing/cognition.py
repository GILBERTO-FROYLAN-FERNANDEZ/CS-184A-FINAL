"""
Dataset	File	Subjects	Columns
COBRE	data/cobre_clinical/1139_Cobre_Neuropsych_V2_20261123.csv	189	111
BrainGluSchi	data/braingluschi_clinical/2079_Neurospsych_20261123.csv	175	61
--
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
        summarize_missingness
        flag_unusable_features
        validate_cognition_table
        export_cognition_table

Flow (per CV fold, fit on training rows only):
    transform_fold
        make_fold_preprocessor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_clinical_csv(path: Path) -> pd.DataFrame:
    """Load a clinical CSV with two header rows (names, then codes).

    Use header=0, skiprows=[1].
    """
    raise NotImplementedError


def load_raw_cognition(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw COBRE and BrainGluSchi neuropsych tables.

    Check that 'Anonymized ID' exists in both and log row, column and
    unique-ID counts for each.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------
def is_non_feature(col: str) -> bool:
    """Return True for ID, visit, days-since and demographic columns."""
    raise NotImplementedError


def find_shared_columns(cobre: pd.DataFrame, bgs: pd.DataFrame) -> list[str]:
    """Columns present in both files (expected: 61, in BrainGluSchi order).

    Warn about any BrainGluSchi column missing from COBRE, and about a
    count other than 61.
    """
    raise NotImplementedError


def select_feature_columns(shared_columns: list[str]) -> list[str]:
    """Remove non-feature columns from the shared set.

    Log which columns were dropped and how many true features remain.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# Visit handling
# --------------------------------------------------------------------------
def find_visit_order_column(df: pd.DataFrame) -> str | None:
    """Return the column that orders visits within a subject, or None.

    Candidates: 'Days since first enrollment', 'Days since enrollment', 'Visit'.
    """
    raise NotImplementedError


def retain_first_visit(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per 'Anonymized ID': the earliest visit.

    If no ordering column exists, fall back to file order and warn.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------
def prepare_study_table(df: pd.DataFrame, study: str, feature_cols: list[str]) -> pd.DataFrame:
    """First visit only; columns restricted to ID + features; add 'study' column."""
    raise NotImplementedError


def coerce_features_numeric(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Convert features to numeric and warn about values that became NaN."""
    raise NotImplementedError


def find_cross_study_duplicates(df: pd.DataFrame) -> list[str]:
    """IDs that appear in both studies (expected: up to 104 people)."""
    raise NotImplementedError


def resolve_cross_study_duplicates(
    df: pd.DataFrame, feature_cols: list[str], policy: str = "fewest_missing"
) -> pd.DataFrame:
    """Reduce people present in both studies to a single row.

    PLACEHOLDER: the final rule should come from the master manifest
    (src/data/) so all three modalities agree.
    Policies: 'fewest_missing', 'prefer_cobre', 'prefer_braingluschi'.
    """
    raise NotImplementedError


def merge_studies(
    cobre: pd.DataFrame,
    bgs: pd.DataFrame,
    feature_cols: list[str],
    dedup_policy: str = "fewest_missing",
) -> pd.DataFrame:
    """Stack both studies on the shared features; one row per person."""
    raise NotImplementedError


# --------------------------------------------------------------------------
# Quality checks
# --------------------------------------------------------------------------
def summarize_missingness(
    df: pd.DataFrame, feature_cols: list[str]
) -> dict[str, pd.DataFrame | pd.Series]:
    """Descriptive missingness: per feature, per subject, per study.

    Must not modify the data.
    """
    raise NotImplementedError


def flag_unusable_features(
    df: pd.DataFrame, feature_cols: list[str], max_missing_pct: float = 50.0
) -> list[str]:
    """Report (do not drop) features that are constant or mostly missing."""
    raise NotImplementedError


def validate_cognition_table(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Raise on: null or duplicate IDs, non-numeric features, unexpected
    study labels, or entirely-missing features."""
    raise NotImplementedError


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export_cognition_table(df: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> None:
    """Write cognition.csv (ID + features), cognition_study.csv (ID + study)
    and cognition_features.csv (feature names).

    No labels or demographics are written; join them from labels.csv
    on 'Anonymized ID'.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# Offline pipeline (no imputation or scaling)
# --------------------------------------------------------------------------
def build_cognition_table(
    data_dir: Path, out_dir: Path, dedup_policy: str = "fewest_missing"
) -> tuple[pd.DataFrame, list[str]]:
    """Raw CSVs -> validated, unimputed, unscaled subject-level feature table."""
    raise NotImplementedError


# --------------------------------------------------------------------------
# Fold-level preprocessing (fit on training rows only)
# --------------------------------------------------------------------------
def make_fold_preprocessor(strategy: str = "median") -> Pipeline:
    """Return a fresh, unfitted imputer + scaler pipeline for one fold."""
    raise NotImplementedError


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
    raise NotImplementedError


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    """Arguments: --data-dir, --out-dir, --dedup-policy."""
    raise NotImplementedError


def main() -> None:
    """Parse arguments and run build_cognition_table."""
    raise NotImplementedError


if __name__ == "__main__":
    main()