"""
Evaluate schizophrenia-versus-control models on held-out people.

Question
--------
Do structural MRI, resting-state connectivity, or both improve prediction
beyond cognitive scores? Report modality-specific results, cross-study
transfer, and early-illness performance separately.

Inputs
------
The experiment YAML; raw demographics and enrollment CSVs; private,
ID-keyed feature tables; and optional private confound tables. The subject
table is built in memory. No subject manifest is exported.

Execution order
---------------
1. Load and validate the experiment configuration and feature schemas.
2. Build one subject table from demographics, enrollment, feature membership,
   and available QC information. Reject conflicting labels and unresolved
   canonical-study assignments.
3. Freeze each comparison cohort before fitting anything.
4. Generate three five-fold, person-level master splits. Restrict their
   assignments to each cohort and verify that every resulting fold is usable.
5. Run the configured shuffled-training-label diagnostic.
6. Within each outer split, tune the model and fit imputation/scaling on
   training people only. Obtain training-only out-of-fold probabilities to
   choose the threshold; predict the untouched outer-test people.
7. Average each person's held-out probabilities and thresholds across seeds.
   Compute ROC-AUC, balanced accuracy, F1, person-bootstrap intervals, paired
   permutation tests, and Holm correction for the confirmatory family.
8. Fit on one canonical study and evaluate on the other, in both directions.
9. Report an explicitly defined early-illness comparison and descriptive
   confound checks where their required inputs are available.
10. Save splits and person-level predictions privately, then export only
    reviewed aggregate results.

Important boundaries
--------------------
A split hash fingerprints assignments; it does not prove that an externally
supplied prediction was actually held out. Cross-study transfer is between
separately recruited studies at the same center/scanner type, not between
hospitals. A subject bootstrap of repeated-CV predictions does not capture
all model-refitting uncertainty.

No globally fitted imputer, scaler, feature selector, or ComBat output may
enter this runner. ComBat is intentionally not implemented: its train/test
transformation policy, particularly for an unseen study, must be resolved
before enabling it. Patient-only forms are never primary model features.

Private directories must be git-ignored. Review aggregate tables for small
cells and data-use-agreement restrictions before committing them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, clone
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

Config = dict[str, Any]
Cohort = pd.DataFrame
Splits = pd.DataFrame
Predictions = pd.DataFrame
ModelSpec = dict[str, Any]
Comparison = dict[str, Any]

ID = "Anonymized ID"
STUDIES = ("COBRE", "BrainGluSchi")
MODALITIES = ("cognition", "structure", "connectivity")
ID_REGEX = re.compile(r"^A\d{8}$")

# Registered factories must return a fresh sklearn-compatible estimator.
# The search space uses Pipeline parameter names, e.g. "model__C".
MODEL_FACTORIES: dict[
    str, tuple[Callable[[], BaseEstimator], dict[str, list[Any]]]
] = {
    "logistic_regression": (
        lambda: LogisticRegression(max_iter=3000, random_state=0),
        {"model__C": list(np.logspace(-3, 3, 30))},
    )
}

FORBIDDEN_FEATURE_NAMES = {
    "subject type", "label", "diagnosis", "age", "current age",
    "gender", "sex", "education", "ethnicity", "race", "study",
    "source_study", "eligible", "medication", "illness duration",
    "mean_fd_mm", "mean framewise displacement",
}


def register_model(
    name: str,
    factory: Callable[[], BaseEstimator],
    search_space: dict[str, list[Any]],
) -> None:
    """Register a cloneable estimator; tuning remains inside outer training."""
    if name in MODEL_FACTORIES:
        raise ValueError(f"Model already registered: {name}")
    if not callable(factory):
        raise TypeError("factory must be callable")
    MODEL_FACTORIES[name] = (factory, search_space)


def _ensure_xgboost() -> None:
    if "xgboost" in MODEL_FACTORIES:
        return
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return
    MODEL_FACTORIES["xgboost"] = (
        lambda: XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=0,
            n_jobs=1,
            tree_method="hist",
        ),
        {
            "model__n_estimators": [100, 200, 350],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__subsample": [0.75, 1.0],
            "model__colsample_bytree": [0.75, 1.0],
        },
    )


def _ids(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize and validate IDs without logging or exporting their values."""
    if ID not in frame.columns:
        raise ValueError(f"{source}: missing {ID!r}")
    out = frame.copy()
    out[ID] = out[ID].astype("string").str.strip()
    valid = out[ID].str.fullmatch(ID_REGEX.pattern).fillna(False)
    if not valid.all():
        raise ValueError(
            f"{source}: {int((~valid).sum())} invalid or missing subject IDs"
        )
    return out


def _read_csv(path: str | Path, *, two_headers: bool = False) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(
        path,
        header=0,
        skiprows=[1] if two_headers else None,
        dtype={ID: "string"},
        keep_default_na=False,
        na_values=[""],
    )
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError(f"{path}: duplicate column names")
    return _ids(frame, path.name)


def _unique(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    frame = _ids(frame, source)
    if frame[ID].duplicated().any():
        raise ValueError(f"{source}: multiple rows per person")
    return frame


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _needs_both_classes(frame: pd.DataFrame, context: str) -> None:
    if frame.empty or set(frame["label"].astype(int)) != {0, 1}:
        raise ValueError(f"{context}: both diagnosis classes are required")


# --------------------------------------------------------------------------
# 1. Configuration and inputs
# --------------------------------------------------------------------------
def load_config(path: Path) -> Config:
    """Read YAML and reject unsupported or inconsistent evaluation policies."""
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if config.get("n_folds") != 5 or len(config.get("seeds", [])) != 3:
        raise ValueError("Expected five folds and exactly three CV seeds")
    if len(set(config["seeds"])) != 3:
        raise ValueError("CV seeds must be distinct")
    if config.get("bootstrap", {}).get("resample_unit") != "person":
        raise ValueError("Bootstrap unit must be person")
    if config.get("metrics", {}).get("primary") != "roc_auc":
        raise ValueError("This implementation requires ROC-AUC as primary")
    if config.get("threshold", {}).get("rule") != "youden_on_training_predictions":
        raise ValueError("Unsupported threshold rule")
    if config.get("stratify_by") != ["label", "study"]:
        raise ValueError("Expected stratification by label and study")
    if config.get("multiple_comparisons", {}).get("method") != "holm":
        raise ValueError("Only Holm correction is implemented")
    if config.get("harmonization", {}).get("method") not in (None, "none"):
        raise ValueError(
            "ComBat/harmonization is blocked until a leakage-safe "
            "train/test policy is specified and implemented"
        )

    nested_control = config.get("early_illness", {}).get("early_control_group")
    top_control = config.get("early_control_group")
    if nested_control and top_control and nested_control != top_control:
        raise ValueError("Conflicting early-control-group settings")
    config["_early_control_group"] = (
        nested_control or top_control or "BrainGluSchi"
    )

    if not config.get("comparisons"):
        raise ValueError("Configure at least one comparison")
    return config


def _load_cognition_whitelist(config: Config) -> list[str]:
    path = Path(config["cognition_whitelist"])
    frame = pd.read_csv(path, dtype="string")
    if "feature" not in frame:
        raise ValueError(f"{path}: missing 'feature' column")
    names = frame["feature"].str.strip()
    if names.isna().any() or names.eq("").any() or names.duplicated().any():
        raise ValueError("Cognition whitelist has blank or duplicate features")
    return names.tolist()


def load_feature_tables(config: Config) -> dict[str, pd.DataFrame]:
    """Load ID-keyed numeric tables; require a documented connectivity export.

    A connectivity directory is supported only when it contains one
    connectivity.csv with one row per person. This function cannot infer how
    to flatten arbitrary matrices or select MRI sessions.
    """
    tables: dict[str, pd.DataFrame] = {}
    paths = config["features"]

    for modality in MODALITIES:
        path = Path(paths[modality])
        if path.is_dir():
            path = path / "connectivity.csv"
        table = _unique(_read_csv(path), str(path))

        metadata = [c for c in ("study", "session", "scan_id") if c in table]
        feature_columns = [c for c in table if c != ID and c not in metadata]
        if not feature_columns:
            raise ValueError(f"{modality}: no numeric feature columns")

        forbidden = [
            c for c in feature_columns
            if c.strip().lower() in FORBIDDEN_FEATURE_NAMES
        ]
        if forbidden:
            raise ValueError(f"{modality}: forbidden feature names: {forbidden}")

        values = table[feature_columns].apply(pd.to_numeric, errors="coerce")
        unexpected = values.isna() & table[feature_columns].notna()
        if unexpected.any().any():
            raise ValueError(f"{modality}: unexpected nonnumeric feature values")
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"{modality}: infinite feature values")

        if modality == "cognition":
            approved = _load_cognition_whitelist(config)
            if feature_columns != approved:
                raise ValueError(
                    "cognition.csv columns/order do not match the reviewed whitelist"
                )
            exported = pd.read_csv(config["cognition_features"], dtype="string")
            if "feature" not in exported or exported["feature"].tolist() != approved:
                raise ValueError(
                    "cognition_features.csv differs from the reviewed whitelist"
                )

        table[feature_columns] = values.astype(float)
        tables[modality] = table

    return tables


def derive_labels(
    demographics: dict[str, pd.DataFrame], config: Config
) -> pd.DataFrame:
    """Map diagnosis fields, rejecting conflicting diagnoses for one person."""
    control = set(config["subjects"]["label_map"]["control"])
    patient = set(config["subjects"]["label_map"]["schizophrenia"])
    excluded = set(config["subjects"]["exclude_subject_types"])
    early_type = config["subjects"]["early_subject_type"]

    rows: list[dict[str, Any]] = []
    for study, frame in demographics.items():
        required = {ID, "Subject Type", "Current Age", "Gender"}
        if not required.issubset(frame.columns):
            raise ValueError(
                f"{study} demographics missing {sorted(required - set(frame.columns))}"
            )

        for person, group in frame.groupby(ID, sort=False):
            diagnoses = set(group["Subject Type"].dropna().astype(str).str.strip())
            if len(diagnoses) != 1:
                raise ValueError(
                    f"{study}: conflicting or missing diagnoses for one person"
                )
            diagnosis = next(iter(diagnoses))
            if diagnosis not in control | patient | excluded:
                raise ValueError(f"{study}: unrecognized Subject Type")
            if diagnosis in excluded:
                label = None
            else:
                label = int(diagnosis in patient)

            # Several visits can change age. Prefer the lowest documented
            # enrollment day; otherwise use the first source row for confounds.
            order = next(
                (
                    c for c in (
                        "Days since first enrollment", "Days since enrollment"
                    )
                    if c in group
                ),
                None,
            )
            if order and len(group) > 1:
                days = pd.to_numeric(group[order], errors="coerce")
                if days.notna().all() and np.isfinite(days).all():
                    group = group.loc[[days.idxmin()]]

            row = group.iloc[0]
            rows.append(
                {
                    ID: person,
                    "demographic_source": study,
                    "subject_type": diagnosis,
                    "label": label,
                    "age": pd.to_numeric(
                        row["Current Age"], errors="coerce"
                    ),
                    "sex_raw": str(row["Gender"]).strip(),
                    "early": diagnosis == early_type,
                }
            )

    combined = pd.DataFrame(rows)
    for _, group in combined.groupby(ID):
        labels = set(group["label"].dropna().astype(int))
        if len(labels) > 1 or (
            group["label"].isna().any() and group["label"].notna().any()
        ):
            raise ValueError(
                "A person has conflicting eligible/excluded diagnoses "
                "across demographic sources"
            )
    return combined


def _enrollment_families(raw: object) -> set[str]:
    text = str(raw).lower()
    return {
        study for study in STUDIES
        if study.lower() in text
    }


def resolve_canonical_study(
    enrollment: pd.DataFrame,
    cognition_study: pd.DataFrame,
    config: Config,
    *,
    imaging_studies: dict[str, set[str]] | None = None,
) -> pd.Series:
    """Resolve study without guessing for dual-enrolled people.

    For dual enrollment, require exactly one imaging-source study. If the
    project decides to use cognition provenance instead, explicitly set
    subjects.canonical_study to cognition_study in YAML.
    """
    enrollment = _unique(enrollment, "enrollment")
    if "Enrolled Substudies" not in enrollment:
        raise ValueError("Enrollment file lacks 'Enrolled Substudies'")
    cognition_study = _unique(cognition_study, "cognition_study")
    if "study" not in cognition_study:
        raise ValueError("cognition_study.csv lacks 'study'")

    cognition_map = cognition_study.set_index(ID)["study"].to_dict()
    imaging_studies = imaging_studies or {}
    chosen: dict[str, str] = {}
    policy = config["subjects"].get("canonical_study")

    for _, row in enrollment.iterrows():
        person = row[ID]
        enrolled = _enrollment_families(row["Enrolled Substudies"])
        if not enrolled:
            raise ValueError("Enrollment contains an unrecognized study family")

        if len(enrolled) == 1:
            study = next(iter(enrolled))
        elif policy == "cognition_study":
            study = cognition_map.get(person)
            if study not in enrolled:
                raise ValueError(
                    "Dual enrollment cannot be resolved from cognition provenance"
                )
        else:
            imaging = imaging_studies.get(person, set()) & enrolled
            if len(imaging) != 1:
                raise ValueError(
                    "Dual-enrolled person has no unique imaging study. "
                    "Supply imaging study metadata, or explicitly approve "
                    "a different canonical-study policy."
                )
            study = next(iter(imaging))

        source = cognition_map.get(person)
        if source is not None and source != study:
            log.warning(
                "Canonical study and cognition source disagree for "
                "at least one person; see private audit"
            )
        chosen[person] = study

    return pd.Series(chosen, name="study", dtype="string")


def infer_modality_availability(
    tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Return one row per ID in the union of approved feature tables."""
    people = sorted(set().union(*(set(t[ID]) for t in tables.values())))
    frame = pd.DataFrame({ID: people})
    for modality, table in tables.items():
        frame[f"has_{modality}"] = frame[ID].isin(set(table[ID]))
    return frame


def build_subject_table(
    config: Config, tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Build the private in-memory cohort index from approved input sources."""
    paths = config["subjects"]["demographics"]
    demos = {
        study: _read_csv(path, two_headers=True)
        for study, path in paths.items()
    }
    diagnoses = derive_labels(demos, config)

    enrollment = _read_csv(config["subjects"]["enrollment_summary"])
    cognition_source = _read_csv(config["cognition_study"])

    # MRI provenance is optional metadata, never an input feature.
    imaging: dict[str, set[str]] = {}
    for modality in ("structure", "connectivity"):
        table = tables[modality]
        if "study" in table:
            if not table["study"].isin(STUDIES).all():
                raise ValueError(f"{modality}: invalid study provenance")
            for person, study in zip(table[ID], table["study"]):
                imaging.setdefault(person, set()).add(study)

    study = resolve_canonical_study(
        enrollment, cognition_source, config, imaging_studies=imaging
    )
    availability = infer_modality_availability(tables)
    people = diagnoses[[ID]].drop_duplicates().merge(
        availability, on=ID, how="left", validate="one_to_one"
    )
    for modality in MODALITIES:
        people[f"has_{modality}"] = (
            people[f"has_{modality}"].fillna(False).astype(bool)
        )

    people["study"] = people[ID].map(study)
    if people["study"].isna().any():
        raise ValueError(
            "Demographic IDs absent from enrollment or lacking canonical study"
        )

    selected = diagnoses.merge(
        people[[ID, "study"]], on=ID, validate="many_to_one"
    )
    selected = selected.loc[
        selected["demographic_source"].eq(selected["study"])
    ].copy()
    if selected[ID].duplicated().any():
        raise ValueError("More than one canonical demographic record per person")

    people = people.merge(
        selected[[ID, "subject_type", "label", "age", "sex_raw", "early"]],
        on=ID, how="left", validate="one_to_one",
    )
    if people["subject_type"].isna().any():
        raise ValueError(
            "Canonical study has no corresponding demographic assessment"
        )

    people["eligible"] = (
        people["label"].notna()
        & people[[f"has_{m}" for m in MODALITIES]].any(axis=1)
    )
    people["label"] = people["label"].astype("Int64")
    people["early"] = people["early"].fillna(False).astype(bool)

    motion_path = Path(config.get("confounds", {}).get("motion", ""))
    if motion_path.is_file():
        motion = _unique(_read_csv(motion_path), "motion")
        if "mean_fd_mm" not in motion:
            raise ValueError("motion.csv must contain mean_fd_mm")
        motion["mean_fd_mm"] = pd.to_numeric(
            motion["mean_fd_mm"], errors="coerce"
        )
        if (
            motion["mean_fd_mm"].isna().any()
            or (motion["mean_fd_mm"] < 0).any()
            or np.isinf(motion["mean_fd_mm"]).any()
        ):
            raise ValueError("Invalid mean_fd_mm values")
        people = people.merge(
            motion[[ID, "mean_fd_mm"]],
            on=ID, how="left", validate="one_to_one",
        )
    else:
        people["mean_fd_mm"] = np.nan

    max_fd = float(config["subjects"]["qc"]["fmri_max_mean_fd_mm"])
    people["connectivity_qc_pass"] = (
        people["has_connectivity"]
        & people["mean_fd_mm"].notna()
        & people["mean_fd_mm"].le(max_fd)
    )
    if people["has_connectivity"].any() and people.loc[
        people["has_connectivity"], "mean_fd_mm"
    ].isna().any():
        raise ValueError(
            "Functional features exist without matching motion QC. "
            "Provide ID-keyed motion.csv with mean_fd_mm."
        )

    return people.sort_values(ID).reset_index(drop=True)


def validate_subject_table(subjects: pd.DataFrame) -> None:
    """Require unique persons, valid canonical studies, and binary labels."""
    _unique(subjects, "subject table")
    if not subjects["study"].isin(STUDIES).all():
        raise ValueError("Invalid canonical study")
    eligible = subjects.loc[subjects["eligible"]]
    if eligible.empty or eligible["label"].isna().any():
        raise ValueError("No complete eligible cohort")
    if not eligible["label"].isin([0, 1]).all():
        raise ValueError("Nonbinary eligible labels")
    _needs_both_classes(eligible, "Eligible cohort")


def validate_feature_joins(
    subjects: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> None:
    """Reject unknown IDs, duplicate rows, and forbidden feature metadata."""
    known = set(subjects[ID])
    for name, table in tables.items():
        _unique(table, name)
        if not set(table[ID]).issubset(known):
            raise ValueError(f"{name}: feature IDs absent from demographics")
        if name == "cognition":
            extra = set(table.columns) - {ID} - set(
                _load_cognition_whitelist(_ACTIVE_CONFIG)
            )
        else:
            extra = set(table.columns) & {
                "label", "Subject Type", "Gender", "Current Age"
            }
        if extra:
            raise ValueError(f"{name}: unexpected columns {sorted(extra)}")


def load_confound_tables(config: Config) -> dict[str, pd.DataFrame]:
    """Read available optional ID-keyed confounds; never join to model X."""
    tables: dict[str, pd.DataFrame] = {}
    for name, raw_path in config.get("confounds", {}).items():
        path = Path(raw_path)
        if path.is_file():
            tables[name] = _unique(_read_csv(path), f"{name} confounds")
        else:
            log.warning("Optional confound file unavailable: %s", name)
    indicators = Path(config.get("cognition_indicators", ""))
    if indicators.is_file():
        tables["indicators"] = _unique(
            _read_csv(indicators), "cognition indicators"
        )
    return tables


# Set only during an evaluation run; prevents a second, drifting whitelist
# argument throughout the original draft's validate_feature_joins signature.
_ACTIVE_CONFIG: Config = {}


# --------------------------------------------------------------------------
# 2. Comparisons and frozen cohorts
# --------------------------------------------------------------------------
def define_comparisons(config: Config) -> list[Comparison]:
    """Return named comparison definitions with unique comparison names."""
    comparisons = config["comparisons"]
    names = [c["name"] for c in comparisons]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate comparison names")
    return comparisons


def validate_matched_pair(comparison: Comparison) -> None:
    """Ensure a confirmatory imaging-gain pair differs only in modalities."""
    baseline, candidate = comparison["baseline"], comparison["candidate"]
    union = set(comparison["modalities"])
    for spec in (baseline, candidate):
        if set(spec["modalities"]) - union:
            raise ValueError("Model modalities fall outside the frozen cohort")
        if spec.get("missing_modality", "complete_case") != "complete_case":
            raise ValueError(
                "This runner currently supports complete-case comparisons only"
            )
    if baseline["name"] == candidate["name"]:
        raise ValueError("Baseline and candidate need distinct names")
    if comparison.get("confirmatory"):
        if baseline["model"] != candidate["model"]:
            raise ValueError("Confirmatory pair must use the same model class")
        if not set(baseline["modalities"]) < set(candidate["modalities"]):
            raise ValueError(
                "Confirmatory candidate must add modalities to baseline"
            )


def freeze_cohort(
    subjects: pd.DataFrame, modalities: list[str]
) -> Cohort:
    """Freeze eligible people with every required modality and applicable QC."""
    if not modalities or set(modalities) - set(MODALITIES):
        raise ValueError("Unknown or empty modality set")
    keep = subjects["eligible"].copy()
    for modality in modalities:
        keep &= subjects[f"has_{modality}"]
    if "connectivity" in modalities:
        keep &= subjects["connectivity_qc_pass"]
    cohort = subjects.loc[keep].copy().reset_index(drop=True)
    _needs_both_classes(cohort, f"Cohort {modalities}")
    return cohort


def summarize_cohort(cohort: Cohort) -> dict[str, Any]:
    """Counts by study and diagnosis; results still require privacy review."""
    return {
        "n": len(cohort),
        "by_study_label": [
            {
                "study": str(study),
                "label": int(label),
                "n": int(n),
            }
            for (study, label), n in cohort.groupby(
                ["study", "label"], observed=True
            ).size().items()
        ],
    }


def summarize_qc_exclusions(
    subjects: pd.DataFrame, cohort: Cohort
) -> dict[str, Any]:
    """Aggregate eligible people absent from a frozen cohort by diagnosis."""
    missing = subjects.loc[
        subjects["eligible"] & ~subjects[ID].isin(cohort[ID])
    ]
    return {
        "not_in_cohort_by_label": {
            str(k): int(v)
            for k, v in missing.groupby("label", observed=True).size().items()
        }
    }


# --------------------------------------------------------------------------
# 3. Master splits
# --------------------------------------------------------------------------
def build_master_split(
    subjects: pd.DataFrame, config: Config
) -> Splits:
    """Assign stable folds using the joint label/study stratum."""
    eligible = subjects.loc[subjects["eligible"], [ID, "label", "study"]].copy()
    strata = eligible["label"].astype(str) + "|" + eligible["study"]
    minimum = strata.value_counts().min()
    if minimum < config["n_folds"]:
        raise ValueError(
            "At least one label-by-study stratum has fewer people than folds"
        )

    frames = []
    for seed in config["seeds"]:
        splitter = StratifiedKFold(
            n_splits=config["n_folds"], shuffle=True, random_state=seed
        )
        # Sort above is independent of diagnosis; positions are stable.
        assignment = np.full(len(eligible), -1, dtype=int)
        for fold, (_, test_positions) in enumerate(
            splitter.split(eligible, strata)
        ):
            assignment[test_positions] = fold
        part = eligible[[ID]].copy()
        part["seed"] = int(seed)
        part["fold"] = assignment
        frames.append(part)

    return pd.concat(frames, ignore_index=True)


def verify_split_independence(
    splits: Splits, subjects: pd.DataFrame
) -> None:
    """Require one test-fold assignment per eligible person and seed."""
    if splits.duplicated([ID, "seed"]).any():
        raise ValueError("A person has multiple fold assignments in one seed")
    target = set(subjects.loc[subjects["eligible"], ID])
    for _, frame in splits.groupby("seed"):
        if set(frame[ID]) != target:
            raise ValueError("A seed does not cover every eligible person")
        if (frame["fold"] < 0).any():
            raise ValueError("Unassigned test fold")


def restrict_split_to_cohort(
    splits: Splits, cohort: Cohort
) -> Splits:
    """Retain assignments without reshuffling; validate each restricted fold."""
    restricted = splits.loc[splits[ID].isin(cohort[ID])].copy()
    joined = restricted.merge(
        cohort[[ID, "label", "study"]],
        on=ID, validate="many_to_one"
    )
    for (seed, fold), test in joined.groupby(["seed", "fold"]):
        _needs_both_classes(test, f"Restricted test seed={seed} fold={fold}")
        train = joined.loc[
            joined["seed"].eq(seed) & joined["fold"].ne(fold)
        ]
        _needs_both_classes(train, f"Restricted train seed={seed} fold={fold}")
    expected = set(splits["seed"]) * len(cohort)
    if len(restricted) != expected:
        raise ValueError("Restricted split is incomplete")
    return restricted.drop(columns=["label", "study"])


def hash_splits(splits: Splits) -> str:
    """Fingerprint stable person/seed/fold assignments for private audits."""
    ordered = splits[[ID, "seed", "fold"]].sort_values(
        [ID, "seed", "fold"]
    )
    payload = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# 4. Models and predictions
# --------------------------------------------------------------------------
def validate_model_interface(spec: ModelSpec) -> None:
    """Accept a registered model or explicitly audited external predictions."""
    if not spec.get("name") or not spec.get("modalities"):
        raise ValueError("Model spec needs name and modalities")
    if "predictions" not in spec:
        _ensure_xgboost()
        if spec["model"] not in MODEL_FACTORIES:
            raise NotImplementedError(
                f"{spec['model']} has no registered estimator factory. "
                "Register the actual model implementation before evaluation."
            )


def record_tuning_budget(
    spec: ModelSpec, config: Config
) -> dict[str, Any]:
    """Read the configured inner-fold count and maximum search trials."""
    if "predictions" in spec:
        return {"mode": "external_predictions"}
    budget = config["models"][spec["model"]]["tuning_budget"]
    folds = int(budget["inner_folds"])
    trials = int(budget["n_trials"])
    if folds < 2 or trials < 1:
        raise ValueError("Invalid tuning budget")
    return {"inner_folds": folds, "n_trials": trials}


def _feature_matrix(
    spec: ModelSpec,
    people: pd.Index | list[str],
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Assemble named, numeric features without ID, study, or confounds."""
    target = pd.Index(people, name=ID)
    if not target.is_unique:
        raise ValueError("Repeated people in feature request")
    parts = []
    for modality in spec["modalities"]:
        table = tables[modality].set_index(ID)
        columns = [
            c for c in table
            if c not in ("study", "session", "scan_id")
        ]
        part = table.reindex(target)[columns]
        if part.isna().all(axis=1).any():
            raise ValueError(
                f"{modality}: subject lacks all features in a frozen cohort"
            )
        part.columns = [f"{modality}::{c}" for c in columns]
        parts.append(part)

    matrix = pd.concat(parts, axis=1)
    if matrix.columns.duplicated().any():
        raise ValueError("Duplicate model feature names")
    return matrix


def select_threshold(
    train_predictions: pd.Series,
    train_labels: pd.Series,
    rule: str,
) -> float:
    """Select Youden threshold from training-only OOF probabilities.

    Ties choose the lowest observed threshold. The test labels are never used.
    """
    if rule != "youden_on_training_predictions":
        raise ValueError(f"Unsupported threshold rule: {rule}")
    labels = np.asarray(train_labels, dtype=int)
    scores = np.asarray(train_predictions, dtype=float)
    if set(labels) != {0, 1}:
        raise ValueError("Threshold selection requires both classes")
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    best = -np.inf
    threshold = 0.5
    for value in candidates:
        positive = scores >= value
        sensitivity = positive[labels == 1].mean()
        specificity = (~positive[labels == 0]).mean()
        score = sensitivity + specificity - 1
        if score > best:
            best, threshold = score, float(value)
    return threshold


def fit_outer_fold(
    spec: ModelSpec,
    train_people: pd.Index,
    features: pd.DataFrame,
    labels: pd.Series,
    config: Config,
    *,
    seed: int,
) -> tuple[Pipeline, float, dict[str, Any]]:
    """Tune and fit exclusively on outer-training subjects.

    The selected hyperparameters are estimated on the outer-training set.
    Threshold probabilities are subsequently obtained by cross-validating
    those selected hyperparameters within that same outer-training set.
    This preserves outer-test isolation, although threshold optimization
    does not account for hyperparameter-selection uncertainty.
    """
    budget = record_tuning_budget(spec, config)
    folds = budget["inner_folds"]
    if labels.value_counts().min() < folds:
        raise ValueError("Not enough people in one class for inner CV")
    factory, space = MODEL_FACTORIES[spec["model"]]
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", factory()),
        ]
    )
    inner = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=space,
        n_iter=budget["n_trials"],
        cv=inner,
        scoring="roc_auc",
        refit=True,
        random_state=seed,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(features.loc[train_people], labels.loc[train_people])
    best = search.best_estimator_

    # clone(best) is unfitted; each threshold OOF fold fits its own imputer,
    # scaler and classifier without touching its validation people.
    inner_oof = cross_val_predict(
        clone(best),
        features.loc[train_people],
        labels.loc[train_people],
        cv=inner,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold = select_threshold(
        pd.Series(inner_oof, index=train_people),
        labels.loc[train_people],
        config["threshold"]["rule"],
    )
    return best, threshold, {
        "best_params": search.best_params_,
        "inner_best_auc": float(search.best_score_),
        "budget": budget,
    }


def run_cross_validation(
    spec: ModelSpec,
    cohort: Cohort,
    splits: Splits,
    tables: dict[str, pd.DataFrame],
    config: Config,
    split_hash: str,
    *,
    shuffle_training_labels_seed: int | None = None,
) -> Predictions:
    """Collect one actual outer-held-out probability per person and seed."""
    indexed = cohort.set_index(ID)
    matrix = _feature_matrix(spec, indexed.index, tables)
    labels = indexed["label"].astype(int)
    records = []

    for seed in config["seeds"]:
        assignment = splits.loc[
            splits["seed"].eq(seed)
        ].set_index(ID)["fold"]
        for fold in range(config["n_folds"]):
            test_people = assignment.index[assignment.eq(fold)]
            train_people = assignment.index[assignment.ne(fold)]
            y_train = labels.loc[train_people].copy()

            if shuffle_training_labels_seed is not None:
                # Permute inside each canonical study to preserve its
                # training-set prevalence. Score on ORIGINAL test labels.
                rng = np.random.default_rng(
                    shuffle_training_labels_seed
                    + 100_003 * int(seed)
                    + int(fold)
                )
                for study in STUDIES:
                    group = indexed.loc[train_people]
                    people = group.index[group["study"].eq(study)]
                    if len(people):
                        y_train.loc[people] = rng.permutation(
                            y_train.loc[people].to_numpy()
                        )

            fitted, threshold, _ = fit_outer_fold(
                spec, train_people, matrix, y_train, config, seed=int(seed)
            )
            probability = np.asarray(
                fitted.predict_proba(matrix.loc[test_people])[:, 1],
                dtype=float,
            )
            for person, value in zip(test_people, probability):
                records.append(
                    {
                        ID: person,
                        "seed": int(seed),
                        "fold": int(fold),
                        "label": int(labels.loc[person]),
                        "study": str(indexed.loc[person, "study"]),
                        "model": spec["name"],
                        "probability": float(value),
                        "threshold": float(threshold),
                        "split_hash": split_hash,
                    }
                )

    return pd.DataFrame.from_records(records)


def load_heldout_predictions(path: Path) -> Predictions:
    """Read external predictions; their training provenance is not provable."""
    return _unique_prediction_rows(_read_csv(path))


def _unique_prediction_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.duplicated([ID, "seed"]).any():
        raise ValueError("Duplicate person/seed prediction")
    return frame


def validate_prediction_provenance(
    predictions: Predictions,
    split_hash: str,
    cohort: Cohort,
    splits: Splits,
    spec: ModelSpec,
) -> None:
    """Audit coverage, labels, folds, probabilities and declared split hash."""
    required = {
        ID, "seed", "fold", "label", "study", "model",
        "probability", "threshold", "split_hash",
    }
    if required - set(predictions):
        raise ValueError(
            f"Predictions missing fields: {sorted(required - set(predictions))}"
        )
    predictions = _unique_prediction_rows(predictions)
    if predictions["split_hash"].nunique() != 1:
        raise ValueError("Prediction rows have differing split hashes")
    if predictions["split_hash"].iloc[0] != split_hash:
        raise ValueError("Prediction split hash differs from master split")
    if not predictions["model"].eq(spec["name"]).all():
        raise ValueError("Prediction model name differs from model spec")

    reference = splits.merge(
        cohort[[ID, "label", "study"]], on=ID, validate="many_to_one"
    )
    expected = reference[[ID, "seed", "fold", "label", "study"]].copy()
    observed = predictions[
        [ID, "seed", "fold", "label", "study"]
    ].copy()
    for column in ("seed", "fold", "label"):
        observed[column] = pd.to_numeric(
            observed[column], errors="raise"
        ).astype(int)
    if (
        len(predictions) != len(expected)
        or not observed.sort_values([ID, "seed"]).reset_index(
            drop=True
        ).equals(
            expected.sort_values([ID, "seed"]).reset_index(drop=True)
        )
    ):
        raise ValueError(
            "Prediction people, labels, study, seeds or folds "
            "do not match the frozen cohort"
        )

    for column in ("probability", "threshold"):
        values = pd.to_numeric(predictions[column], errors="raise")
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise ValueError(f"Invalid {column} values")
    # These checks cannot establish that an external producer trained only
    # on the claimed training folds.


# --------------------------------------------------------------------------
# 5. Shuffled-label diagnostic
# --------------------------------------------------------------------------
def shuffle_labels(cohort: Cohort, seed: int) -> Cohort:
    """Diagnostic utility; never replace the held-out scoring labels."""
    rng = np.random.default_rng(seed)
    changed = cohort.copy()
    for study, positions in changed.groupby("study").groups.items():
        changed.loc[positions, "label"] = rng.permutation(
            changed.loc[positions, "label"].to_numpy()
        )
    return changed


def run_shuffled_label_gate(
    specs: list[ModelSpec],
    cohort: Cohort,
    splits: Splits,
    tables: dict[str, pd.DataFrame],
    config: Config,
    split_hash: str,
) -> dict[str, float]:
    """Retrain with shuffled TRAINING labels; score ORIGINAL held-out labels.

    The proposed 0.55 rule is a heuristic diagnostic, not a leakage proof.
    """
    gate = config.get("shuffled_label_gate", {})
    if not gate.get("enabled", False):
        return {"status": "disabled"}  # type: ignore[return-value]

    if any("predictions" in spec for spec in specs):
        raise ValueError(
            "Cannot run shuffled-label retraining from external predictions. "
            "Disable the gate explicitly or provide trainable models."
        )

    results: dict[str, float] = {}
    for spec in specs:
        aucs = []
        for repeat in range(int(gate["n_repeats"])):
            raw = run_cross_validation(
                spec, cohort, splits, tables, config, split_hash,
                shuffle_training_labels_seed=90_000 + repeat,
            )
            averaged = average_across_seeds(pool_out_of_fold(raw))
            aucs.append(
                float(roc_auc_score(
                    averaged["label"], averaged["probability"]
                ))
            )
        mean_auc = float(np.mean(aucs))
        results[spec["name"]] = mean_auc
        if mean_auc > float(gate["max_mean_auc"]):
            raise RuntimeError(
                f"Shuffled-label diagnostic exceeded configured "
                f"mean-AUC limit for {spec['name']}; investigate before "
                "reporting results. This is a heuristic stop."
            )
    return results


# --------------------------------------------------------------------------
# 6. Metrics and statistics
# --------------------------------------------------------------------------
def compute_metrics(
    labels: pd.Series,
    probabilities: pd.Series,
    threshold: float | pd.Series,
) -> dict[str, float | None]:
    """Calculate metrics; leave one-class ROC-AUC undefined."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    t = np.asarray(threshold, dtype=float)
    if t.ndim == 0:
        t = np.repeat(t, len(y))
    if len(y) != len(p) or len(y) != len(t):
        raise ValueError("Mismatched metric input lengths")
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("Metrics received nonfinite values")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must lie in [0, 1]")
    if len(y) == 0:
        return {
            "roc_auc": None, "balanced_accuracy": None, "f1": None
        }

    predicted = (p >= t).astype(int)
    both = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, predicted))
            if both else None
        ),
        "f1": (
            float(f1_score(y, predicted, zero_division=0))
            if (y == 1).any() else None
        ),
    }


def pool_out_of_fold(predictions: Predictions) -> Predictions:
    """Check that each person has one held-out prediction per seed."""
    return _unique_prediction_rows(predictions.copy())


def average_across_seeds(predictions: Predictions) -> Predictions:
    """Reduce repeated CV to one probability and threshold per person."""
    for _, group in predictions.groupby(ID):
        if group["seed"].nunique() != len(_ACTIVE_CONFIG["seeds"]):
            raise ValueError("Person lacks a prediction for one or more seeds")
        if group["label"].nunique() != 1 or group["study"].nunique() != 1:
            raise ValueError("Inconsistent labels/studies across seeds")

    averaged = predictions.groupby(
        [ID, "label", "study", "model"], as_index=False
    ).agg(
        probability=("probability", "mean"),
        threshold=("threshold", "mean"),
    )
    if averaged[ID].duplicated().any():
        raise ValueError("Multiple models mixed before seed averaging")
    return averaged


def bootstrap_ci(
    labels: pd.Series,
    probabilities: pd.Series,
    config: Config,
    thresholds: pd.Series | None = None,
) -> dict[str, tuple[float, float] | None]:
    """Percentile intervals from independent person indices.

    One-class bootstrap resamples are skipped for metrics requiring two
    classes. The number of successful resamples is checked.
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    t = (
        np.full(len(y), 0.5)
        if thresholds is None
        else np.asarray(thresholds, dtype=float)
    )
    _needs_both_classes(
        pd.DataFrame({"label": y}), "Bootstrap input"
    )
    rng = np.random.default_rng(config["seeds"][0] + 20_001)
    samples: dict[str, list[float]] = {
        "roc_auc": [], "balanced_accuracy": [], "f1": []
    }
    n_resamples = int(config["bootstrap"]["n_resamples"])
    level = float(config["bootstrap"]["ci_level"])

    for _ in range(n_resamples):
        take = rng.integers(0, len(y), len(y))
        metrics = compute_metrics(
            pd.Series(y[take]), pd.Series(p[take]), pd.Series(t[take])
        )
        for metric, value in metrics.items():
            if value is not None:
                samples[metric].append(value)

    intervals: dict[str, tuple[float, float] | None] = {}
    alpha = (1 - level) / 2
    for metric, values in samples.items():
        if len(values) < max(100, n_resamples // 2):
            intervals[metric] = None
        else:
            lower, upper = np.quantile(
                values, [alpha, 1 - alpha]
            )
            intervals[metric] = (float(lower), float(upper))
    return intervals


def paired_permutation_test(
    labels: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
    config: Config,
) -> dict[str, float]:
    """Randomly swap each person's two model probabilities for AUC difference."""
    y = np.asarray(labels, dtype=int)
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(candidate, dtype=float)
    if len(y) != len(a) or len(y) != len(b) or len(set(y)) != 2:
        raise ValueError("Paired AUC needs aligned people and both classes")

    observed = float(roc_auc_score(y, b) - roc_auc_score(y, a))
    n = int(config["permutation"]["n_permutations"])
    if n < 1:
        raise ValueError("At least one permutation is required")
    rng = np.random.default_rng(config["seeds"][0] + 40_001)
    extreme = 0
    for _ in range(n):
        swap = rng.integers(0, 2, len(y)).astype(bool)
        perm_a = np.where(swap, b, a)
        perm_b = np.where(swap, a, b)
        difference = (
            roc_auc_score(y, perm_b) - roc_auc_score(y, perm_a)
        )
        extreme += abs(difference) >= abs(observed) - 1e-12
    return {
        "delta_auc": observed,
        "p_value": float((extreme + 1) / (n + 1)),
    }


def holm_correct(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjust only the caller-supplied confirmatory p-values."""
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(
            running, min(1.0, (count - rank) * p_values[key])
        )
        adjusted[key] = running
    return adjusted


def evaluate_model(
    spec: ModelSpec,
    predictions: Predictions,
    cohort: Cohort,
    config: Config,
) -> dict[str, Any]:
    """Score one model's averaged, person-level held-out predictions."""
    if set(predictions[ID]) != set(cohort[ID]):
        raise ValueError("Model predictions do not cover their frozen cohort")
    return {
        "model": spec["name"],
        "metrics": compute_metrics(
            predictions["label"],
            predictions["probability"],
            predictions["threshold"],
        ),
        "bootstrap_ci": bootstrap_ci(
            predictions["label"],
            predictions["probability"],
            config,
            predictions["threshold"],
        ),
    }


def evaluate_comparison(
    comparison: Comparison,
    results: dict[str, Predictions],
    cohort: Cohort,
    config: Config,
) -> dict[str, Any]:
    """Match identical people and labels before testing an imaging gain."""
    baseline = results[comparison["baseline"]["name"]].set_index(ID)
    candidate = results[comparison["candidate"]["name"]].set_index(ID)
    if (
        set(baseline.index) != set(candidate.index)
        or set(baseline.index) != set(cohort[ID])
    ):
        raise ValueError("Paired models do not cover identical people")
    candidate = candidate.loc[baseline.index]
    if not baseline["label"].equals(candidate["label"]):
        raise ValueError("Paired model labels disagree")
    if not baseline["study"].equals(candidate["study"]):
        raise ValueError("Paired model study assignments disagree")

    result = paired_permutation_test(
        baseline["label"],
        baseline["probability"],
        candidate["probability"],
        config,
    )
    result["confirmatory"] = bool(
        comparison.get("confirmatory", False)
    )
    return result


# --------------------------------------------------------------------------
# 7. Cross-study transfer
# --------------------------------------------------------------------------
def split_by_study(
    cohort: Cohort, train_study: str, test_study: str
) -> tuple[Cohort, Cohort]:
    """Partition by canonical study, requiring both classes on each side."""
    train = cohort.loc[cohort["study"].eq(train_study)].copy()
    test = cohort.loc[cohort["study"].eq(test_study)].copy()
    _needs_both_classes(train, f"Cross-study training {train_study}")
    _needs_both_classes(test, f"Cross-study testing {test_study}")
    return train, test


def assert_no_person_overlap(train: Cohort, test: Cohort) -> None:
    """Enforce person-disjoint train and test partitions."""
    if set(train[ID]) & set(test[ID]):
        raise ValueError("A person occurs in both cross-study partitions")


def summarize_study_differences(
    cohort: Cohort,
) -> dict[str, Any]:
    """Aggregate age, recorded sex values, and diagnosis balance."""
    summary = {}
    for study, group in cohort.groupby("study"):
        age = pd.to_numeric(group["age"], errors="coerce")
        summary[study] = {
            "n": int(len(group)),
            "mean_age": (
                float(age.mean()) if age.notna().any() else None
            ),
            "label_counts": {
                str(k): int(v)
                for k, v in group["label"].value_counts().items()
            },
            "sex_raw_counts": {
                str(k): int(v)
                for k, v in group["sex_raw"].value_counts().items()
            },
        }
    return summary


def run_cross_study(
    spec: ModelSpec,
    cohort: Cohort,
    tables: dict[str, pd.DataFrame],
    config: Config,
) -> dict[str, Any]:
    """Tune on training study only; score the untouched other study."""
    if "predictions" in spec:
        raise ValueError(
            "CV prediction files cannot substitute for cross-study predictions"
        )
    matrix = _feature_matrix(spec, cohort[ID].tolist(), tables)
    labels = cohort.set_index(ID)["label"].astype(int)
    outputs: dict[str, Any] = {}

    for direction in config["cross_study"]["directions"]:
        train, test = split_by_study(
            cohort, direction["train"], direction["test"]
        )
        assert_no_person_overlap(train, test)
        train_people = pd.Index(train[ID])
        test_people = pd.Index(test[ID])
        fitted, threshold, tuning = fit_outer_fold(
            spec, train_people, matrix, labels, config,
            seed=int(config["seeds"][0]),
        )
        probabilities = pd.Series(
            fitted.predict_proba(matrix.loc[test_people])[:, 1],
            index=test_people,
        )
        truth = labels.loc[test_people]
        name = f"{direction['train']}_to_{direction['test']}"
        outputs[name] = {
            "n_train": len(train_people),
            "n_test": len(test_people),
            "metrics": compute_metrics(truth, probabilities, threshold),
            "bootstrap_ci": bootstrap_ci(
                truth, probabilities, config,
                pd.Series(threshold, index=test_people),
            ),
            "tuning_budget": tuning["budget"],
            # Do not put person-level predictions into aggregate results.
        }

    return outputs


# --------------------------------------------------------------------------
# 8. Early illness
# --------------------------------------------------------------------------
def select_early_illness_cohort(
    subjects: pd.DataFrame,
    control_group: str,
) -> Cohort:
    """Select Early Sz people plus controls from the stated study."""
    if control_group not in STUDIES:
        raise ValueError("Unsupported early control group")
    patients = subjects["early"] & subjects["label"].eq(1)
    controls = (
        subjects["study"].eq(control_group)
        & subjects["label"].eq(0)
    )
    return subjects.loc[
        subjects["eligible"] & (patients | controls)
    ].copy()


def evaluate_early_illness(
    spec: ModelSpec,
    predictions: Predictions,
    early_cohort: Cohort,
    config: Config,
) -> dict[str, Any]:
    """Use existing held-out predictions; do not train on the subgroup."""
    selected = predictions.loc[
        predictions[ID].isin(early_cohort[ID])
    ]
    if selected.empty:
        return {"status": "unavailable"}
    if not selected["label"].eq(1).any():
        return {"status": "no_early_patients_in_model_cohort"}

    positive = selected.loc[selected["label"].eq(1)]
    sensitivity = float(
        (positive["probability"] >= positive["threshold"]).mean()
    )
    return {
        "model": spec["name"],
        "n_early_patients": len(positive),
        "n_controls": int(selected["label"].eq(0).sum()),
        "early_sensitivity": sensitivity,
        "early_mean_probability": float(
            positive["probability"].mean()
        ),
        "metrics_against_stated_controls": compute_metrics(
            selected["label"],
            selected["probability"],
            selected["threshold"],
        ),
    }


# --------------------------------------------------------------------------
# 9. Confound checks
# --------------------------------------------------------------------------
def _safe_spearman(
    left: pd.Series, right: pd.Series
) -> dict[str, float | int | None]:
    x = pd.to_numeric(left, errors="coerce")
    y = pd.to_numeric(right, errors="coerce")
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    count = int(valid.sum())
    if count < 6 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return {"n": count, "rho": None}
    return {
        "n": count,
        "rho": float(spearmanr(x[valid], y[valid]).statistic),
    }


def check_demographic_confounds(
    predictions: Predictions,
    cohort: Cohort,
    confounds: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Describe associations; do not claim confounding has been ruled out."""
    joined = predictions[[ID, "probability", "label"]].merge(
        cohort[[ID, "age", "sex_raw", "study", "mean_fd_mm"]],
        on=ID, validate="one_to_one",
    )
    out: dict[str, Any] = {
        "age_probability_spearman": _safe_spearman(
            joined["age"], joined["probability"]
        ),
        "study_balance": summarize_study_differences(cohort),
    }
    if joined["mean_fd_mm"].notna().any():
        out["motion_probability_spearman"] = _safe_spearman(
            joined["mean_fd_mm"], joined["probability"]
        )
    # Do not numerically equate COBRE's coded Gender with BGS free text.
    return out


def check_patient_only_confounds(
    predictions: Predictions,
    cohort: Cohort,
    confounds: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Relate held-out probabilities to available patient-only measures."""
    patients = predictions.loc[
        predictions["label"].eq(1), [ID, "probability"]
    ]
    output: dict[str, Any] = {}
    for name in ("medication", "illness_duration"):
        if name not in confounds:
            output[name] = {"status": "file_unavailable"}
            continue
        table = confounds[name]
        measurements = [c for c in table if c != ID]
        if len(measurements) != 1:
            raise ValueError(
                f"{name} must contain ID and exactly one reviewed "
                "numeric measurement"
            )
        joined = patients.merge(
            table, on=ID, how="left", validate="one_to_one"
        )
        output[name] = _safe_spearman(
            joined[measurements[0]], joined["probability"]
        )
    return output


def check_missingness_indicators(
    indicators: pd.DataFrame,
    cohort: Cohort,
) -> dict[str, Any]:
    """Report within-study, label-stratified sentinel rates."""
    joined = cohort[[ID, "study", "label"]].merge(
        indicators, on=ID, how="left", validate="one_to_one"
    )
    flags = [c for c in indicators if c != ID]
    output: dict[str, Any] = {}
    for study, group in joined.groupby("study"):
        output[study] = {}
        for label, subset in group.groupby("label"):
            active = {
                flag: float(pd.to_numeric(
                    subset[flag], errors="coerce"
                ).mean())
                for flag in flags
                if pd.to_numeric(
                    subset[flag], errors="coerce"
                ).fillna(0).sum() > 0
            }
            output[study][str(int(label))] = {
                "n": len(subset),
                "nonzero_flag_rates": active,
            }
    return output


# --------------------------------------------------------------------------
# 10. Interpretation
# --------------------------------------------------------------------------
def compute_permutation_importance(
    fitted: Pipeline,
    heldout_features: pd.DataFrame,
    heldout_labels: pd.Series,
    *,
    seed: int,
) -> pd.DataFrame:
    """Aggregate held-out AUC permutation importance for a fitted model."""
    if len(set(heldout_labels)) < 2:
        raise ValueError("Importance requires both held-out classes")
    result = permutation_importance(
        fitted,
        heldout_features,
        heldout_labels,
        scoring="roc_auc",
        n_repeats=5,
        random_state=seed,
        n_jobs=1,
    )
    return pd.DataFrame({
        "feature": heldout_features.columns,
        "mean_auc_decrease": result.importances_mean,
    })


def compute_shap_summary(
    fitted: Pipeline,
    heldout_features: pd.DataFrame,
    *,
    max_people: int = 50,
) -> pd.DataFrame:
    """Aggregate absolute SHAP values for supported sklearn estimators.

    SHAP is calculated on the held-out rows after applying the fitted
    training-only imputer and scaler. Unsupported model types fail rather
    than producing an invented explanation.
    """
    try:
        import shap
    except ImportError as error:
        raise ImportError(
            "SHAP interpretation requested, but shap is not installed"
        ) from error

    chosen = heldout_features.iloc[:max_people]
    transformed = fitted[:-1].transform(chosen)
    estimator = fitted.named_steps["model"]
    name = type(estimator).__name__

    if name == "LogisticRegression":
        background = transformed
        explainer = shap.LinearExplainer(estimator, background)
    elif name == "XGBClassifier":
        explainer = shap.TreeExplainer(estimator)
    else:
        raise NotImplementedError(
            f"SHAP summary is not implemented for {name}"
        )

    values = np.asarray(explainer.shap_values(transformed))
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.shape != transformed.shape:
        raise ValueError("Unexpected SHAP value shape")

    return pd.DataFrame({
        "feature": heldout_features.columns,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    })


# --------------------------------------------------------------------------
# 11. Export
# --------------------------------------------------------------------------
def export_aggregate_results(
    results: dict[str, Any],
    out_dir: Path,
) -> None:
    """Write aggregate JSON; review small cells before committing it."""
    if ID in json.dumps(_jsonable(results)):
        raise ValueError("Aggregate result contains an ID-column reference")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(_jsonable(results), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_private_audit_record(
    splits: Splits,
    predictions: dict[str, Predictions],
    split_hash: str,
    private_dir: Path,
) -> None:
    """Write subject-level assignments and probabilities only privately."""
    private_dir.mkdir(parents=True, exist_ok=True)
    splits.to_csv(private_dir / "evaluation_splits.csv", index=False)
    for name, frame in predictions.items():
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        frame.to_csv(
            private_dir / f"predictions_{safe}.csv", index=False
        )
    (private_dir / "evaluation_audit.json").write_text(
        json.dumps({
            "split_hash": split_hash,
            "prediction_sets": list(predictions),
            "heldout_prediction_mode_is_auditable_not_verifiable": True,
        }, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_evaluation(config_path: Path) -> dict[str, Any]:
    """Run the configured comparison family and return aggregates only."""
    global _ACTIVE_CONFIG
    config = load_config(config_path)
    _ACTIVE_CONFIG = config

    results_path = Path(config["results_dir"]).resolve()
    private_path = Path(config["private_dir"]).resolve()
    if (
        results_path == private_path
        or results_path in private_path.parents
        or private_path in results_path.parents
    ):
        raise ValueError(
            "Private and aggregate-output directories must be separate"
        )

    tables = load_feature_tables(config)
    subjects = build_subject_table(config, tables)
    validate_subject_table(subjects)
    validate_feature_joins(subjects, tables)
    confounds = load_confound_tables(config)

    comparisons = define_comparisons(config)
    for comparison in comparisons:
        validate_matched_pair(comparison)

    master = build_master_split(subjects, config)
    verify_split_independence(master, subjects)
    split_hash = hash_splits(master)

    aggregate: dict[str, Any] = {
        "split_hash": split_hash,
        "comparisons": {},
        "cross_study": {},
        "early": {},
        "confounds": {},
        "shuffled_label_diagnostic": {},
    }
    private_predictions: dict[str, Predictions] = {}

    for comparison in comparisons:
        cohort = freeze_cohort(
            subjects, comparison["modalities"]
        )
        restricted = restrict_split_to_cohort(master, cohort)
        specs = [
            comparison["baseline"], comparison["candidate"]
        ]
        for spec in specs:
            validate_model_interface(spec)
            record_tuning_budget(spec, config)

        gate = run_shuffled_label_gate(
            specs, cohort, restricted, tables, config, split_hash
        )
        aggregate["shuffled_label_diagnostic"][
            comparison["name"]
        ] = gate

        averaged: dict[str, Predictions] = {}
        model_summaries = {}
        for spec in specs:
            if "predictions" in spec:
                raw = load_heldout_predictions(
                    Path(spec["predictions"])
                )
            else:
                raw = run_cross_validation(
                    spec, cohort, restricted, tables,
                    config, split_hash,
                )
            validate_prediction_provenance(
                raw, split_hash, cohort, restricted, spec
            )
            private_predictions[
                f"{comparison['name']}_{spec['name']}"
            ] = raw
            mean = average_across_seeds(
                pool_out_of_fold(raw)
            )
            averaged[spec["name"]] = mean
            model_summaries[spec["name"]] = evaluate_model(
                spec, mean, cohort, config
            )

        aggregate["comparisons"][comparison["name"]] = {
            "cohort": summarize_cohort(cohort),
            "qc_exclusions": summarize_qc_exclusions(
                subjects, cohort
            ),
            "models": model_summaries,
            "paired": evaluate_comparison(
                comparison, averaged, cohort, config
            ),
        }
        aggregate["cross_study"][comparison["name"]] = {
            spec["name"]: run_cross_study(
                spec, cohort, tables, config
            )
            for spec in specs
        }

        baseline = averaged[specs[0]["name"]]
        aggregate["confounds"][comparison["name"]] = {
            "demographic": check_demographic_confounds(
                baseline, cohort, confounds
            ),
            "patient_only": check_patient_only_confounds(
                baseline, cohort, confounds
            ),
        }
        if "indicators" in confounds:
            aggregate["confounds"][
                comparison["name"]
            ]["cognition_indicators"] = (
                check_missingness_indicators(
                    confounds["indicators"], cohort
                )
            )

        early = select_early_illness_cohort(
            subjects, config["_early_control_group"]
        )
        for spec in specs:
            key = (
                f"{comparison['name']}/{spec['name']}"
            )
            aggregate["early"][key] = (
                evaluate_early_illness(
                    spec, averaged[spec["name"]],
                    early, config,
                )
            )

    confirmatory = {
        name: value["paired"]["p_value"]
        for name, value in aggregate["comparisons"].items()
        if value["paired"]["confirmatory"]
    }
    aggregate["holm_adjusted_confirmatory_p"] = (
        holm_correct(confirmatory)
    )

    # Save private records first. No aggregate file is written if the
    # evaluation fails before reaching this point.
    write_private_audit_record(
        master, private_predictions,
        split_hash, Path(config["private_dir"]),
    )
    export_aggregate_results(
        aggregate, Path(config["results_dir"])
    )
    return aggregate


def main(argv: list[str] | None = None) -> None:
    """Run evaluation from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate held-out schizophrenia classifiers"
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/config.yaml"),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    run_evaluation(args.config)


if __name__ == "__main__":
    main()