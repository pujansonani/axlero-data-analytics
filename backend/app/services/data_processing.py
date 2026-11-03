"""Dataset loading, validation, data-quality reporting and feature typing.

Nothing in this module *fits* statistics used by the model. Learned
preprocessing (imputation, scaling, encoding) lives in the sklearn pipeline
built in model_training.py and is fitted on the training split only.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

from app.services.errors import DataValidationError, InsufficientDataError

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MIN_TRAINING_ROWS = 50
MIN_ROWS_PER_CLASS = 10

# Labels that are commonly used for the *default* (positive) class.
KNOWN_DEFAULT_LABELS = {
    "1", "1.0", "true", "yes", "y", "default", "defaulted", "charged off", "charged_off",
    "chargedoff", "bad", "delinquent", "written off", "write off",
}
KNOWN_NON_DEFAULT_LABELS = {
    "0", "0.0", "false", "no", "n", "fully paid", "fully_paid", "paid", "repaid", "good",
    "non-default", "non_default", "no default", "current",
}

# Column names that commonly identify the target. Used only for *suggestions*.
TARGET_NAME_CANDIDATES = [
    "loan_status", "default", "is_default", "default_flag", "defaulted", "target",
    "loan_outcome", "bad_loan", "loan_default", "label", "y",
]

# Name fragments of fields that are only known *after* the loan outcome.
# Training on them would leak the target at loan-approval time.
LEAKAGE_NAME_PATTERNS = [
    "recover", "collection_recovery", "chargeoff", "charge_off", "charged_off", "settlement",
    "last_pymnt", "last_payment", "total_pymnt", "total_payment", "total_rec",
    "out_prncp", "outstanding_principal", "next_pymnt", "default_date", "days_past_due",
    "write_off", "writeoff", "hardship", "paid_off", "payoff_date", "post_default",
    "closed_date", "loss_amount", "net_loss",
]

ID_NAME_RE = re.compile(r"(^id$|_id$|^id_|uuid|guid|^member_?id|^loan_?id|^customer_?id|^account_?(no|number))", re.I)
NUMERIC_CLEAN_RE = re.compile(r"[,\s$€£%]")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_csv(source: bytes | str | Path) -> pd.DataFrame:
    """Load a CSV from raw bytes or a path, raising DataValidationError on failure."""
    try:
        if isinstance(source, (bytes, bytearray)):
            if not source.strip():
                raise DataValidationError("Uploaded file is empty.")
            df = pd.read_csv(io.BytesIO(source), low_memory=False)
        else:
            df = pd.read_csv(source, low_memory=False)
    except DataValidationError:
        raise
    except UnicodeDecodeError as exc:
        raise DataValidationError("File is not valid UTF-8 encoded CSV text.") from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise DataValidationError(f"Could not parse CSV: {exc}") from exc

    if df.empty or df.shape[1] == 0:
        raise DataValidationError("CSV contains no data rows.")
    df.columns = [str(c).strip() for c in df.columns]
    dupes = [c for c in df.columns if list(df.columns).count(c) > 1]
    if dupes:
        raise DataValidationError("CSV has duplicate column names.", {"columns": sorted(set(dupes))})
    if any(c == "" or c.lower().startswith("unnamed:") for c in df.columns):
        # Index columns written by pandas (e.g. "Unnamed: 0") carry no information.
        df = df.loc[:, [c for c in df.columns if not (c == "" or c.lower().startswith("unnamed:"))]]
    return df


# --------------------------------------------------------------------------- #
# Target handling
# --------------------------------------------------------------------------- #
def _norm_label(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip().lower()


@dataclass
class TargetMapping:
    target_column: str
    positive_labels: list[str]
    negative_labels: list[str]
    dropped_rows_missing_target: int = 0
    dropped_rows_unmapped_label: int = 0
    unmapped_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target_column": self.target_column,
            "positive_labels": self.positive_labels,
            "negative_labels": self.negative_labels,
            "positive_class_meaning": "default",
            "dropped_rows_missing_target": self.dropped_rows_missing_target,
            "dropped_rows_unmapped_label": self.dropped_rows_unmapped_label,
            "unmapped_labels": self.unmapped_labels,
        }


def suggest_target_columns(df: pd.DataFrame) -> list[str]:
    lower = {c.lower(): c for c in df.columns}
    found = [lower[n] for n in TARGET_NAME_CANDIDATES if n in lower]
    # Also suggest any low-cardinality column whose labels look like default flags.
    for col in df.columns:
        if col in found:
            continue
        vals = df[col].dropna()
        if 1 < vals.nunique() <= 3:
            labels = {_norm_label(v) for v in vals.unique()}
            if labels & KNOWN_DEFAULT_LABELS and labels <= (KNOWN_DEFAULT_LABELS | KNOWN_NON_DEFAULT_LABELS):
                found.append(col)
    return found


def resolve_target(
    df: pd.DataFrame,
    target_column: str,
    positive_labels: Iterable[Any] | None = None,
    negative_labels: Iterable[Any] | None = None,
) -> tuple[pd.Series, pd.Index, TargetMapping]:
    """Map the raw target column to a binary 0/1 series (1 = default).

    Returns (y, kept_index, mapping). Rows with a missing target or a label
    that is in neither class (when negative_labels is given) are dropped.
    """
    if target_column not in df.columns:
        raise DataValidationError(
            f"Target column '{target_column}' not found in dataset.",
            {"available_columns": list(df.columns), "suggested_targets": suggest_target_columns(df)},
        )
    raw = df[target_column]
    missing_mask = raw.isna() | raw.astype(str).str.strip().eq("")
    observed = raw[~missing_mask]
    if observed.empty:
        raise DataValidationError(f"Target column '{target_column}' has no non-missing values.")
    norm = observed.map(_norm_label)
    labels = sorted(norm.unique())

    if positive_labels is not None:
        pos = {_norm_label(v) for v in positive_labels}
        if not pos:
            raise DataValidationError("positive_labels must not be empty.")
    else:
        pos = set(labels) & KNOWN_DEFAULT_LABELS
        if not pos or len(pos) == len(labels):
            raise DataValidationError(
                "Could not infer which target labels mean 'default'. Provide positive_labels.",
                {"target_column": target_column, "observed_labels": labels[:50]},
            )

    unknown_pos = pos - set(labels)
    if unknown_pos == pos:
        raise DataValidationError(
            "None of the positive_labels occur in the target column.",
            {"positive_labels": sorted(pos), "observed_labels": labels[:50]},
        )

    if negative_labels is not None:
        neg = {_norm_label(v) for v in negative_labels}
        overlap = pos & neg
        if overlap:
            raise DataValidationError("Labels cannot be both positive and negative.", {"labels": sorted(overlap)})
    else:
        neg = set(labels) - pos
        if len(labels) > 2 and len(neg) > 1:
            raise DataValidationError(
                "Target has more than two labels. Provide negative_labels so ambiguous outcomes "
                "(e.g. loans still 'Current') are not silently treated as non-default.",
                {"observed_labels": labels[:50], "positive_labels": sorted(pos)},
            )

    keep = norm.isin(pos | neg)
    unmapped = sorted(set(norm[~keep].unique()))
    y = norm[keep].isin(pos).astype(int)
    mapping = TargetMapping(
        target_column=target_column,
        positive_labels=sorted(pos & set(labels)),
        negative_labels=sorted(neg & set(labels)),
        dropped_rows_missing_target=int(missing_mask.sum()),
        dropped_rows_unmapped_label=int((~keep).sum()),
        unmapped_labels=unmapped[:50],
    )
    return y, y.index, mapping


# --------------------------------------------------------------------------- #
# Feature typing
# --------------------------------------------------------------------------- #
def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = s.astype("string").str.replace(NUMERIC_CLEAN_RE, "", regex=True)
    cleaned = cleaned.mask(cleaned.isin(["", "nan", "NaN", "None", "null", "NA", "<NA>"]))
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def _looks_numeric(s: pd.Series, threshold: float = 0.95) -> bool:
    non_null = s.dropna()
    if non_null.empty:
        return False
    parsed = _coerce_numeric_series(non_null)
    return parsed.notna().mean() >= threshold


@dataclass
class FeatureTypes:
    numeric: list[str]
    categorical: list[str]
    excluded: dict[str, str]  # column -> reason
    dtype_issues: list[dict]


def detect_leakage_by_name(columns: Iterable[str], target_column: str | None) -> list[str]:
    out = []
    for col in columns:
        if col == target_column:
            continue
        name = col.lower()
        if any(p in name for p in LEAKAGE_NAME_PATTERNS):
            out.append(col)
    return out


def infer_feature_types(
    df: pd.DataFrame,
    target_column: str | None,
    exclude_columns: Iterable[str] = (),
    auto_exclude_leakage: bool = True,
) -> FeatureTypes:
    excluded: dict[str, str] = {}
    for col in exclude_columns:
        if col in df.columns and col != target_column:
            excluded[col] = "excluded by user"
    if auto_exclude_leakage:
        for col in detect_leakage_by_name(df.columns, target_column):
            excluded.setdefault(col, "suspected post-outcome field (target leakage)")

    numeric, categorical, dtype_issues = [], [], []
    n = len(df)
    for col in df.columns:
        if col == target_column or col in excluded:
            continue
        s = df[col]
        if s.isna().all():
            excluded[col] = "all values missing"
            continue
        if s.nunique(dropna=True) <= 1:
            excluded[col] = "constant column"
            continue
        if ID_NAME_RE.search(col):
            excluded[col] = "identifier column"
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            excluded[col] = "datetime column (derive explicit features before upload)"
            continue
        if pd.api.types.is_bool_dtype(s):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        elif _looks_numeric(s):
            numeric.append(col)
            dtype_issues.append({
                "column": col,
                "issue": "numeric values stored as text; coerced to numbers",
                "unparseable_values": int(_coerce_numeric_series(s.dropna()).isna().sum()),
            })
        else:
            nunique = s.nunique(dropna=True)
            if n >= 20 and nunique / max(s.notna().sum(), 1) > 0.9:
                excluded[col] = "high-cardinality text (likely identifier or free text)"
                continue
            if s.dropna().map(type).nunique() > 1:
                dtype_issues.append({"column": col, "issue": "mixed value types; treated as categorical text"})
            categorical.append(col)
    return FeatureTypes(numeric=numeric, categorical=categorical, excluded=excluded, dtype_issues=dtype_issues)


def detect_statistical_leakage(X: pd.DataFrame, y: pd.Series, numeric: list[str], threshold: float = 0.98) -> list[dict]:
    """Flag numeric features that almost perfectly separate the classes on their own."""
    flagged = []
    if y.nunique() < 2:
        return flagged
    for col in numeric:
        vals = _coerce_numeric_series(X[col])
        mask = vals.notna()
        if mask.sum() < 20 or y[mask].nunique() < 2:
            continue
        auc = roc_auc_score(y[mask], vals[mask])
        auc = max(auc, 1 - auc)
        if auc >= threshold:
            flagged.append({"column": col, "single_feature_auc": round(float(auc), 4)})
    return flagged


# --------------------------------------------------------------------------- #
# Stateless coercion step used *inside* the saved sklearn pipeline
# --------------------------------------------------------------------------- #
class FeatureCoercer(BaseEstimator, TransformerMixin):
    """Select model features and coerce types consistently at train and predict time.

    Stateless: `fit` learns nothing from the data, so it cannot leak information.
    """

    def __init__(self, numeric_features: list[str], categorical_features: list[str]):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def fit(self, X, y=None):  # noqa: D401 - sklearn API
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.numeric_features + self.categorical_features:
            if col not in X.columns:
                X[col] = np.nan
        out = pd.DataFrame(index=X.index)
        for col in self.numeric_features:
            out[col] = _coerce_numeric_series(X[col]).replace([np.inf, -np.inf], np.nan)
        for col in self.categorical_features:
            s = X[col]
            as_str = s.map(lambda v: _norm_category(v))
            out[col] = as_str.astype(object)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.numeric_features + self.categorical_features, dtype=object)


def _norm_category(v: Any):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    s = str(v).strip()
    return np.nan if s == "" else s


# --------------------------------------------------------------------------- #
# Data-quality report
# --------------------------------------------------------------------------- #
def _safe(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 6)
    if isinstance(v, float):
        return None if np.isnan(v) else round(v, 6)
    return v


def build_quality_report(
    df: pd.DataFrame,
    target_column: str | None = None,
    positive_labels: list[Any] | None = None,
    negative_labels: list[Any] | None = None,
    exclude_columns: Iterable[str] = (),
) -> dict:
    """Dataset summary + data-quality diagnostics. Does not modify `df`."""
    warnings: list[str] = []
    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    missing_info = {
        c: {"count": int(missing[c]), "percent": round(100 * missing[c] / n_rows, 2)}
        for c in df.columns if missing[c] > 0
    }
    dup_count = int(df.duplicated().sum())
    if dup_count:
        warnings.append(f"{dup_count} exact duplicate rows found; they are removed before training.")
    high_missing = [c for c, m in missing_info.items() if m["percent"] > 40]
    if high_missing:
        warnings.append(f"Columns with >40% missing values: {high_missing}.")

    types = infer_feature_types(df, target_column, exclude_columns)

    numeric_stats = {}
    for col in types.numeric:
        s = _coerce_numeric_series(df[col])
        desc = s.describe()
        numeric_stats[col] = {k: _safe(desc.get(k)) for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]}
    categorical_stats = {}
    for col in types.categorical:
        vc = df[col].astype("string").value_counts(dropna=True)
        categorical_stats[col] = {
            "unique": int(vc.size),
            "top_values": {str(k): int(v) for k, v in vc.head(10).items()},
        }

    report: dict[str, Any] = {
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        "missing_values": missing_info,
        "total_missing_cells": int(missing.sum()),
        "duplicate_rows": dup_count,
        "dtype_issues": types.dtype_issues,
        "numeric_features": types.numeric,
        "categorical_features": types.categorical,
        "excluded_columns": types.excluded,
        "numeric_summary": numeric_stats,
        "categorical_summary": categorical_stats,
        "suggested_target_columns": suggest_target_columns(df),
        "target": None,
        "leakage_warnings": [
            {"column": c, "reason": r} for c, r in types.excluded.items() if "leakage" in r
        ],
        "warnings": warnings,
    }

    if target_column:
        try:
            y, idx, mapping = resolve_target(df, target_column, positive_labels, negative_labels)
        except DataValidationError as exc:
            report["target"] = {"target_column": target_column, "error": exc.message, "details": exc.details}
            warnings.append(f"Target column problem: {exc.message}")
        else:
            counts = y.value_counts().to_dict()
            n_pos, n_neg = int(counts.get(1, 0)), int(counts.get(0, 0))
            minority = min(n_pos, n_neg)
            report["target"] = {
                **mapping.to_dict(),
                "class_counts": {"default": n_pos, "non_default": n_neg},
                "default_rate": round(n_pos / max(len(y), 1), 4),
                "imbalance_ratio": round(max(n_pos, n_neg) / minority, 3) if minority else None,
                "is_imbalanced": bool(minority == 0 or minority / len(y) < 0.3),
            }
            if minority == 0:
                warnings.append("Only one target class present; a classifier cannot be trained.")
            elif minority / len(y) < 0.3:
                warnings.append("Default class is imbalanced; training uses balanced class weights.")
            stat_leaks = detect_statistical_leakage(df.loc[idx], y, types.numeric)
            for item in stat_leaks:
                report["leakage_warnings"].append({
                    "column": item["column"],
                    "reason": f"single feature separates classes almost perfectly (AUC {item['single_feature_auc']}); "
                              "verify it is known at loan-approval time",
                })
    return report


# --------------------------------------------------------------------------- #
# Training-data preparation
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]
    excluded_columns: dict[str, str]
    target_mapping: TargetMapping
    duplicates_removed: int
    statistical_leakage: list[dict]


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str,
    positive_labels: list[Any] | None = None,
    negative_labels: list[Any] | None = None,
    exclude_columns: Iterable[str] = (),
    auto_exclude_leakage: bool = True,
) -> PreparedData:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before - len(df)

    y, idx, mapping = resolve_target(df, target_column, positive_labels, negative_labels)
    df = df.loc[idx]
    types = infer_feature_types(df, target_column, exclude_columns, auto_exclude_leakage)
    features = types.numeric + types.categorical
    if not features:
        raise DataValidationError("No usable feature columns remain after exclusions.", {"excluded": types.excluded})

    counts = y.value_counts()
    if len(y) < MIN_TRAINING_ROWS:
        raise InsufficientDataError(
            f"At least {MIN_TRAINING_ROWS} labelled rows are required; found {len(y)}."
        )
    if y.nunique() < 2:
        raise InsufficientDataError(
            "Target contains only one class after mapping; both default and non-default examples are required.",
            {"class_counts": {int(k): int(v) for k, v in counts.items()}},
        )
    if counts.min() < MIN_ROWS_PER_CLASS:
        raise InsufficientDataError(
            f"Each class needs at least {MIN_ROWS_PER_CLASS} rows for a stratified train/validation/test split.",
            {"class_counts": {"default": int(counts.get(1, 0)), "non_default": int(counts.get(0, 0))}},
        )

    X = df[features].copy()
    return PreparedData(
        X=X,
        y=y.astype(int),
        numeric_features=types.numeric,
        categorical_features=types.categorical,
        excluded_columns=types.excluded,
        target_mapping=mapping,
        duplicates_removed=duplicates_removed,
        statistical_leakage=detect_statistical_leakage(X, y, types.numeric),
    )
