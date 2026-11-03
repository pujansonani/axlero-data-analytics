"""Train, compare and persist credit-risk classifiers.

Workflow
--------
1. Prepare data (dedupe, map target, type features, drop leakage columns).
2. Stratified split into train / validation / test.
3. For each model: fit preprocessing + classifier on *train only*.
4. Use *validation* to tune the decision threshold and select the best model.
5. Report final metrics on the untouched *test* split.
6. Persist the selected pipeline and metadata with joblib.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.config import DEFAULT_RISK_BANDS
from app.services.data_processing import FeatureCoercer, PreparedData, prepare_training_data
from app.services.errors import DataValidationError, InsufficientDataError
from app.services.evaluation import evaluate_predictions, tune_threshold

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("logistic_regression", "random_forest", "decision_tree")
SELECTION_METRICS = ("roc_auc", "average_precision", "f1", "recall")
ARTIFACT_VERSION = 1
CATEGORY_SEP = "=="


def ohe_name_combiner(feature: str, category: Any) -> str:
    """Readable, unambiguous one-hot column names: '<feature>==<category>'."""
    return f"{feature}{CATEGORY_SEP}{category}"


@dataclass
class TrainingConfig:
    target_column: str
    positive_labels: list[Any] | None = None
    negative_labels: list[Any] | None = None
    exclude_columns: list[str] = field(default_factory=list)
    auto_exclude_leakage: bool = True
    models: list[str] = field(default_factory=lambda: list(SUPPORTED_MODELS))
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    selection_metric: str = "roc_auc"
    min_default_recall: float | None = None
    risk_bands: list[dict] = field(default_factory=lambda: [dict(b) for b in DEFAULT_RISK_BANDS])

    def validate(self) -> None:
        unknown = [m for m in self.models if m not in SUPPORTED_MODELS]
        if unknown or not self.models:
            raise DataValidationError("Unsupported or empty model list.", {"unknown": unknown, "supported": list(SUPPORTED_MODELS)})
        if self.selection_metric not in SELECTION_METRICS:
            raise DataValidationError("Unsupported selection_metric.", {"supported": list(SELECTION_METRICS)})
        if not (0.05 <= self.test_size <= 0.5 and 0.05 <= self.validation_size <= 0.5):
            raise DataValidationError("test_size and validation_size must each be between 0.05 and 0.5.")
        if self.test_size + self.validation_size >= 0.8:
            raise DataValidationError("test_size + validation_size must leave at least 20% for training.")
        validate_risk_bands(self.risk_bands)


def validate_risk_bands(bands: list[dict]) -> None:
    if not bands:
        raise DataValidationError("risk_bands must contain at least one band.")
    prev = 0.0
    names = set()
    for b in bands:
        name, upper = b.get("name"), b.get("max_probability")
        if not name or upper is None:
            raise DataValidationError("Each risk band needs 'name' and 'max_probability'.")
        if name in names:
            raise DataValidationError(f"Duplicate risk band name '{name}'.")
        names.add(name)
        if not (prev < float(upper) <= 1.0):
            raise DataValidationError("risk_bands max_probability values must be strictly increasing within (0, 1].")
        prev = float(upper)
    if abs(prev - 1.0) > 1e-9:
        raise DataValidationError("The last risk band must have max_probability = 1.0.")


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #
def build_preprocessor(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
                ("onehot", OneHotEncoder(
                    handle_unknown="infrequent_if_exist",  # unseen categories -> infrequent bucket or all-zeros
                    min_frequency=5,
                    max_categories=30,
                    sparse_output=False,
                    feature_name_combiner=ohe_name_combiner,
                )),
            ]),
            categorical,
        ))
    return Pipeline([
        ("coerce", FeatureCoercer(numeric, categorical)),
        ("columns", ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)),
    ])


def build_classifier(name: str, n_train: int, random_state: int):
    leaf = int(np.clip(n_train // 100, 2, 20))
    if name == "logistic_regression":
        return LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0, solver="lbfgs")
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=leaf, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=random_state,
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=leaf, class_weight="balanced", random_state=random_state)
    raise ValueError(name)


def transformed_feature_map(pipeline: Pipeline) -> tuple[list[str], list[str]]:
    """Return (transformed_feature_names, original_feature_for_each)."""
    names = list(pipeline.named_steps["preprocessor"].named_steps["columns"].get_feature_names_out())
    originals = []
    for n in names:
        block, _, rest = n.partition("__")
        if block == "num":
            originals.append(rest[len("missingindicator_"):] if rest.startswith("missingindicator_") else rest)
        else:
            originals.append(rest.split(CATEGORY_SEP, 1)[0])
    return names, originals


def build_feature_schema(X_train: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    coerced = FeatureCoercer(numeric, categorical).transform(X_train)
    schema = {"numeric": {}, "categorical": {}}
    for col in numeric:
        s = coerced[col]
        schema["numeric"][col] = {
            "min": None if s.isna().all() else float(s.min()),
            "max": None if s.isna().all() else float(s.max()),
            "median": None if s.isna().all() else float(s.median()),
            "missing_in_training": int(s.isna().sum()),
        }
    for col in categorical:
        vc = coerced[col].value_counts(dropna=True)
        schema["categorical"][col] = {"categories": [str(v) for v in vc.index[:200]], "missing_in_training": int(coerced[col].isna().sum())}
    return schema


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _split(prepared: PreparedData, cfg: TrainingConfig):
    X, y = prepared.X, prepared.y
    try:
        X_rest, X_test, y_rest, y_test = train_test_split(
            X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
        )
        rel_val = cfg.validation_size / (1 - cfg.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest, test_size=rel_val, stratify=y_rest, random_state=cfg.random_state
        )
    except ValueError as exc:
        raise InsufficientDataError(f"Could not create a stratified split: {exc}") from exc
    for name, part in (("train", y_train), ("validation", y_val), ("test", y_test)):
        if part.nunique() < 2:
            raise InsufficientDataError(f"The {name} split contains only one class; provide more default examples.")
    return X_train, X_val, X_test, y_train, y_val, y_test


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def train_models(df: pd.DataFrame, cfg: TrainingConfig, dataset_meta: dict | None = None) -> dict:
    """Train all requested models, select the best and return the artifact dict."""
    cfg.validate()
    prepared = prepare_training_data(
        df, cfg.target_column, cfg.positive_labels, cfg.negative_labels,
        cfg.exclude_columns, cfg.auto_exclude_leakage,
    )
    X_train, X_val, X_test, y_train, y_val, y_test = _split(prepared, cfg)

    results: dict[str, dict] = {}
    pipelines: dict[str, Pipeline] = {}
    for name in cfg.models:
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(prepared.numeric_features, prepared.categorical_features)),
            ("classifier", build_classifier(name, len(X_train), cfg.random_state)),
        ])
        pipe.fit(X_train, y_train)  # preprocessing statistics come from the training split only
        p_train = pipe.predict_proba(X_train)[:, 1]
        p_val = pipe.predict_proba(X_val)[:, 1]
        p_test = pipe.predict_proba(X_test)[:, 1]
        tuning = tune_threshold(y_val, p_val, cfg.min_default_recall)
        thr = float(tuning["threshold"])
        results[name] = {
            "model": name,
            "threshold_tuning": tuning,
            "train": {k: v for k, v in evaluate_predictions(y_train, p_train, thr).items() if k in ("roc_auc", "f1", "recall", "precision")},
            "validation": evaluate_predictions(y_val, p_val, thr),
            "test": evaluate_predictions(y_test, p_test, thr),
            "test_at_default_threshold_0_5": evaluate_predictions(y_test, p_test, 0.5),
        }
        pipelines[name] = pipe
        logger.info("Trained %s: val %s=%s", name, cfg.selection_metric, results[name]["validation"].get(cfg.selection_metric))

    def score(name: str) -> float:
        v = results[name]["validation"].get(cfg.selection_metric)
        return -1.0 if v is None else float(v)

    best = max(cfg.models, key=score)
    ranking = sorted(cfg.models, key=score, reverse=True)
    comparison = [
        {
            "model": m,
            "rank": i + 1,
            "selected": m == best,
            "threshold": results[m]["threshold_tuning"]["threshold"],
            "validation_" + cfg.selection_metric: results[m]["validation"].get(cfg.selection_metric),
            "test_precision": results[m]["test"]["precision"],
            "test_recall": results[m]["test"]["recall"],
            "test_f1": results[m]["test"]["f1"],
            "test_roc_auc": results[m]["test"]["roc_auc"],
            "test_average_precision": results[m]["test"]["average_precision"],
            "train_roc_auc": results[m]["train"]["roc_auc"],
        }
        for i, m in enumerate(ranking)
    ]

    best_pipe = pipelines[best]
    transformed_names, original_map = transformed_feature_map(best_pipe)
    dataset_meta = dict(dataset_meta or {})
    trained_at = datetime.now(timezone.utc).isoformat()

    metrics = {
        "trained_at": trained_at,
        "selected_model": best,
        "selection_metric": cfg.selection_metric,
        "selection_basis": "validation split",
        "reported_metrics_basis": "held-out test split (not used for training, tuning or selection)",
        "decision_threshold": results[best]["threshold_tuning"]["threshold"],
        "split": {
            "strategy": "stratified",
            "random_state": cfg.random_state,
            "train_rows": int(len(y_train)),
            "validation_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "train_default_rate": round(float(y_train.mean()), 4),
            "validation_default_rate": round(float(y_val.mean()), 4),
            "test_default_rate": round(float(y_test.mean()), 4),
        },
        "class_imbalance_handling": "class_weight='balanced' (random forest: 'balanced_subsample')",
        "comparison": comparison,
        "models": results,
        "data": {
            "is_synthetic": bool(dataset_meta.get("is_synthetic", False)),
            "filename": dataset_meta.get("filename"),
            "rows_used": int(len(prepared.y)),
            "duplicates_removed": prepared.duplicates_removed,
            "target": prepared.target_mapping.to_dict(),
            "numeric_features": prepared.numeric_features,
            "categorical_features": prepared.categorical_features,
            "excluded_columns": prepared.excluded_columns,
            "statistical_leakage_warnings": prepared.statistical_leakage,
        },
        "notes": [
            "Selected model is fitted on the training split only; it is not refitted on validation data, "
            "so the tuned threshold stays consistent with the saved model.",
        ],
    }
    if dataset_meta.get("is_synthetic"):
        metrics["notes"].append(
            "Trained on SYNTHETIC development data. These metrics do not indicate real-world performance."
        )

    # Background sample (raw features) for SHAP; drawn from training data only.
    bg = X_train.sample(n=min(200, len(X_train)), random_state=cfg.random_state)
    sample_for_global = X_test.sample(n=min(500, len(X_test)), random_state=cfg.random_state)

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "pipeline": best_pipe,
        "model_name": best,
        "threshold": float(results[best]["threshold_tuning"]["threshold"]),
        "numeric_features": prepared.numeric_features,
        "categorical_features": prepared.categorical_features,
        "feature_schema": build_feature_schema(X_train, prepared.numeric_features, prepared.categorical_features),
        "transformed_features": transformed_names,
        "transformed_to_original": original_map,
        "target_mapping": prepared.target_mapping.to_dict(),
        "excluded_columns": prepared.excluded_columns,
        "risk_bands": cfg.risk_bands,
        "metrics": metrics,
        "shap_background": bg,
        "shap_global_sample": sample_for_global,
        "trained_at": trained_at,
        "dataset": {**dataset_meta, "fingerprint": _dataset_fingerprint(df)},
        "library_versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    return artifact


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_artifact(artifact: dict, model_path: Path, metrics_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_suffix(".tmp")
    # Runtime caches (SHAP explainer etc.) are rebuilt on load and never persisted.
    joblib.dump({k: v for k, v in artifact.items() if not k.startswith("_")}, tmp, compress=3)
    tmp.replace(model_path)  # atomic swap so readers never see a half-written file
    metrics_path.write_text(json.dumps(artifact["metrics"], indent=2, default=str), encoding="utf-8")


def load_artifact(model_path: Path) -> dict | None:
    if not model_path.exists():
        return None
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or artifact.get("artifact_version") != ARTIFACT_VERSION:
        logger.warning("Ignoring incompatible model artifact at %s", model_path)
        return None
    return artifact
