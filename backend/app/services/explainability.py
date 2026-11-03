"""SHAP explanations for the selected credit-risk pipeline.

- Tree models (random forest, decision tree): shap.TreeExplainer. Output space
  is the predicted probability of default.
- Logistic regression: shap.LinearExplainer with a training-data background.
  Output space is log-odds of default.

One-hot / missing-indicator columns are aggregated back to the original
borrower fields so explanations refer to inputs the user actually provided.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
import pandas as pd

from app.config import EXPLANATION_DISCLAIMER

logger = logging.getLogger(__name__)
_lock = threading.Lock()

TREE_MODELS = {"random_forest", "decision_tree"}
LINEAR_MODELS = {"logistic_regression"}


def _shap():
    import shap  # imported lazily: shap/numba import is slow

    return shap


def _preprocess(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    pre = artifact["pipeline"].named_steps["preprocessor"]
    return np.asarray(pre.transform(X), dtype=float)


def _coerced(artifact: dict, X: pd.DataFrame) -> pd.DataFrame:
    return artifact["pipeline"].named_steps["preprocessor"].named_steps["coerce"].transform(X)


def get_explainer(artifact: dict):
    """Build (once) and cache the SHAP explainer in the artifact's runtime slot."""
    runtime = artifact.setdefault("_runtime", {})
    if "explainer" in runtime:
        return runtime["explainer"]
    with _lock:
        if "explainer" in runtime:
            return runtime["explainer"]
        shap = _shap()
        clf = artifact["pipeline"].named_steps["classifier"]
        name = artifact["model_name"]
        if name in TREE_MODELS:
            explainer = shap.TreeExplainer(clf)
            space = "probability"
        elif name in LINEAR_MODELS:
            background = _preprocess(artifact, artifact["shap_background"])
            explainer = shap.LinearExplainer(clf, shap.maskers.Independent(background, max_samples=200))
            space = "log_odds"
        else:  # pragma: no cover - guarded by SUPPORTED_MODELS
            raise ValueError(f"No SHAP explainer configured for model '{name}'")
        runtime["explainer"] = (explainer, space)
        return runtime["explainer"]


def _positive_class_shap(explainer, Xt: np.ndarray) -> tuple[np.ndarray, float]:
    values = explainer.shap_values(Xt)
    base = explainer.expected_value
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:  # (n_samples, n_features, n_classes)
        values = values[:, :, 1]
    base_arr = np.atleast_1d(np.asarray(base, dtype=float))
    base_val = float(base_arr[1] if base_arr.size > 1 else base_arr[0])
    return values, base_val


def _aggregate(values: np.ndarray, originals: list[str], feature_order: list[str]) -> np.ndarray:
    """Sum transformed-column SHAP values into original-feature SHAP values."""
    idx = {f: i for i, f in enumerate(feature_order)}
    out = np.zeros((values.shape[0], len(feature_order)))
    for j, orig in enumerate(originals):
        out[:, idx[orig]] += values[:, j]
    return out


def _display_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def compute_shap(artifact: dict, X: pd.DataFrame) -> dict:
    explainer, space = get_explainer(artifact)
    Xt = _preprocess(artifact, X)
    values, base = _positive_class_shap(explainer, Xt)
    features = artifact["numeric_features"] + artifact["categorical_features"]
    agg = _aggregate(values, artifact["transformed_to_original"], features)
    return {"values": agg, "transformed_values": values, "base_value": base, "output_space": space, "features": features}


# --------------------------------------------------------------------------- #
# Global explanations
# --------------------------------------------------------------------------- #
def _model_native_importance(artifact: dict) -> list[dict]:
    clf = artifact["pipeline"].named_steps["classifier"]
    originals = artifact["transformed_to_original"]
    if hasattr(clf, "feature_importances_"):
        raw = np.asarray(clf.feature_importances_, dtype=float)
        kind = "impurity_importance"
    elif hasattr(clf, "coef_"):
        raw = np.abs(np.asarray(clf.coef_, dtype=float).ravel())
        kind = "abs_standardized_coefficient"
    else:
        return []
    totals: dict[str, float] = {}
    for v, o in zip(raw, originals):
        totals[o] = totals.get(o, 0.0) + float(v)
    return [{"feature": f, "importance": round(v, 6), "type": kind} for f, v in sorted(totals.items(), key=lambda kv: -kv[1])]


def global_explanation(artifact: dict, max_points_per_feature: int = 200, top_n: int = 15) -> dict:
    runtime = artifact.setdefault("_runtime", {})
    if "global" in runtime:
        return runtime["global"]
    sample = artifact["shap_global_sample"]
    res = compute_shap(artifact, sample)
    vals = res["values"]
    features = res["features"]
    mean_abs = np.abs(vals).mean(axis=0)
    total = float(mean_abs.sum()) or 1.0
    order = np.argsort(-mean_abs)
    coerced = _coerced(artifact, sample)

    importance = [
        {
            "feature": features[i],
            "feature_type": "numeric" if features[i] in artifact["numeric_features"] else "categorical",
            "mean_abs_shap": round(float(mean_abs[i]), 6),
            "mean_shap": round(float(vals[:, i].mean()), 6),
            "share_of_total": round(float(mean_abs[i]) / total, 4),
            "rank": r + 1,
        }
        for r, i in enumerate(order)
    ]
    rng = np.random.default_rng(0)
    n = len(sample)
    pick = rng.choice(n, size=min(n, max_points_per_feature), replace=False)
    summary = []
    for i in order[:top_n]:
        f = features[i]
        summary.append({
            "feature": f,
            "points": [
                {"feature_value": _display_value(coerced[f].iloc[k]), "shap_value": round(float(vals[k, i]), 6)}
                for k in pick
            ],
        })
    result = {
        "model": artifact["model_name"],
        "output_space": res["output_space"],
        "base_value": round(res["base_value"], 6),
        "n_samples_explained": int(n),
        "sample_source": "held-out test split",
        "feature_importance": importance,
        "summary_plot_data": summary,
        "model_native_importance": _model_native_importance(artifact),
        "disclaimer": EXPLANATION_DISCLAIMER,
    }
    runtime["global"] = result
    return result


# --------------------------------------------------------------------------- #
# Local explanations
# --------------------------------------------------------------------------- #
def local_explanations(artifact: dict, X: pd.DataFrame, top_k: int = 5) -> list[dict]:
    res = compute_shap(artifact, X)
    vals, features, base = res["values"], res["features"], res["base_value"]
    coerced = _coerced(artifact, X)
    proba = artifact["pipeline"].predict_proba(X)[:, 1]
    out = []
    for r in range(len(X)):
        contribs = [
            {
                "feature": f,
                "value": _display_value(coerced[f].iloc[r]),
                "shap_value": round(float(vals[r, j]), 6),
                "effect": "increases_risk" if vals[r, j] > 0 else ("decreases_risk" if vals[r, j] < 0 else "neutral"),
            }
            for j, f in enumerate(features)
        ]
        contribs.sort(key=lambda c: -abs(c["shap_value"]))
        model_output = base + float(vals[r].sum())
        out.append({
            "output_space": res["output_space"],
            "base_value": round(base, 6),
            "model_output": round(model_output, 6),
            "predicted_default_probability": round(float(proba[r]), 6),
            "top_risk_increasing_factors": [c for c in contribs if c["effect"] == "increases_risk"][:top_k],
            "top_risk_decreasing_factors": [c for c in contribs if c["effect"] == "decreases_risk"][:top_k],
            "contributions": contribs,
        })
    return out
