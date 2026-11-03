"""Model evaluation and decision-threshold tuning.

Every number here is computed from actual model predictions on held-out data.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_NAMES = ["non_default", "default"]


def _r(x: float | None, nd: int = 4) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, nd)


def evaluate_predictions(y_true, proba, threshold: float = 0.5) -> dict[str, Any]:
    """Compute classification metrics for the default (positive=1) class."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    report = {k: ({m: _r(v) for m, v in d.items()} if isinstance(d, dict) else _r(d)) for k, d in report.items()}

    return {
        "threshold": _r(threshold),
        "n_samples": int(len(y_true)),
        "n_default": int(y_true.sum()),
        "precision": _r(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _r(recall_score(y_true, y_pred, zero_division=0)),
        "default_class_recall": _r(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _r(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": _r(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _r(balanced_accuracy_score(y_true, y_pred)) if both_classes else None,
        "roc_auc": _r(roc_auc_score(y_true, proba)) if both_classes else None,
        "average_precision": _r(average_precision_score(y_true, proba)) if both_classes else None,
        "brier_score": _r(brier_score_loss(y_true, proba)),
        "confusion_matrix": {
            "labels": CLASS_NAMES,
            "matrix": cm.tolist(),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "classification_report": report,
    }


def tune_threshold(y_true, proba, min_recall: float | None = None) -> dict[str, Any]:
    """Pick a decision threshold on *validation* data.

    Default strategy maximises default-class F1. If `min_recall` is set, the
    threshold with the best precision among those reaching that recall is used.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall have one more element than thresholds; drop the last point.
    precision, recall = precision[:-1], recall[:-1]
    if thresholds.size == 0:
        return {"threshold": 0.5, "strategy": "default_0.5", "note": "validation data could not support tuning"}

    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
    if min_recall is not None:
        ok = recall >= min_recall
        if ok.any():
            idx = int(np.flatnonzero(ok)[np.argmax(precision[ok])])
            strategy = f"max_precision_at_recall>={min_recall}"
        else:
            idx = int(np.argmax(recall))
            strategy = f"max_recall (target recall {min_recall} not reachable)"
    else:
        idx = int(np.argmax(f1))
        strategy = "max_f1"
    return {
        "threshold": _r(float(thresholds[idx]), 6),
        "strategy": strategy,
        "validation_precision": _r(precision[idx]),
        "validation_recall": _r(recall[idx]),
        "validation_f1": _r(f1[idx]),
    }
