"""Credit-risk prediction service: input validation, scoring and risk bands."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app.config import PREDICTION_DISCLAIMER
from app.services.errors import InvalidInputError
from app.services.explainability import local_explanations

MAX_RECORDS = 1000


def assign_risk_band(probability: float, bands: list[dict]) -> str:
    """Bands are ordered by `max_probability`; a probability below a band's bound falls in that band."""
    for band in bands[:-1]:
        if probability < float(band["max_probability"]):
            return band["name"]
    return bands[-1]["name"]


def _parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid number")
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        s = value.strip().replace(",", "")
        if s == "":
            return None
        f = float(s)
    else:
        raise ValueError(f"unsupported type {type(value).__name__}")
    if not math.isfinite(f):
        raise ValueError("value must be finite")
    return f


def validate_records(records: list[dict], artifact: dict) -> tuple[pd.DataFrame, list[list[str]]]:
    """Validate borrower records against the trained feature schema.

    Every model feature must be present as a key (use null for unknown values,
    which the pipeline imputes). Returns the clean DataFrame and per-record warnings.
    """
    if not records:
        raise InvalidInputError("At least one borrower record is required.")
    if len(records) > MAX_RECORDS:
        raise InvalidInputError(f"At most {MAX_RECORDS} records per request.")

    numeric = artifact["numeric_features"]
    categorical = artifact["categorical_features"]
    schema = artifact["feature_schema"]
    expected = set(numeric) | set(categorical)
    target = artifact["target_mapping"]["target_column"]

    errors: list[dict] = []
    warnings: list[list[str]] = []
    rows: list[dict] = []
    for i, rec in enumerate(records):
        w: list[str] = []
        if not isinstance(rec, dict):
            errors.append({"record": i, "error": "record must be a JSON object"})
            warnings.append(w)
            continue
        missing = sorted(expected - set(rec))
        if missing:
            errors.append({"record": i, "field": missing, "error": "missing required fields (send null if unknown)"})
        extra = sorted(set(rec) - expected)
        if target in extra:
            w.append(f"Field '{target}' is the training target and was ignored.")
        ignored = [e for e in extra if e != target]
        if ignored:
            w.append(f"Ignored fields not used by the model: {ignored}.")

        row: dict[str, Any] = {}
        for col in numeric:
            if col not in rec:
                continue
            try:
                val = _parse_number(rec[col])
            except (ValueError, TypeError) as exc:
                errors.append({"record": i, "field": col, "error": f"expected a number ({exc})", "value": repr(rec[col])[:100]})
                continue
            row[col] = val
            if val is None:
                w.append(f"'{col}' is missing; imputed with the training median.")
            else:
                info = schema["numeric"].get(col, {})
                lo, hi = info.get("min"), info.get("max")
                if lo is not None and hi is not None and not (lo <= val <= hi):
                    w.append(f"'{col}'={val} is outside the training range [{lo}, {hi}]; prediction is less reliable.")
        for col in categorical:
            if col not in rec:
                continue
            val = rec[col]
            if val is not None and not isinstance(val, (str, int, float, bool)):
                errors.append({"record": i, "field": col, "error": "expected a string/number/boolean category"})
                continue
            if isinstance(val, str) and val.strip() == "":
                val = None
            row[col] = val
            if val is None:
                w.append(f"'{col}' is missing; treated as an explicit 'missing' category.")
            else:
                cats = schema["categorical"].get(col, {}).get("categories", [])
                norm = ("True" if val else "False") if isinstance(val, bool) else str(val).strip()
                if cats and norm not in cats:
                    w.append(f"'{col}'='{norm}' was not seen during training; handled as an unknown/infrequent category.")
        rows.append(row)
        warnings.append(w)

    if errors:
        raise InvalidInputError("Borrower data failed validation.", errors[:200])
    return pd.DataFrame(rows, columns=numeric + categorical), warnings


def predict(artifact: dict, records: list[dict], include_explanation: bool = True, top_k: int = 5) -> dict:
    X, warnings = validate_records(records, artifact)
    proba = artifact["pipeline"].predict_proba(X)[:, 1]
    threshold = float(artifact["threshold"])
    bands = artifact["risk_bands"]

    explanations: list[dict | None] = [None] * len(X)
    explanation_error = None
    if include_explanation:
        try:
            explanations = local_explanations(artifact, X, top_k=top_k)
        except Exception as exc:  # explanation failure must not block scoring
            explanation_error = f"Explanation unavailable: {exc}"

    results = []
    for i, p in enumerate(proba):
        p = float(p)
        item = {
            "index": i,
            "default_probability": round(p, 6),
            "predicted_class": "default" if p >= threshold else "non_default",
            "predicted_label": int(p >= threshold),
            "risk_band": assign_risk_band(p, bands),
            "warnings": warnings[i],
        }
        exp = explanations[i]
        if exp is not None:
            item["top_factors"] = {
                "output_space": exp["output_space"],
                "increasing_risk": exp["top_risk_increasing_factors"],
                "decreasing_risk": exp["top_risk_decreasing_factors"],
            }
        results.append(item)

    return {
        "model": artifact["model_name"],
        "trained_at": artifact["trained_at"],
        "decision_threshold": round(threshold, 6),
        "risk_bands": bands,
        "trained_on_synthetic_data": bool(artifact["metrics"]["data"]["is_synthetic"]),
        "predictions": results,
        "explanation_error": explanation_error,
        "disclaimer": PREDICTION_DISCLAIMER,
    }
