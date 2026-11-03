"""Dataset upload and summary endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from app.schemas.credit_risk import ErrorResponse, SummaryResponse, UploadResponse
from app.services.data_processing import build_quality_report, load_csv, resolve_target
from app.services.errors import DataValidationError
from app.services.store import Store
from app.routes.deps import get_store

router = APIRouter(prefix="/api/data", tags=["data"])

ALLOWED_CONTENT_TYPES = {
    "text/csv", "application/csv", "text/plain", "application/vnd.ms-excel",
    "application/octet-stream", "text/x-csv", "application/x-csv",
}
_ERRORS = {
    409: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
    415: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
}


def parse_labels(raw: Optional[str]) -> Optional[list[Any]]:
    """Accept a JSON list (e.g. '["Default", 1]') or a comma-separated string."""
    if raw is None or not raw.strip():
        return None
    raw = raw.strip()
    if raw.startswith("["):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"Labels are not valid JSON: {exc.msg}") from exc
        if not isinstance(value, list) or not all(isinstance(v, (str, int, float, bool)) for v in value):
            raise DataValidationError("Labels must be a JSON list of strings/numbers.")
        return value
    return [p.strip() for p in raw.split(",") if p.strip()]


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    chunks, size = [], 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds the {limit // (1024 * 1024)} MB upload limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED, responses=_ERRORS)
async def upload_dataset(
    file: UploadFile = File(..., description="Loan history CSV"),
    target_column: Optional[str] = Form(None),
    positive_labels: Optional[str] = Form(None, description='JSON list or comma-separated, e.g. "Default"'),
    negative_labels: Optional[str] = Form(None),
    is_synthetic: Optional[bool] = Form(None, description="Mark the dataset as synthetic development data"),
    store: Store = Depends(get_store),
):
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only .csv files are accepted.")
    ctype = (file.content_type or "").split(";")[0].strip().lower()
    if ctype and ctype not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"Unsupported content type '{ctype}'.")

    raw = await _read_limited(file, store.settings.max_upload_bytes)
    df = load_csv(raw)
    pos, neg = parse_labels(positive_labels), parse_labels(negative_labels)
    target = target_column.strip() if target_column and target_column.strip() else None
    if target:
        resolve_target(df, target, pos, neg)  # fail fast on a bad target configuration

    synthetic = bool(is_synthetic) if is_synthetic is not None else ("synthetic" in filename.lower())
    meta = store.set_dataset(df, raw, {
        "filename": filename,
        "is_synthetic": synthetic,
        "target_column": target,
        "positive_labels": pos,
        "negative_labels": neg,
        "size_bytes": len(raw),
    })
    summary = build_quality_report(df, target, pos, neg)
    return UploadResponse(message="Dataset uploaded and validated.", dataset=meta, summary=summary)


@router.get("/summary", response_model=SummaryResponse, responses=_ERRORS)
def dataset_summary(
    target_column: Optional[str] = Query(None, description="Override the target configured at upload"),
    positive_labels: Optional[str] = Query(None),
    negative_labels: Optional[str] = Query(None),
    store: Store = Depends(get_store),
):
    df, meta = store.get_dataset()
    target = target_column or meta.get("target_column")
    # Upload-time label mappings only apply to the upload-time target column.
    same_target = target == meta.get("target_column")
    pos = parse_labels(positive_labels) if positive_labels else (meta.get("positive_labels") if same_target else None)
    neg = parse_labels(negative_labels) if negative_labels else (meta.get("negative_labels") if same_target else None)
    return SummaryResponse(dataset=meta, summary=build_quality_report(df, target, pos, neg))
