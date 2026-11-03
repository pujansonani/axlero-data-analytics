"""Model training, metrics and status endpoints."""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends

from app.routes.deps import get_store
from app.schemas.credit_risk import ErrorResponse, ModelStatusResponse, TrainRequest, TrainResponse
from app.services.errors import DataValidationError
from app.services.explainability import global_explanation
from app.services.model_training import TrainingConfig, train_models
from app.services.store import Store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/model", tags=["model"])
_ERRORS = {409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}


@router.post("/train", response_model=TrainResponse, responses=_ERRORS)
def train(req: Optional[TrainRequest] = Body(None), store: Store = Depends(get_store)):
    # Sync handler: FastAPI runs it in a worker thread, so CPU-bound training
    # does not block the event loop.
    req = req or TrainRequest()
    df, meta = store.get_dataset()
    target = req.target_column or meta.get("target_column")
    if not target:
        raise DataValidationError(
            "target_column is required (not set at upload time).",
            {"available_columns": list(df.columns)},
        )
    same_target = target == meta.get("target_column")
    cfg = TrainingConfig(
        target_column=target,
        positive_labels=req.positive_labels if req.positive_labels is not None else (meta.get("positive_labels") if same_target else None),
        negative_labels=req.negative_labels if req.negative_labels is not None else (meta.get("negative_labels") if same_target else None),
        exclude_columns=req.exclude_columns,
        auto_exclude_leakage=req.auto_exclude_leakage,
        models=list(req.models),
        test_size=req.test_size,
        validation_size=req.validation_size,
        random_state=req.random_state,
        selection_metric=req.selection_metric,
        min_default_recall=req.min_default_recall,
        **({"risk_bands": [b.model_dump() for b in req.risk_bands]} if req.risk_bands else {}),
    )
    with store.training_guard():
        artifact = train_models(df, cfg, meta)
        # Fail before persisting if SHAP cannot explain the selected model.
        global_explanation(artifact)
        store.set_artifact(artifact)

    m = artifact["metrics"]
    warnings = list(m["notes"])
    for leak in m["data"]["statistical_leakage_warnings"]:
        warnings.append(
            f"'{leak['column']}' alone separates classes almost perfectly (AUC {leak['single_feature_auc']}); "
            "confirm it is known at loan-approval time or add it to exclude_columns."
        )
    return TrainResponse(
        message=f"Trained {len(cfg.models)} model(s); selected '{m['selected_model']}'.",
        selected_model=m["selected_model"],
        selection_metric=m["selection_metric"],
        decision_threshold=m["decision_threshold"],
        comparison=m["comparison"],
        split=m["split"],
        excluded_columns=m["data"]["excluded_columns"],
        trained_on_synthetic_data=m["data"]["is_synthetic"],
        warnings=warnings,
    )


@router.get("/metrics", response_model=dict[str, Any], responses=_ERRORS)
def metrics(store: Store = Depends(get_store)):
    return store.get_artifact()["metrics"]


@router.get("/status", response_model=ModelStatusResponse)
def model_status(store: Store = Depends(get_store)):
    art = store.peek_artifact()
    base = {"model_available": art is not None, "is_training": store.is_training, "dataset_loaded": store.has_dataset}
    if art is None:
        return ModelStatusResponse(**base)
    m = art["metrics"]
    test = m["models"][art["model_name"]]["test"]
    return ModelStatusResponse(
        **base,
        model_name=art["model_name"],
        trained_at=art["trained_at"],
        decision_threshold=art["threshold"],
        target=art["target_mapping"],
        feature_schema=art["feature_schema"],
        risk_bands=art["risk_bands"],
        trained_on_synthetic_data=m["data"]["is_synthetic"],
        test_metrics={k: test[k] for k in ("precision", "recall", "f1", "roc_auc", "average_precision")},
    )
