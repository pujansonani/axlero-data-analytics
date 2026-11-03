"""Credit-risk prediction endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.routes.deps import get_store
from app.schemas.credit_risk import ErrorResponse, PredictRequest, PredictResponse
from app.services.prediction import predict
from app.services.store import Store

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictResponse, responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def predict_default_risk(req: PredictRequest, store: Store = Depends(get_store)):
    artifact = store.get_artifact()
    return predict(artifact, req.records, include_explanation=req.include_explanation, top_k=req.top_k)
