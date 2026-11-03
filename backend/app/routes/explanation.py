"""SHAP explanation endpoint (global and per-borrower)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, Depends

from app.config import EXPLANATION_DISCLAIMER
from app.routes.deps import get_store
from app.schemas.credit_risk import ErrorResponse, ExplainRequest, ExplainResponse
from app.services.errors import InvalidInputError
from app.services.explainability import TREE_MODELS, global_explanation, local_explanations
from app.services.prediction import assign_risk_band, validate_records
from app.services.store import Store

router = APIRouter(prefix="/api", tags=["explanation"])


@router.post("/explain", response_model=ExplainResponse, responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def explain(req: Optional[ExplainRequest] = Body(None), store: Store = Depends(get_store)):
    req = req or ExplainRequest()
    artifact = store.get_artifact()
    if not req.include_global and not req.records:
        raise InvalidInputError("Provide borrower records and/or set include_global=true.")

    glob = None
    if req.include_global:
        glob = global_explanation(artifact)
        if req.max_summary_points < 200:
            glob = {
                **glob,
                "summary_plot_data": [
                    {**f, "points": f["points"][: req.max_summary_points]} for f in glob["summary_plot_data"]
                ],
            }

    local = []
    if req.records:
        X, warnings = validate_records(req.records, artifact)
        for i, exp in enumerate(local_explanations(artifact, X, top_k=req.top_k)):
            local.append({
                "index": i,
                **exp,
                "risk_band": assign_risk_band(exp["predicted_default_probability"], artifact["risk_bands"]),
                "warnings": warnings[i],
            })

    return ExplainResponse(
        model=artifact["model_name"],
        explainer="TreeExplainer" if artifact["model_name"] in TREE_MODELS else "LinearExplainer",
        global_explanation=glob,
        local_explanations=local,
        disclaimer=EXPLANATION_DISCLAIMER,
    )
