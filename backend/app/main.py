"""FastAPI application for the credit-risk ML backend.

Run:  uvicorn app.main:app --reload   (from the backend/ directory)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings
from app.routes import data, explanation, prediction, training
from app.schemas.credit_risk import HealthResponse
from app.services.errors import CreditRiskError
from app.services.store import Store

__version__ = "1.0.0"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("credit_risk")


def _error(status_code: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder({"error": {"code": code, "message": message, "details": details}}),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(
        title="Credit Risk Scoring & Loan Default Analysis API",
        version=__version__,
        description="Member 4 ML backend: data validation, model training/evaluation, prediction and SHAP explanations.",
    )
    app.state.store = Store(settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(CreditRiskError)
    async def _domain_error(_: Request, exc: CreditRiskError):
        return _error(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, f"http_{exc.status_code}", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        details = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()
        ]
        return _error(422, "request_validation_error", "Request body or parameters are invalid.", details)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("Unhandled error")
        return _error(500, "internal_error", "Unexpected server error.")

    @app.get("/api/health", response_model=HealthResponse, tags=["health"])
    def health(request: Request):
        store: Store = request.app.state.store
        return HealthResponse(
            status="ok",
            service="credit-risk-ml",
            version=__version__,
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_loaded=store.has_dataset,
            model_loaded=store.peek_artifact() is not None,
        )

    for r in (data.router, training.router, prediction.router, explanation.router):
        app.include_router(r)
    return app


app = create_app()
