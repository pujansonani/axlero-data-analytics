"""Pydantic request/response schemas for the credit-risk API."""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

ModelName = Literal["logistic_regression", "random_forest", "decision_tree"]
SelectionMetric = Literal["roc_auc", "average_precision", "f1", "recall"]
Label = Union[str, int, float, bool]
FeatureValue = Union[str, int, float, bool, None]


class _Schema(BaseModel):
    # Allow field names like "model_name" without pydantic namespace warnings.
    model_config = ConfigDict(protected_namespaces=())


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# ----------------------------------------------------------------- health / data
class HealthResponse(_Schema):
    status: Literal["ok"]
    service: str
    version: str
    timestamp: str
    dataset_loaded: bool
    model_loaded: bool


class DatasetInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    filename: Optional[str] = None
    is_synthetic: bool = False
    target_column: Optional[str] = None
    positive_labels: Optional[list[Label]] = None
    negative_labels: Optional[list[Label]] = None
    uploaded_at: Optional[str] = None
    n_rows: Optional[int] = None
    n_columns: Optional[int] = None
    size_bytes: Optional[int] = None


class UploadResponse(BaseModel):
    message: str
    dataset: DatasetInfo
    summary: dict[str, Any]


class SummaryResponse(BaseModel):
    dataset: DatasetInfo
    summary: dict[str, Any]


# ----------------------------------------------------------------------- training
class RiskBand(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    max_probability: float = Field(gt=0, le=1)


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_column: Optional[str] = Field(None, description="Defaults to the target given at upload time.")
    positive_labels: Optional[list[Label]] = Field(None, description="Target labels meaning 'default'.")
    negative_labels: Optional[list[Label]] = Field(None, description="Target labels meaning 'non-default'. Others are dropped.")
    exclude_columns: list[str] = Field(default_factory=list, description="Extra columns to exclude (e.g. post-outcome fields).")
    auto_exclude_leakage: bool = True
    models: list[ModelName] = Field(default_factory=lambda: ["logistic_regression", "random_forest", "decision_tree"], min_length=1)
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    validation_size: float = Field(0.2, ge=0.05, le=0.5)
    random_state: int = 42
    selection_metric: SelectionMetric = "roc_auc"
    min_default_recall: Optional[float] = Field(None, gt=0, le=1, description="If set, tune threshold for at least this validation recall.")
    risk_bands: Optional[list[RiskBand]] = None

    @field_validator("models")
    @classmethod
    def _unique_models(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))


class TrainResponse(BaseModel):
    message: str
    selected_model: str
    selection_metric: str
    decision_threshold: float
    comparison: list[dict[str, Any]]
    split: dict[str, Any]
    excluded_columns: dict[str, str]
    trained_on_synthetic_data: bool
    warnings: list[str]


# -------------------------------------------------------------------- prediction
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, FeatureValue]] = Field(min_length=1, max_length=1000)
    include_explanation: bool = True
    top_k: int = Field(5, ge=1, le=20)


class Factor(BaseModel):
    feature: str
    value: Any
    shap_value: float
    effect: Literal["increases_risk", "decreases_risk", "neutral"]


class TopFactors(BaseModel):
    output_space: Literal["probability", "log_odds"]
    increasing_risk: list[Factor]
    decreasing_risk: list[Factor]


class PredictionItem(BaseModel):
    index: int
    default_probability: float = Field(ge=0, le=1)
    predicted_class: Literal["default", "non_default"]
    predicted_label: Literal[0, 1]
    risk_band: str
    warnings: list[str]
    top_factors: Optional[TopFactors] = None


class PredictResponse(BaseModel):
    model: str
    trained_at: str
    decision_threshold: float
    risk_bands: list[RiskBand]
    trained_on_synthetic_data: bool
    predictions: list[PredictionItem]
    explanation_error: Optional[str] = None
    disclaimer: str


# ------------------------------------------------------------------- explanation
class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: Optional[list[dict[str, FeatureValue]]] = Field(None, max_length=100)
    include_global: bool = True
    top_k: int = Field(5, ge=1, le=20)
    max_summary_points: int = Field(200, ge=10, le=500)


class LocalExplanation(_Schema):
    index: int
    output_space: Literal["probability", "log_odds"]
    base_value: float
    model_output: float
    predicted_default_probability: float
    risk_band: str
    top_risk_increasing_factors: list[Factor]
    top_risk_decreasing_factors: list[Factor]
    contributions: list[Factor]
    warnings: list[str]


class ExplainResponse(BaseModel):
    model: str
    explainer: str
    global_explanation: Optional[dict[str, Any]] = None
    local_explanations: list[LocalExplanation] = Field(default_factory=list)
    disclaimer: str


# ------------------------------------------------------------------------ status
class ModelStatusResponse(_Schema):
    model_available: bool
    is_training: bool
    dataset_loaded: bool
    model_name: Optional[str] = None
    trained_at: Optional[str] = None
    decision_threshold: Optional[float] = None
    target: Optional[dict[str, Any]] = None
    feature_schema: Optional[dict[str, Any]] = None
    risk_bands: Optional[list[RiskBand]] = None
    trained_on_synthetic_data: Optional[bool] = None
    test_metrics: Optional[dict[str, Any]] = None
