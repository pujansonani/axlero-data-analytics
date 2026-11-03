"""Domain errors raised by ML services and mapped to HTTP responses in main.py."""
from __future__ import annotations


class CreditRiskError(Exception):
    """Base class. `status_code` is the HTTP status the API should return."""

    status_code = 400
    code = "credit_risk_error"

    def __init__(self, message: str, details: dict | list | None = None):
        super().__init__(message)
        self.message = message
        self.details = details


class DataValidationError(CreditRiskError):
    status_code = 422
    code = "data_validation_error"


class InsufficientDataError(CreditRiskError):
    status_code = 422
    code = "insufficient_data"


class DatasetNotLoadedError(CreditRiskError):
    status_code = 409
    code = "dataset_not_loaded"


class ModelNotTrainedError(CreditRiskError):
    status_code = 409
    code = "model_not_trained"


class InvalidInputError(CreditRiskError):
    status_code = 422
    code = "invalid_input"
