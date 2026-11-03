"""Runtime configuration for the ML backend.

All values can be overridden with environment variables so the module can be
deployed alongside other team members' services without code changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    """Paths and limits used by the ML services."""

    model_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("CREDIT_RISK_MODEL_DIR", BACKEND_DIR / "models"))
    )
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("CREDIT_RISK_DATA_DIR", BACKEND_DIR / "data" / "uploads"))
    )
    max_upload_mb: float = field(default_factory=lambda: _env_float("CREDIT_RISK_MAX_UPLOAD_MB", 20.0))
    # CORS origins for the team's frontend/dashboard (comma separated).
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.environ.get(
                "CREDIT_RISK_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
            ).split(",")
            if o.strip()
        ]
    )

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def model_path(self) -> Path:
        return self.model_dir / "credit_risk_model.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.model_dir / "metrics.json"

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / "current_dataset.csv"

    @property
    def dataset_meta_path(self) -> Path:
        return self.data_dir / "current_dataset_meta.json"

    def ensure_dirs(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Default risk bands: probability upper bounds (exclusive) for each band.
# The last band always extends to 1.0.
DEFAULT_RISK_BANDS: list[dict] = [
    {"name": "Low", "max_probability": 0.20},
    {"name": "Medium", "max_probability": 0.50},
    {"name": "High", "max_probability": 1.00},
]

PREDICTION_DISCLAIMER = (
    "Predictions are statistical estimates from a model trained on historical data. "
    "They are not guarantees of future borrower behaviour and should support, not replace, "
    "human credit decisions and applicable lending regulations."
)

EXPLANATION_DISCLAIMER = (
    "SHAP values describe how each input moved this model's output relative to a baseline. "
    "They explain the model's behaviour, not the real-world causes of default."
)
