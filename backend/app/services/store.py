"""Process-level state: the current dataset and the active model artifact.

The dataset is persisted as CSV and the model with joblib so both survive a
restart. No database is introduced; the workspace had none to integrate with.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone

import pandas as pd

from app.config import Settings
from app.services.data_processing import load_csv
from app.services.errors import CreditRiskError, DatasetNotLoadedError, ModelNotTrainedError
from app.services.model_training import load_artifact, save_artifact

logger = logging.getLogger(__name__)


class TrainingInProgressError(CreditRiskError):
    status_code = 409
    code = "training_in_progress"


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.ensure_dirs()
        self._lock = threading.RLock()
        self._train_lock = threading.Lock()
        self._df: pd.DataFrame | None = None
        self._meta: dict | None = None
        self._artifact: dict | None = None
        self._load_from_disk()

    # ----------------------------------------------------------------- dataset
    def _load_from_disk(self) -> None:
        s = self.settings
        if s.dataset_path.exists():
            try:
                self._df = load_csv(s.dataset_path)
                self._meta = json.loads(s.dataset_meta_path.read_text(encoding="utf-8")) if s.dataset_meta_path.exists() else {}
            except Exception as exc:  # corrupt files should not prevent startup
                logger.warning("Could not reload stored dataset: %s", exc)
        try:
            self._artifact = load_artifact(s.model_path)
        except Exception as exc:
            logger.warning("Could not load model artifact: %s", exc)
            self._artifact = None

    def set_dataset(self, df: pd.DataFrame, raw: bytes, meta: dict) -> dict:
        meta = {**meta, "uploaded_at": datetime.now(timezone.utc).isoformat(), "n_rows": int(len(df)), "n_columns": int(df.shape[1])}
        with self._lock:
            tmp = self.settings.dataset_path.with_suffix(".tmp")
            tmp.write_bytes(raw)
            tmp.replace(self.settings.dataset_path)
            self.settings.dataset_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self._df, self._meta = df, meta
        return meta

    def get_dataset(self) -> tuple[pd.DataFrame, dict]:
        with self._lock:
            if self._df is None:
                raise DatasetNotLoadedError("No dataset uploaded. Call POST /api/data/upload first.")
            return self._df, dict(self._meta or {})

    @property
    def has_dataset(self) -> bool:
        return self._df is not None

    # ------------------------------------------------------------------- model
    def get_artifact(self) -> dict:
        with self._lock:
            if self._artifact is None:
                raise ModelNotTrainedError("No trained model available. Call POST /api/model/train first.")
            return self._artifact

    def peek_artifact(self) -> dict | None:
        return self._artifact

    def set_artifact(self, artifact: dict) -> None:
        save_artifact(artifact, self.settings.model_path, self.settings.metrics_path)
        with self._lock:
            self._artifact = artifact

    def training_guard(self):
        """Context manager that rejects concurrent training requests."""
        store = self

        class _Guard:
            def __enter__(self_inner):
                if not store._train_lock.acquire(blocking=False):
                    raise TrainingInProgressError("A training job is already running.")
                return self_inner

            def __exit__(self_inner, *exc):
                store._train_lock.release()
                return False

        return _Guard()

    @property
    def is_training(self) -> bool:
        return self._train_lock.locked()
