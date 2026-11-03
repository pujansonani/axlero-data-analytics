from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "scripts"))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from generate_synthetic_data import generate  # noqa: E402

TARGET = "loan_outcome"
POSITIVE = ["Defaulted"]

BORROWER = {
    "age": 29, "annual_income": 38000, "employment_length_years": 1.5, "employment_type": "contract",
    "home_ownership": "RENT", "loan_amount": 25000, "loan_term_months": 60, "interest_rate": 21.5,
    "loan_purpose": "small_business", "credit_score": 590, "debt_to_income": 34.0,
    "credit_utilization_pct": 85.0, "num_open_accounts": 12, "delinquencies_2y": 2,
}


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return generate(n_rows=1200, seed=11)


@pytest.fixture(scope="session")
def synthetic_csv(synthetic_df) -> bytes:
    buf = io.StringIO()
    synthetic_df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def make_client(tmp_path: Path, max_upload_mb: float = 20.0) -> TestClient:
    settings = Settings(model_dir=tmp_path / "models", data_dir=tmp_path / "data", max_upload_mb=max_upload_mb)
    return TestClient(create_app(settings))


def upload(client: TestClient, content: bytes, name: str = "synthetic_loans.csv", **form):
    data = {"target_column": TARGET, "positive_labels": "Defaulted", **form}
    return client.post("/api/data/upload", files={"file": (name, content, "text/csv")}, data=data)


@pytest.fixture
def client(tmp_path) -> TestClient:
    return make_client(tmp_path)


@pytest.fixture(scope="session")
def trained_client(tmp_path_factory, synthetic_csv) -> TestClient:
    c = make_client(tmp_path_factory.mktemp("trained"))
    assert upload(c, synthetic_csv).status_code == 201
    r = c.post("/api/model/train", json={})
    assert r.status_code == 200, r.text
    return c
