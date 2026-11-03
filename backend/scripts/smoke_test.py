"""End-to-end smoke test against the HTTP API using the synthetic dataset.

Usage: python scripts/smoke_test.py   (uses an in-process TestClient, no server needed)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "synthetic_loans.csv"


def main() -> None:
    client = TestClient(create_app())
    print("health:", client.get("/api/health").json())
    with DATA.open("rb") as fh:
        r = client.post(
            "/api/data/upload",
            files={"file": (DATA.name, fh, "text/csv")},
            data={"target_column": "loan_outcome", "positive_labels": "Defaulted", "is_synthetic": "true"},
        )
    r.raise_for_status()
    s = r.json()["summary"]
    print("upload:", r.status_code, "rows", s["n_rows"], "dupes", s["duplicate_rows"], "target", s["target"]["class_counts"])
    print("excluded:", s["excluded_columns"])

    r = client.post("/api/model/train", json={})
    r.raise_for_status()
    t = r.json()
    print("selected:", t["selected_model"], "threshold", t["decision_threshold"])
    for row in t["comparison"]:
        print("  ", json.dumps(row))
    print("warnings:", t["warnings"])

    borrower = {
        "age": 29, "annual_income": 38000, "employment_length_years": 1.5, "employment_type": "contract",
        "home_ownership": "RENT", "loan_amount": 25000, "loan_term_months": 60, "interest_rate": 21.5,
        "loan_purpose": "small_business", "credit_score": 590, "debt_to_income": 34.0,
        "credit_utilization_pct": 85.0, "num_open_accounts": 12, "delinquencies_2y": 2,
    }
    r = client.post("/api/predict", json={"records": [borrower], "top_k": 3})
    r.raise_for_status()
    p = r.json()["predictions"][0]
    print("predict:", p["default_probability"], p["predicted_class"], p["risk_band"])
    print("  up:", [(f["feature"], f["shap_value"]) for f in p["top_factors"]["increasing_risk"]])
    r = client.post("/api/explain", json={"records": [borrower], "top_k": 3})
    r.raise_for_status()
    e = r.json()
    print("explain:", e["explainer"], [(f["feature"], f["share_of_total"]) for f in e["global_explanation"]["feature_importance"][:5]])
    print("status:", client.get("/api/model/status").json()["test_metrics"])


if __name__ == "__main__":
    main()
