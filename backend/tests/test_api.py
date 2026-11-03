from __future__ import annotations

from tests.conftest import BORROWER, TARGET, make_client, upload


def _err(r):
    body = r.json()
    assert set(body) == {"error"} and {"code", "message"} <= set(body["error"])
    return body["error"]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is False and body["dataset_loaded"] is False


def test_endpoints_without_data_or_model(client):
    assert _err(client.get("/api/data/summary"))["code"] == "dataset_not_loaded"
    assert client.post("/api/model/train", json={"target_column": TARGET}).status_code == 409
    for r in (
        client.get("/api/model/metrics"),
        client.post("/api/predict", json={"records": [BORROWER]}),
        client.post("/api/explain", json={}),
    ):
        assert r.status_code == 409
        assert _err(r)["code"] == "model_not_trained"
    status = client.get("/api/model/status").json()
    assert status["model_available"] is False and status["model_name"] is None


def test_upload_validation(client, tmp_path, synthetic_csv):
    r = client.post("/api/data/upload", files={"file": ("loans.xlsx", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 415
    r = client.post("/api/data/upload", files={"file": ("loans.csv", b"a,b\n1,2\n", "image/png")})
    assert r.status_code == 415
    r = client.post("/api/data/upload", files={"file": ("loans.csv", b"", "text/csv")})
    assert r.status_code == 422
    r = upload(client, synthetic_csv, target_column="nope")
    assert r.status_code == 422
    assert "suggested_targets" in _err(r)["details"]
    small = make_client(tmp_path / "small", max_upload_mb=0.01)
    assert upload(small, synthetic_csv).status_code == 413


def test_upload_and_summary(client, synthetic_csv):
    r = upload(client, synthetic_csv)
    assert r.status_code == 201
    body = r.json()
    assert body["dataset"]["is_synthetic"] is True  # inferred from filename
    assert body["summary"]["target"]["positive_labels"] == ["defaulted"]
    s = client.get("/api/data/summary").json()["summary"]
    assert s["n_rows"] == body["summary"]["n_rows"]
    assert s["duplicate_rows"] > 0
    assert "total_recovered_amount" in s["excluded_columns"]


def test_train_request_validation(client, synthetic_csv):
    upload(client, synthetic_csv)
    r = client.post("/api/model/train", json={"models": ["xgboost"]})
    assert r.status_code == 422 and _err(r)["code"] == "request_validation_error"
    r = client.post("/api/model/train", json={"unknown_field": 1})
    assert r.status_code == 422
    r = client.post("/api/model/train", json={"risk_bands": [{"name": "Low", "max_probability": 0.5}]})
    assert r.status_code == 422


def test_train_metrics_status(trained_client):
    m = trained_client.get("/api/model/metrics")
    assert m.status_code == 200
    metrics = m.json()
    assert metrics["selected_model"] in {"logistic_regression", "random_forest", "decision_tree"}
    assert len(metrics["comparison"]) == 3
    for res in metrics["models"].values():
        assert {"precision", "recall", "f1", "roc_auc", "confusion_matrix", "classification_report", "default_class_recall"} <= set(res["test"])
    assert metrics["data"]["is_synthetic"] is True
    status = trained_client.get("/api/model/status").json()
    assert status["model_available"] is True
    assert "credit_score" in status["feature_schema"]["numeric"]
    assert trained_client.get("/api/health").json()["model_loaded"] is True


def test_predict_endpoint(trained_client):
    r = trained_client.post("/api/predict", json={"records": [BORROWER, {**BORROWER, "credit_score": 820}], "top_k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["predictions"]) == 2
    p = body["predictions"][0]
    assert 0 <= p["default_probability"] <= 1
    assert p["risk_band"] in {"Low", "Medium", "High"}
    assert len(p["top_factors"]["increasing_risk"]) <= 3
    assert body["trained_on_synthetic_data"] is True
    assert "not guarantees" in body["disclaimer"]
    # better credit score should not increase predicted risk in this simulation
    assert body["predictions"][1]["default_probability"] < p["default_probability"]


def test_predict_invalid_requests(trained_client):
    assert trained_client.post("/api/predict", json={"records": []}).status_code == 422
    assert trained_client.post("/api/predict", json={}).status_code == 422
    r = trained_client.post("/api/predict", json={"records": [{**BORROWER, "credit_score": "high"}]})
    assert r.status_code == 422 and _err(r)["code"] == "invalid_input"
    r = trained_client.post("/api/predict", json={"records": [{"age": 30}]})
    assert r.status_code == 422
    assert "missing required fields" in _err(r)["details"][0]["error"]
    r = trained_client.post("/api/predict", json={"records": [{**BORROWER, "home_ownership": {"x": 1}}]})
    assert r.status_code == 422


def test_explain_endpoint(trained_client):
    r = trained_client.post("/api/explain", json={"records": [BORROWER], "top_k": 4, "max_summary_points": 20})
    assert r.status_code == 200, r.text
    body = r.json()
    g = body["global_explanation"]
    assert g["feature_importance"][0]["rank"] == 1
    assert all(len(f["points"]) <= 20 for f in g["summary_plot_data"])
    loc = body["local_explanations"][0]
    assert loc["risk_band"] in {"Low", "Medium", "High"}
    assert len(loc["top_risk_increasing_factors"]) <= 4
    assert "not the real-world causes" in body["disclaimer"]
    only_global = trained_client.post("/api/explain").json()
    assert only_global["local_explanations"] == [] and only_global["global_explanation"]
    r = trained_client.post("/api/explain", json={"include_global": False})
    assert r.status_code == 422


def test_model_persists_across_restart(tmp_path, synthetic_csv):
    c = make_client(tmp_path)
    upload(c, synthetic_csv)
    assert c.post("/api/model/train", json={"models": ["decision_tree"]}).status_code == 200
    first = c.post("/api/predict", json={"records": [BORROWER], "include_explanation": False}).json()
    c2 = make_client(tmp_path)  # new app instance, same storage
    assert c2.get("/api/model/status").json()["model_name"] == "decision_tree"
    assert c2.get("/api/data/summary").status_code == 200
    second = c2.post("/api/predict", json={"records": [BORROWER], "include_explanation": False}).json()
    assert first["predictions"][0]["default_probability"] == second["predictions"][0]["default_probability"]
