from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.data_processing import prepare_training_data
from app.services.errors import DataValidationError, InvalidInputError
from app.services.evaluation import evaluate_predictions, tune_threshold
from app.services.explainability import global_explanation, local_explanations
from app.services.model_training import (
    TrainingConfig,
    build_preprocessor,
    load_artifact,
    save_artifact,
    train_models,
    validate_risk_bands,
)
from app.services.prediction import assign_risk_band, predict, validate_records
from tests.conftest import BORROWER, POSITIVE, TARGET


@pytest.fixture(scope="module")
def artifacts(synthetic_df) -> dict[str, dict]:
    """One trained artifact per model type so SHAP is tested for each explainer."""
    out = {}
    for name in ("logistic_regression", "random_forest", "decision_tree"):
        cfg = TrainingConfig(target_column=TARGET, positive_labels=POSITIVE, models=[name])
        out[name] = train_models(synthetic_df, cfg, {"is_synthetic": True})
    return out


# ------------------------------------------------------------------ preprocessing
def test_preprocessor_imputes_missing_values(synthetic_df):
    prepared = prepare_training_data(synthetic_df, TARGET, POSITIVE)
    assert prepared.X.isna().any().any()  # dataset really has gaps
    pre = build_preprocessor(prepared.numeric_features, prepared.categorical_features)
    Xt = pre.fit_transform(prepared.X)
    assert not np.isnan(Xt).any()


def test_categorical_encoding_handles_unseen_values():
    X = pd.DataFrame({"n": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2, "c": ["a"] * 6 + ["b"] * 6})
    pre = build_preprocessor(["n"], ["c"]).fit(X)
    names = list(pre.named_steps["columns"].get_feature_names_out())
    assert "cat__c==a" in names and "cat__c==b" in names
    unseen = pre.transform(pd.DataFrame({"n": [1.0], "c": ["never_seen"]}))
    cat_cols = [i for i, n in enumerate(names) if n.startswith("cat__")]
    assert unseen[0, cat_cols].sum() == 0  # unknown -> all-zero one-hot, no error


def test_preprocessing_is_fitted_on_training_split_only(artifacts):
    art = artifacts["logistic_regression"]
    scaler = art["pipeline"].named_steps["preprocessor"].named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    assert scaler.n_samples_seen_ == art["metrics"]["split"]["train_rows"]


# ---------------------------------------------------------------------- training
def test_train_all_models_and_compare(synthetic_df):
    art = train_models(synthetic_df, TrainingConfig(target_column=TARGET, positive_labels=POSITIVE))
    m = art["metrics"]
    assert set(m["models"]) == {"logistic_regression", "random_forest", "decision_tree"}
    assert [c["rank"] for c in m["comparison"]] == [1, 2, 3]
    assert m["comparison"][0]["model"] == m["selected_model"] == art["model_name"]
    split = m["split"]
    assert split["train_rows"] + split["validation_rows"] + split["test_rows"] == m["data"]["rows_used"]
    # stratification keeps default rates close across splits
    assert abs(split["train_default_rate"] - split["test_default_rate"]) < 0.02
    assert "total_recovered_amount" not in art["numeric_features"]


def test_training_is_reproducible(synthetic_df):
    cfg = lambda: TrainingConfig(target_column=TARGET, positive_labels=POSITIVE, models=["random_forest"])  # noqa: E731
    a, b = train_models(synthetic_df, cfg()), train_models(synthetic_df, cfg())
    assert a["metrics"]["models"]["random_forest"]["test"] == b["metrics"]["models"]["random_forest"]["test"]


def test_training_config_validation(synthetic_df):
    with pytest.raises(DataValidationError):
        train_models(synthetic_df, TrainingConfig(target_column=TARGET, positive_labels=POSITIVE, models=["svm"]))
    with pytest.raises(DataValidationError):
        train_models(synthetic_df, TrainingConfig(target_column=TARGET, positive_labels=POSITIVE, test_size=0.45, validation_size=0.4))


def test_artifact_roundtrip(tmp_path, artifacts):
    art = artifacts["random_forest"]
    save_artifact(art, tmp_path / "m.joblib", tmp_path / "metrics.json")
    loaded = load_artifact(tmp_path / "m.joblib")
    X, _ = validate_records([BORROWER], art)
    assert np.allclose(art["pipeline"].predict_proba(X), loaded["pipeline"].predict_proba(X))
    assert (tmp_path / "metrics.json").exists()
    assert "_runtime" not in loaded


# -------------------------------------------------------------------- evaluation
def test_evaluate_predictions_exact_values():
    y = [0, 0, 1, 1, 1, 0]
    p = [0.1, 0.6, 0.7, 0.4, 0.9, 0.2]
    m = evaluate_predictions(y, p, threshold=0.5)
    # predictions: 0,1,1,0,1,0 -> TP=2 FP=1 FN=1 TN=2
    assert m["confusion_matrix"]["matrix"] == [[2, 1], [1, 2]]
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == m["default_class_recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["roc_auc"] == pytest.approx(8 / 9, abs=1e-4)
    assert m["classification_report"]["default"]["support"] == 3


def test_evaluate_single_class_has_no_auc():
    m = evaluate_predictions([0, 0, 0], [0.1, 0.2, 0.9])
    assert m["roc_auc"] is None


def test_tune_threshold_strategies():
    y = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 1])
    p = np.linspace(0.05, 0.95, 10)
    f1 = tune_threshold(y, p)
    assert f1["strategy"] == "max_f1" and 0 < f1["threshold"] < 1
    rec = tune_threshold(y, p, min_recall=1.0)
    assert rec["validation_recall"] == 1.0


def test_metrics_are_real_and_in_range(artifacts):
    for art in artifacts.values():
        test = art["metrics"]["models"][art["model_name"]]["test"]
        for k in ("precision", "recall", "f1", "roc_auc", "average_precision"):
            assert 0.0 <= test[k] <= 1.0
        cm = test["confusion_matrix"]
        assert sum(map(sum, cm["matrix"])) == art["metrics"]["split"]["test_rows"]
        assert cm["true_positive"] + cm["false_negative"] == test["n_default"]


# -------------------------------------------------------------------- prediction
def test_prediction_probabilities_and_bands(artifacts):
    rng = np.random.default_rng(1)
    records = []
    for _ in range(25):
        r = dict(BORROWER)
        r["credit_score"] = float(rng.integers(450, 820))
        r["debt_to_income"] = float(rng.uniform(0, 50))
        records.append(r)
    for art in artifacts.values():
        out = predict(art, records, include_explanation=False)
        for p in out["predictions"]:
            assert 0.0 <= p["default_probability"] <= 1.0
            assert p["risk_band"] in {"Low", "Medium", "High"}
            assert p["predicted_label"] == int(p["default_probability"] >= out["decision_threshold"])


def test_prediction_validation(artifacts):
    art = artifacts["logistic_regression"]
    bad_type = {**BORROWER, "credit_score": "excellent"}
    with pytest.raises(InvalidInputError) as exc:
        validate_records([bad_type], art)
    assert exc.value.details[0]["field"] == "credit_score"
    missing = {k: v for k, v in BORROWER.items() if k != "age"}
    with pytest.raises(InvalidInputError):
        validate_records([missing], art)
    with pytest.raises(InvalidInputError):
        validate_records([{**BORROWER, "age": True}], art)


def test_prediction_unseen_category_and_nulls_are_safe(artifacts):
    rec = {**BORROWER, "loan_purpose": "space_travel", "employment_length_years": None, "extra": 1}
    for art in artifacts.values():
        p = predict(art, [rec], include_explanation=True)["predictions"][0]
        assert 0 <= p["default_probability"] <= 1
        joined = " ".join(p["warnings"])
        assert "not seen during training" in joined and "imputed" in joined and "Ignored fields" in joined


def test_risk_bands():
    bands = [{"name": "Low", "max_probability": 0.3}, {"name": "High", "max_probability": 1.0}]
    assert assign_risk_band(0.29, bands) == "Low"
    assert assign_risk_band(0.3, bands) == "High"
    assert assign_risk_band(1.0, bands) == "High"
    with pytest.raises(DataValidationError):
        validate_risk_bands([{"name": "A", "max_probability": 0.5}])
    with pytest.raises(DataValidationError):
        validate_risk_bands([{"name": "A", "max_probability": 0.6}, {"name": "B", "max_probability": 0.4}, {"name": "C", "max_probability": 1.0}])


# -------------------------------------------------------------------------- SHAP
@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "decision_tree"])
def test_shap_local_explanations_are_additive(artifacts, name):
    art = artifacts[name]
    X, _ = validate_records([BORROWER, {**BORROWER, "credit_score": 800, "delinquencies_2y": 0}], art)
    exps = local_explanations(art, X, top_k=3)
    proba = art["pipeline"].predict_proba(X)[:, 1]
    for exp, p in zip(exps, proba):
        total = exp["base_value"] + sum(c["shap_value"] for c in exp["contributions"])
        expected = p if exp["output_space"] == "probability" else np.log(p / (1 - p))
        assert total == pytest.approx(expected, abs=1e-3)
        assert len(exp["top_risk_increasing_factors"]) <= 3
        assert all(c["shap_value"] > 0 for c in exp["top_risk_increasing_factors"])
        assert all(c["shap_value"] < 0 for c in exp["top_risk_decreasing_factors"])
        # explanations are reported on original borrower fields, not one-hot columns
        assert {c["feature"] for c in exp["contributions"]} == set(art["numeric_features"] + art["categorical_features"])
    assert exps[0]["output_space"] == ("log_odds" if name == "logistic_regression" else "probability")


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest"])
def test_shap_global_importance(artifacts, name):
    g = global_explanation(artifacts[name])
    shares = [f["share_of_total"] for f in g["feature_importance"]]
    assert sum(shares) == pytest.approx(1.0, abs=0.01)
    assert shares == sorted(shares, reverse=True)
    # credit_score drives the simulated risk most strongly
    assert "credit_score" in [f["feature"] for f in g["feature_importance"][:3]]
    assert g["summary_plot_data"] and g["summary_plot_data"][0]["points"]
    assert "not the real-world causes" in g["disclaimer"]
