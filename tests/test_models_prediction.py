"""Unit tests for trained model loading, input validation, and credit risk prediction."""

import joblib
import pytest

from src.models.train_model import predict_single_borrower, validate_borrower_input
from src.utils import load_config, resolve_path


def test_validate_borrower_input_valid_and_invalid():
    valid_payload = {
        "person_age": 30,
        "person_income": 65000,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 6.0,
        "loan_intent": "EDUCATION",
        "loan_grade": "A",
        "loan_amnt": 8000,
        "loan_int_rate": 7.49,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 7,
    }
    cleaned = validate_borrower_input(valid_payload)
    assert cleaned["loan_percent_income"] == round(8000 / 65000, 2)

    # Invalid age
    bad_age = dict(valid_payload, person_age=144)
    with pytest.raises(ValueError, match="person_age"):
        validate_borrower_input(bad_age)

    # Employment length exceeding working age
    bad_emp = dict(valid_payload, person_age=22, person_emp_length=15)
    with pytest.raises(ValueError, match="Employment length"):
        validate_borrower_input(bad_emp)

    # Invalid categorical value
    bad_grade = dict(valid_payload, loan_grade="Z")
    with pytest.raises(ValueError, match="loan_grade"):
        validate_borrower_input(bad_grade)


def test_saved_pipeline_loads_and_predicts_accurately():
    cfg = load_config()
    pipeline_path = resolve_path(cfg["paths"]["pipeline_path"])
    assert pipeline_path.exists(), "Serialized pipeline must exist in models/"

    bundle = joblib.load(pipeline_path)
    assert "feature_engineer" in bundle
    assert "champion_model" in bundle
    assert "selected_features" in bundle

    low_risk_borrower = {
        "person_age": 32,
        "person_income": 110000,
        "person_home_ownership": "OWN",
        "person_emp_length": 10.0,
        "loan_intent": "VENTURE",
        "loan_grade": "A",
        "loan_amnt": 6000,
        "loan_int_rate": 6.99,
        "loan_percent_income": 0.05,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 9,
    }
    high_risk_borrower = {
        "person_age": 22,
        "person_income": 24000,
        "person_home_ownership": "RENT",
        "person_emp_length": 0.0,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_grade": "F",
        "loan_amnt": 12000,
        "loan_int_rate": 18.53,
        "loan_percent_income": 0.50,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 2,
    }

    low_res = predict_single_borrower(bundle, low_risk_borrower)
    high_res = predict_single_borrower(bundle, high_risk_borrower)

    assert 0.0 <= low_res["default_probability"] < 0.15
    assert low_res["risk_category"] == "Low Risk"
    assert low_res["predicted_default"] == 0

    assert 0.60 < high_res["default_probability"] <= 1.0
    assert high_res["risk_category"] == "High Risk"
    assert high_res["predicted_default"] == 1
    assert len(high_res["top_factors"]) == 8

    # Verify strict prediction reproducibility
    high_res_repeat = predict_single_borrower(bundle, high_risk_borrower)
    assert high_res["default_probability"] == high_res_repeat["default_probability"]
    assert high_res["top_factors"] == high_res_repeat["top_factors"]


def test_streamlit_dashboard_all_8_pages_and_invalid_input():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(resolve_path("app/streamlit_app.py"), default_timeout=30)
    at.run()
    assert not at.exception

    for page in [
        "2. Dataset Overview",
        "3. Data Quality & Preprocessing",
        "4. Exploratory Data Analysis",
        "5. Risk Factors & Statistical Analysis",
        "6. WoE & IV Analysis",
        "7. Model Performance Comparison",
        "8. Credit Risk Prediction",
    ]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"Exception on {page}: {at.exception}"

    # Trigger invalid input on Page 8 (Age=20, Employment Length=12.0 > 20-14)
    at.number_input[0].set_value(20)
    at.number_input[2].set_value(12.0)
    at.button[-1].click().run()
    assert not at.exception
    assert len(at.error) > 0
    assert "Input Validation Error" in at.error[0].value



def test_no_hardcoded_local_paths_in_code_or_notebooks():
    root = resolve_path(".")
    forbidden_substrings = [r"C:\Users\darsh", "/Users/pujansonani"]
    for ext in ["*.py", "*.ipynb"]:
        for path in root.rglob(ext):
            if ".venv" in path.parts or ".pytest_cache" in path.parts or path.name == "test_models_prediction.py":
                continue
            text = path.read_text(encoding="utf-8")
            for bad in forbidden_substrings:
                assert bad not in text, f"Found hardcoded path '{bad}' in {path}"


def test_all_16_figures_and_reports_exist():
    root = resolve_path(".")
    fig_dir = root / "reports" / "figures"
    pngs = list(fig_dir.glob("fig*.png"))
    assert len(pngs) == 16, f"Expected 16 figures, found {len(pngs)}"
    for rep in [
        "data_quality_checklist.md",
        "data_quality_report.md",
        "eda_report.md",
        "statistical_analysis.md",
        "model_evaluation_report.md",
        "final_project_report.md",
        "final_qa_report.md",
    ]:
        assert (root / "reports" / rep).exists(), f"Missing report {rep}"

