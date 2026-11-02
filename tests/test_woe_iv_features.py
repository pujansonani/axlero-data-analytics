"""Unit tests for Statistical Analysis, WoE/IV calculations, and Feature Engineering pipeline."""

import numpy as np
import pandas as pd

from src.data.load_data import load_validated_data, stratified_train_val_test_split
from src.features.feature_engineering import CreditRiskFeatureEngineer, compute_vif_table
from src.features.statistical_tests import analyze_categorical_features, analyze_numerical_features
from src.features.woe_iv import WoEIVAnalyzer, interpret_iv


def test_statistical_analysis_outputs():
    df = load_validated_data()
    num_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]
    cat_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]
    num_df = analyze_numerical_features(df, num_cols)
    cat_df, lvl_tables = analyze_categorical_features(df, cat_cols)

    assert len(num_df) == 7
    assert (num_df["ks_statistic"] >= 0).all()
    assert len(cat_df) == 4
    assert "loan_grade" in lvl_tables
    assert len(lvl_tables["loan_grade"]) == 7


def test_woe_iv_analyzer_zero_bin_laplace_smoothing():
    df = load_validated_data()
    df_train, df_val, _ = stratified_train_val_test_split(df)

    # Create a synthetic feature with a zero-event bin to test Laplace smoothing
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_train["zero_event_cat"] = np.where(df_train["loan_status"] == 0, "ONLY_NON_EVENT", "MIXED")
    # Make a small portion of MIXED also non-event
    df_train.loc[df_train.index[:50], "zero_event_cat"] = "MIXED"
    df_val["zero_event_cat"] = "ONLY_NON_EVENT"

    analyzer = WoEIVAnalyzer(num_bins=5, laplace_alpha=0.5).fit(
        df_train,
        numeric_cols=["person_income", "loan_int_rate"],
        categorical_cols=["loan_grade", "zero_event_cat"],
    )
    assert analyzer.iv_summary_ is not None
    assert np.isfinite(analyzer.iv_summary_["iv"]).all()
    assert (analyzer.iv_summary_["iv"] >= 0).all()

    woe_val = analyzer.transform_woe(df_val)
    assert not woe_val.isnull().any().any()
    assert np.isfinite(woe_val.values).all()
    assert interpret_iv(0.01) == "Not Predictive (< 0.02)"
    assert interpret_iv(0.35) == "Strong Predictor (0.30 - 0.50)"


def test_feature_engineering_fit_on_train_only():
    df = load_validated_data().iloc[:1500].copy()
    df_train, df_val, df_test = stratified_train_val_test_split(df)

    fe = CreditRiskFeatureEngineer(imputation_strategy="grade_median", scale_numeric=True)
    fe.fit(df_train.drop(columns=["loan_status"]))

    X_tr = fe.transform(df_train.drop(columns=["loan_status"]))
    X_te = fe.transform(df_test.drop(columns=["loan_status"]))

    assert X_tr.shape[1] == X_te.shape[1]
    assert list(X_tr.columns) == list(X_te.columns)
    assert not X_tr.isnull().any().any()
    assert not X_te.isnull().any().any()

    vif_df = compute_vif_table(X_tr[["person_age", "loan_int_rate", "loan_percent_income"]])
    assert len(vif_df) == 3
    assert (vif_df["vif"] >= 1.0).all()

