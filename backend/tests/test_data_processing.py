from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.data_processing import (
    FeatureCoercer,
    build_quality_report,
    infer_feature_types,
    load_csv,
    prepare_training_data,
    resolve_target,
)
from app.services.errors import DataValidationError, InsufficientDataError


def test_load_csv_parses_bytes_and_drops_index_column():
    df = load_csv(b"Unnamed: 0,a,b\n0,1,x\n1,2,y\n")
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


@pytest.mark.parametrize("content", [b"", b"   \n", b"a,b\n"])
def test_load_csv_rejects_empty(content):
    with pytest.raises(DataValidationError):
        load_csv(content)


def test_load_csv_rejects_duplicate_columns():
    # pandas renames duplicates to a / a.1, so build a case-collision after stripping.
    with pytest.raises(DataValidationError):
        load_csv(b"a , a\n1,2\n")


def test_resolve_target_with_explicit_labels(synthetic_df):
    y, idx, mapping = resolve_target(synthetic_df, "loan_outcome", ["Defaulted"])
    assert set(y.unique()) == {0, 1}
    assert mapping.positive_labels == ["defaulted"]
    assert y.sum() == (synthetic_df["loan_outcome"] == "Defaulted").sum()


def test_resolve_target_infers_common_labels():
    df = pd.DataFrame({"t": ["Charged Off", "Fully Paid", "Fully Paid"], "x": [1, 2, 3]})
    y, _, mapping = resolve_target(df, "t")
    assert y.tolist() == [1, 0, 0]
    assert mapping.positive_labels == ["charged off"]


def test_resolve_target_numeric_01_and_missing_target_rows():
    df = pd.DataFrame({"t": [1, 0, np.nan, 0], "x": [1, 2, 3, 4]})
    y, idx, mapping = resolve_target(df, "t")
    assert y.tolist() == [1, 0, 0]
    assert mapping.dropped_rows_missing_target == 1


def test_resolve_target_errors():
    df = pd.DataFrame({"t": ["A", "B", "C"], "x": [1, 2, 3]})
    with pytest.raises(DataValidationError, match="not found"):
        resolve_target(df, "missing")
    with pytest.raises(DataValidationError, match="infer"):
        resolve_target(df, "t")
    with pytest.raises(DataValidationError, match="more than two labels"):
        resolve_target(df, "t", ["A"])
    with pytest.raises(DataValidationError, match="None of the positive_labels"):
        resolve_target(df, "t", ["Z"])


def test_resolve_target_negative_labels_drop_ambiguous_rows():
    df = pd.DataFrame({"t": ["Default", "Paid", "Current", "Paid"]})
    y, idx, mapping = resolve_target(df, "t", ["Default"], ["Paid"])
    assert y.tolist() == [1, 0, 0]
    assert mapping.dropped_rows_unmapped_label == 1
    assert mapping.unmapped_labels == ["current"]


def test_infer_feature_types_handles_ids_leakage_text_numbers(synthetic_df):
    types = infer_feature_types(synthetic_df, "loan_outcome")
    assert "loan_id" in types.excluded
    assert "total_recovered_amount" in types.excluded
    assert "leakage" in types.excluded["total_recovered_amount"]
    assert "annual_income" in types.numeric  # stored as "38,000" text
    assert any(d["column"] == "annual_income" for d in types.dtype_issues)
    assert {"home_ownership", "loan_purpose", "employment_type"} <= set(types.categorical)
    assert "loan_outcome" not in types.numeric + types.categorical


def test_infer_feature_types_excludes_constant_and_user_columns():
    df = pd.DataFrame({"c": [1] * 30, "u": range(30), "v": np.arange(30) % 3, "t": [0, 1] * 15})
    types = infer_feature_types(df, "t", exclude_columns=["u"])
    assert types.excluded["c"] == "constant column"
    assert types.excluded["u"] == "excluded by user"
    assert types.numeric == ["v"]


def test_quality_report_detects_missing_duplicates_imbalance(synthetic_df):
    report = build_quality_report(synthetic_df, "loan_outcome", ["Defaulted"])
    assert report["duplicate_rows"] > 0
    assert "employment_length_years" in report["missing_values"]
    assert report["target"]["class_counts"]["default"] > 0
    assert report["target"]["is_imbalanced"] is True
    assert any(w["column"] == "total_recovered_amount" for w in report["leakage_warnings"])
    assert "loan_outcome" in report["suggested_target_columns"]


def test_quality_report_flags_statistical_leakage():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    df = pd.DataFrame({"leaky": y * 10 + rng.normal(0, 0.1, 200), "noise": rng.normal(size=200), "t": y})
    report = build_quality_report(df, "t")
    assert any(w["column"] == "leaky" for w in report["leakage_warnings"])
    assert not any(w["column"] == "noise" for w in report["leakage_warnings"])


def test_prepare_training_data_rejects_small_and_single_class():
    small = pd.DataFrame({"x": range(20), "t": [0, 1] * 10})
    with pytest.raises(InsufficientDataError, match="At least"):
        prepare_training_data(small, "t")
    one_class = pd.DataFrame({"x": range(100), "t": ["Default"] * 100})
    with pytest.raises(InsufficientDataError, match="only one class"):
        prepare_training_data(one_class, "t", positive_labels=["Default"])
    few_pos = pd.DataFrame({"x": range(100), "t": [1] * 5 + [0] * 95})
    with pytest.raises(InsufficientDataError, match="Each class"):
        prepare_training_data(few_pos, "t")


def test_prepare_training_data_removes_duplicates(synthetic_df):
    prepared = prepare_training_data(synthetic_df, "loan_outcome", ["Defaulted"])
    assert prepared.duplicates_removed == synthetic_df.duplicated().sum()
    assert "total_recovered_amount" not in prepared.X.columns
    assert "loan_outcome" not in prepared.X.columns


def test_feature_coercer_normalises_types_and_missing():
    fc = FeatureCoercer(["n"], ["c"])
    out = fc.fit_transform(pd.DataFrame({"n": ["1,200", None, "x"], "c": [None, "", True]}))
    assert out["n"].iloc[0] == 1200.0
    assert out["n"].iloc[1:].isna().all()
    assert pd.isna(out["c"].iloc[0]) and pd.isna(out["c"].iloc[1])
    assert out["c"].iloc[2] == "True"
    # Missing columns are added as NaN instead of failing.
    assert FeatureCoercer(["n"], ["c"]).transform(pd.DataFrame({"n": [1]}))["c"].isna().all()
