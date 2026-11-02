"""Unit tests for data loading, deduplication, invalid record handling, and imputation."""

import pandas as pd
import pytest

from src.data.load_data import (
    load_raw_data,
    load_validated_data,
    stratified_train_val_test_split,
    validate_schema,
)
from src.preprocessing.cleaner import (
    CreditRiskImputer,
    analyze_loan_to_income_mismatches,
    apply_member1_imputation,
    deduplicate_and_invalidate,
)


def test_raw_and_validated_dataset_shapes():
    df_raw = load_raw_data()
    df_val = load_validated_data()
    assert df_raw.shape == (32581, 12)
    assert df_val.shape == (32416, 12)
    assert df_val.isnull().sum().sum() == 0
    assert df_val.duplicated().sum() == 0


def test_deduplication_and_invalidation_exact_counts():
    df_raw = load_raw_data()
    df_interim, audit = deduplicate_and_invalidate(df_raw)
    assert audit["raw_rows"] == 32581
    assert audit["dedup_rows"] == 32416
    assert audit["duplicates_removed_keep_first"] == 165
    assert audit["total_rows_in_duplicate_pairs_keep_false"] == 330
    assert audit["invalid_records_flagged"]["person_age_gt_100_count"] == 5
    assert audit["invalid_records_flagged"]["person_emp_length_gt_60_count"] == 2
    assert audit["post_invalidation_missing"]["person_age"] == 5
    assert audit["post_invalidation_missing"]["person_emp_length"] == 889
    assert audit["post_invalidation_missing"]["loan_int_rate"] == 3095

    # Verify Member 1 exact reproduction
    df_reproduced = apply_member1_imputation(df_interim)
    df_val = load_validated_data()
    pd.testing.assert_frame_equal(df_reproduced.reset_index(drop=True), df_val.reset_index(drop=True))


def test_loan_to_income_mismatch_decomposition():
    df_val = load_validated_data()
    mismatch = analyze_loan_to_income_mismatches(df_val)
    assert mismatch["total_records"] == 32416
    assert mismatch["exact_round2_matches"] == 31411
    assert mismatch["total_mismatches"] == 1005
    decomp = mismatch["decomposition"]
    assert decomp["rounding_or_truncation_boundary_le_0_01"] == 618
    assert decomp["net_income_0_85_scaled_denominator_gt_0_01"] == 387
    assert decomp["unexplained_other"] == 0


def test_imputers_no_leakage():
    df_raw = load_raw_data()
    df_interim, _ = deduplicate_and_invalidate(df_raw)
    df_train, df_val, _ = stratified_train_val_test_split(df_interim.iloc[:2000])

    for strategy in ["grade_median", "global_median", "knn"]:
        imp = CreditRiskImputer(strategy=strategy, n_neighbors=5).fit(df_train)
        val_imputed = imp.transform(df_val)
        assert val_imputed.isnull().sum().sum() == 0
        assert (val_imputed["person_age"] <= 100).all()
        assert (val_imputed["person_emp_length"] <= 60).all()


def test_validate_schema_raises_on_empty_or_missing():
    with pytest.raises(ValueError):
        validate_schema(pd.DataFrame())
    with pytest.raises(ValueError):
        validate_schema(pd.DataFrame({"person_age": [25]}))

