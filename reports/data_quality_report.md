# Data Quality & Preprocessing Audit Report
**Project:** Axlero Solutions — Credit Risk Scoring and Loan Default Analysis  
**Original Preprocessing Contributor:** Darshan Patil (Member 1, Commit `48d20e6`, PR #1)  
**Integration & Audit Lead:** Pujan Sonani

---

## 1. Executive Summary
The raw credit risk dataset (`data/raw/credit_risk_dataset.csv`) comprises **32,581 records and 12 columns**. Following exact deduplication, domain anomaly invalidation, and missing-value imputation, the final validated dataset (`data/processed/final_validated_dataset.csv`) contains **32,416 records and 12 columns** with **0 missing values** and **0 duplicate rows**.

---

## 2. Resolution of Duplicate Record Count Discrepancies
During the audit of `notebooks/data_preprocessing_validation.ipynb`, two different duplicate counts appeared in consecutive cells:
- **Cell 3 (`df.duplicated(keep='first').sum()`):** `165`
- **Cell 4 (`len(df[df.duplicated(keep=False)])`):** `330`

**Resolution:** There is no contradiction between the two numbers:
- `165` is the number of **redundant duplicate copies** removed by `df.drop_duplicates(keep='first')`, reducing the row count from `32,581` to `32,416`.
- `330` is the total number of rows participating in those `165` identical pairs (`165` retained first occurrences + `165` dropped second occurrences = `330`).

---

## 3. Invalid Domain Records & Extreme Boundary Values
1. **Impossible Ages (`person_age > 100`):** 5 records (`144, 144, 123, 123, 144` years at raw indices `81, 183, 575, 747, 32297`) were flagged as data-entry errors, set to `NaN`, and imputed using the training/portfolio median age (`26.0` years).
2. **Impossible Employment Tenure (`person_emp_length > 60`):** 2 records (`123.0` years for applicants aged `22` and `21` at raw indices `0` and `210`) were flagged as invalid, set to `NaN`, and imputed with the median employment tenure (`4.0` years).
3. **Remaining Extreme Values Flagged for Feature Engineering:**
   - 2 applicants with `80 < person_age <= 100` (ages `84` and `94`).
   - 1 applicant with `person_emp_length = 41.0` years (aged `78`).
   - 9 applicants with `person_income > 1,000,000` (maximum `6,000,000`).
   - In `src/features/feature_engineering.py`, these extreme tail observations are winsorized/capped (`person_age <= 80`, `person_emp_length <= 40`, and `person_income` capped at the 99.5th training percentile alongside `log1p` transformation) fitted strictly on the training split.

---

## 4. Missing-Value Treatment & Imputation Benchmark
### 4.1 Missing Value Counts Before vs. After Deduplication
| Column | Raw Missing (`n = 32,581`) | Post-Deduplication Missing (`n = 32,416`) | Invalid Values Set to `NaN` | Total Imputed (`n = 32,416`) |
| :--- | ---: | ---: | ---: | ---: |
| `person_age` | 0 (0.00%) | 0 (0.00%) | +5 (`> 100` yrs) | **5** (0.015%) |
| `person_emp_length` | 895 (2.75%) | 887 (2.74%) | +2 (`> 60` yrs) | **889** (2.742%) |
| `loan_int_rate` | 3,116 (9.56%) | 3,095 (9.55%) | 0 | **3,095** (9.548%) |

*(Note: 8 of the 165 removed duplicate rows had missing `person_emp_length`, and 21 had missing `loan_int_rate`.)*

### 4.2 Quantitative Comparison of Imputation Strategies
To evaluate **Global Median**, **Loan-Grade Stratified Median**, and **Multivariate KNN Imputation (`k=5`, distance-weighted)** without data leakage, `1,500` complete validation records were masked and reconstructed using parameters fitted exclusively on the remaining training pool:

| Imputation Strategy | `loan_int_rate` RMSE (%) | `loan_int_rate` MAE (%) | `person_emp_length` RMSE (Yrs) | `person_emp_length` MAE (Yrs) |
| :--- | ---: | ---: | ---: | ---: |
| **Global Median Imputation** | 3.2917 | 2.7365 | 4.2409 | 3.1707 |
| **Loan-Grade Median Imputation (Selected)** | **1.0184** | **0.7976** | **4.2409** | **3.1707** |
| **Multivariate KNN Imputation (`k=5`)** | 1.1338 | 0.8714 | 4.2883 | 3.2788 |

**Selection Justification:** Because lender interest rates are heavily stratified by `loan_grade` (grade medians rise monotonically from `7.49%` in Grade A to `20.16%` in Grade G), **Loan-Grade Stratified Median Imputation** reduces interest rate reconstruction RMSE by **69.1%** compared to Global Median (`1.0184%` vs `3.2917%`) and outperforms 5-NN imputation (`1.1338%`) while preserving 100% determinism and compatibility with Member 1's validated dataset. Both `grade_median` and `knn` are supported in `CreditRiskImputer`.

---

## 5. Root-Cause Investigation of the 1,005 `loan_percent_income` Mismatches
Comparing `round(loan_amnt / person_income, 2)` against the recorded `loan_percent_income` column across all `32,416` records yields:
- **Exact Matches:** `31,411` records (`96.90%`)
- **Mismatches:** `1,005` records (`3.10%`)

Our root-cause decomposition proved that **100% of the 1,005 mismatches** are explained by two deterministic mechanisms:

| Mismatch Category | Record Count | Share of Mismatches | Mathematical Mechanism & Explanation |
| :--- | ---: | ---: | :--- |
| **Category 1: Decimal Rounding / Truncation Boundary (`\|diff\| <= 0.01`)** | **618** | 61.49% | Differences of exactly `0.01` caused by floor truncation or round-half-up vs. IEEE-754 round-half-to-even at the second decimal place (e.g., `1,500 / 12,000 = 0.125 -> 0.13` vs `0.12`). |
| **Category 2: `0.85x` Post-Deduction Net Income Scaling (`\|diff\| > 0.01`)** | **387** | 38.51% | In all 387 records where `\|diff\| > 0.01`, `person_income` equals exactly `0.85 × gross_income` (a 15% tax/withholding deduction on a round gross salary), while `loan_percent_income` was computed against pre-deduction `gross_income` (e.g., `person_income = 54,400 = 0.85 × 64,000`, `loan_amnt = 35,000`, and `35,000 / 64,000 = 0.5469 -> 0.55`). |
| **Unexplained / Other** | **0** | 0.00% | Zero unexplained discrepancies remain. |

In our feature engineering pipeline (`src/features/feature_engineering.py`), we preserve `loan_percent_income`, compute the exact `calc_loan_to_income = loan_amnt / person_income`, and include a binary `ratio_discrepancy_flag` to capture the 387 net-income-scaled records.

