# FINAL STATUS REPORT — AXLERO CREDIT RISK ANALYTICS PROJECT

**Date:** November 2, 2026  
**Project:** Finance & Banking — Credit Risk Scoring and Loan Default Analysis  
**Repository:** `https://github.com/pujansonani/axlero-data-analytics`  
**Status:** **COMPLETE & VERIFIED END-TO-END**

---

## 1. Completed Modules Summary

| Module | Status | Key Deliverables Completed & Verified |
| :--- | :--- | :--- |
| **1. Data Understanding & Preprocessing** *(Darshan Patil + Integration Audit)* | **COMPLETED** | `src/data/load_data.py`, `src/preprocessing/cleaner.py`, `notebooks/01_data_understanding.ipynb`, `notebooks/02_data_preprocessing.ipynb`, `notebooks/data_preprocessing_validation.ipynb` (portable paths), `reports/data_quality_checklist.md`, `reports/data_quality_report.md`, `data/processed/final_validated_dataset.csv` (`32,416 × 12`). |
| **2. Exploratory Data Analysis** *(Srirami Reddy Ailuri + Integration Audit)* | **COMPLETED** | `src/visualization/eda_plots.py`, `notebooks/03_exploratory_data_analysis.ipynb`, `reports/eda_report.md`, `reports/figures/fig01_loan_status.png` through `fig09_correlation_heatmap.png`. |
| **3. Statistical Analysis & WoE/IV** *(Implemented from scratch for Module 3)* | **COMPLETED** | `src/features/statistical_tests.py`, `src/features/woe_iv.py`, `notebooks/04_statistical_analysis.ipynb`, `notebooks/05_woe_iv_analysis.ipynb`, `reports/statistical_analysis.md`, `reports/figures/fig10_woe_iv_summary.png`, `reports/figures/fig11_woe_curves.png`. |
| **4. Feature Engineering & Selection** *(Implemented from scratch for Module 4)* | **COMPLETED** | `src/features/feature_engineering.py` (`CreditRiskFeatureEngineer`, 14 domain features, train-only outlier caps & scaling, VIF multicollinearity audit, composite feature ranking), `notebooks/06_feature_engineering.ipynb`. |
| **5. Machine Learning, Tuning & Explainability** *(Implemented from scratch for Module 5)* | **COMPLETED** | `src/models/train_model.py`, `src/evaluation/evaluate_model.py`, `notebooks/07_model_training.ipynb`, `notebooks/08_model_evaluation.ipynb`, `reports/model_evaluation_report.md`, `reports/figures/fig12`–`fig16`, `models/credit_risk_pipeline.joblib`, `models/best_model.joblib`. |
| **6. Integration, Testing & Streamlit App** *(Pujan Sonani — Team Lead)* | **COMPLETED** | `src/run_pipeline.py`, `app/streamlit_app.py` (8 interactive pages), `tests/` (`13 / 13` unit & integration tests passing), `config/config.yaml`, `README.md`, `reports/final_project_report.md`, `reports/final_qa_report.md`, `docs/*.md`. |

---

## 2. Pending Modules
- **None.** All 6 team modules, 9 notebooks, 16 figures, 7 reports (including `reports/final_qa_report.md`), 5 documentation guides, serialized models, 13 automated tests, and the 8-page Streamlit application are complete and independently verified.

---

## 3. Bugs & Issues Identified and Fixed

1. **Hardcoded Windows OneDrive File Paths in Member 1's Notebook:**
   - *Issue:* `notebooks/data_preprocessing_validation.ipynb` contained hardcoded local paths (`r"C:\Users\darsh\OneDrive\Axlero-Data-Analytics-Project\..."`), causing `FileNotFoundError` on macOS/Linux or other machines.
   - *Fix:* Replaced all hardcoded paths with portable relative repository paths (`data/raw/credit_risk_dataset.csv`, `data/processed/final_validated_dataset.csv`, `reports/data_quality_checklist.md`) and centralized paths in `config/config.yaml`.
2. **Duplicate Record Count Discrepancy (`165` vs `330`):**
   - *Issue:* Consecutive cells in `data_preprocessing_validation.ipynb` printed `Total Duplicate Rows: 165` (`df.duplicated().sum()`) and `Total duplicate records: 330` (`df[df.duplicated(keep=False)]`).
   - *Fix:* Documented and tested that `165` represents the redundant duplicate copies removed (`32,581 - 165 = 32,416`), while `330` counts both the original and duplicate rows in those `165` identical pairs (`165 × 2 = 330`).
3. **Unexplained `1,005` `loan_percent_income` Mismatches:**
   - *Issue:* `round(loan_amnt / person_income, 2) != loan_percent_income` occurred on `1,005` rows (`387` with `|diff| > 0.01`).
   - *Fix:* Proved mathematically that `618` rows are `<= 0.01` decimal rounding/truncation boundary differences and all `387` remaining rows have `person_income = 0.85 × gross_income` (a 15% tax/deduction factor) where `loan_percent_income` was computed against pre-deduction gross income. Added `calc_loan_to_income` and `ratio_discrepancy_flag` to the feature engineering pipeline.
4. **Nested Duplicate Directory in Member 2's Branch (`origin/Sriram`):**
   - *Issue:* `origin/Sriram` uploaded files inside `member2_analysis_visualization/member2_analysis_visualization/` along with a duplicate copy of `final_validated_dataset.csv`.
   - *Fix:* Integrated figures cleanly into `reports/figures/`, plotting code into `src/visualization/eda_plots.py`, and analysis into `notebooks/03_exploratory_data_analysis.ipynb` and `reports/eda_report.md`.
5. **Prevention of Imputation, WoE, Scaling, and Model Selection Data Leakage:**
   - *Issue:* Imputing or scaling on the full dataset prior to train/test splitting leaks test distribution statistics into training; sorting models by `test_roc_auc` selects the champion on the held-out test set.
   - *Fix:* Partitioned `data/interim/cleaned_credit_risk.csv` via stratified `70/15/15` split *before* fitting `CreditRiskImputer`, `WoEIVAnalyzer`, and `CreditRiskFeatureEngineer` strictly on `df_train`. Updated `evaluate_all_models()` to sort strictly by `["val_roc_auc", "cv_roc_auc_mean"]` so the held-out test set (`n = 4,863`) is used exclusively for unbiased final reporting.
6. **Phase-2 Code Quality, Dynamic Dashboard & Cross-Working-Directory Fixes:**
   - *Fix:* Removed 12 unused imports across 10 files; added `GradientBoostingClassifier` CV tuning grid in `src/models/train_model.py`; added explicit invalidation (`person_age > 100 -> NaN`, `person_emp_length > 60 -> NaN`) inside `CreditRiskImputer.fit()`; replaced all hardcoded summary strings in `app/streamlit_app.py` with dynamic JSON/DataFrame calculations and non-interactive `Agg` Matplotlib backend; and ensured all 9 notebooks execute cleanly from both the repository root and `notebooks/`.

---

## 4. Dataset Statistics Summary
- **Raw Dataset:** `32,581` rows × `12` columns (`895` missing `person_emp_length`, `3,116` missing `loan_int_rate`, `165` exact duplicate rows).
- **Validated Dataset:** `32,416` rows × `12` columns (`0` missing values, `0` duplicates).
- **Target Distribution:** `25,327` Non-Default (`78.13%`) vs `7,089` Default (`21.87%`).
- **Stratified Splits:** Train = `22,691` (`70.0%`), Validation = `4,862` (`15.0%`), Held-Out Test = `4,863` (`15.0%`).

---

## 5. Model Comparison Results (Held-Out Test Set, `n = 4,863`)

| Rank | Model | CV ROC-AUC | Val ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **Gradient Boosting (Tuned)** | **0.9442** | **0.9469** | **0.9439** | 0.9759 | 0.7622 | **0.8559** | **0.9553** | **0.9199** |
| 2 | **XGBoost (Champion Tuned)** | 0.9438 | 0.9463 | 0.9422 | **0.9769** | 0.7538 | 0.8509 | 0.9547 | 0.9182 |
| 3 | **XGBoost (Balanced)** | 0.9441 | 0.9451 | 0.9350 | 0.8864 | 0.8064 | 0.8445 | 0.9529 | 0.9165 |
| 4 | **Random Forest (Tuned)** | 0.9239 | 0.9232 | 0.9297 | 0.9011 | 0.7622 | 0.8259 | 0.9355 | 0.8946 |
| 5 | **Decision Tree (Tuned)** | 0.9060 | 0.9041 | 0.9321 | 0.9520 | 0.7265 | 0.8241 | 0.9154 | 0.8626 |
| 6 | **Logistic Regression (Balanced)** | 0.8990 | 0.8997 | 0.8468 | 0.6129 | **0.8139** | 0.6992 | 0.9017 | 0.7834 |
| 7 | **Logistic Regression (L2)** | 0.8968 | 0.8972 | 0.8888 | 0.8030 | 0.6513 | 0.7193 | 0.8997 | 0.7983 |
| 8 | **Dummy Baseline (Stratified)** | 0.5046 | 0.4961 | 0.6572 | 0.2153 | 0.2143 | 0.2148 | 0.4978 | 0.2180 |

---

## 6. Reproducibility Instructions
```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run full pipeline from raw data to serialized models & figures
PYTHONPATH=. python src/run_pipeline.py

# 3. Run unit tests
PYTHONPATH=. python -m pytest -v

# 4. Launch Streamlit application
streamlit run app/streamlit_app.py
```

---

## 7. Known Limitations
- Observational credit data measure association, not causation.
- `loan_grade` and `loan_int_rate` reflect lender underwriting policy at origination.
- Dataset exhibits survivorship/reject bias (only originated loans are included).
- Subgroups `loan_grade = G` (`n = 64`) and `person_home_ownership = OTHER` (`n = 106`) have small sample sizes.

