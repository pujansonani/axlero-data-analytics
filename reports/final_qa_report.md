# Final Quality Assurance (QA) & Independent Verification Report
**Project:** Axlero Solutions — Credit Risk Scoring and Loan Default Analysis  
**Repository:** `https://github.com/pujansonani/axlero-data-analytics`  
**Branch:** `Main-TEAM-LEAD-ONLY`  
**QA Execution Date:** November 2, 2026  
**Overall Verification Status:** **COMPLETE — Independently Executed and Verified**

---

## 1. Executive QA Summary

An independent Phase-2 quality assurance audit was conducted across the entire repository, covering:
1. **Static & AST Code Quality Audit** across all Python modules (`src/`, `app/`, `tests/`) and Jupyter notebooks (`notebooks/`).
2. **End-to-End Pipeline Execution** (`PYTHONPATH=. python src/run_pipeline.py`) from immutable raw data (`32,581 × 12`) to serialized model bundles and 16 visual figures.
3. **Sequential Execution of All 9 Notebooks** (`01_data_understanding.ipynb` through `08_model_evaluation.ipynb` plus `data_preprocessing_validation.ipynb`) with working directory set inside `notebooks/` to verify portable path resolution.
4. **Automated Unit & Integration Testing** (`pytest -q` and `pytest -v`, `13 / 13` tests passing).
5. **Interactive Streamlit Dashboard QA** (`app/streamlit_app.py`) across all 8 pages, including filter interactions, preset buttons, valid borrower scoring with local SHAP attributions, and invalid borrower input validation.
6. **Git Provenance & Pull Request Audit** for PR #1 (`origin/Darshan`) and PR #2 (`origin/Sriram`).

---

## 2. Issues Identified & Fixed During Phase-2 QA

| Issue ID | Severity | Component | Description of Issue Found | Fix Implemented & Verified |
| :--- | :--- | :--- | :--- | :--- |
| **QA-01** | **High (Methodology)** | `src/evaluation/evaluate_model.py` | `evaluate_all_models()` previously sorted the model comparison table by `test_roc_auc` before `run_pipeline.py` selected `eval_df.iloc[0]` as the champion model, risking subtle test-set selection bias. | Changed sorting in `evaluate_all_models()` to rank strictly by `["val_roc_auc", "cv_roc_auc_mean"]` (validation set ROC-AUC with 5-Fold CV ROC-AUC tie-breaker). Verified that `Gradient Boosting (Tuned)` is genuinely #1 on both 5-Fold CV (`0.9442`) and Validation ROC-AUC (`0.9469`). |
| **QA-02** | **Medium (Completeness)** | `src/models/train_model.py` | `tune_hyperparameters_cv()` tuned Logistic Regression, Decision Tree, Random Forest, and XGBoost via 5-Fold Stratified CV, but omitted `GradientBoostingClassifier` from the recorded CV search dictionary. | Added 5-Fold Stratified CV hyperparameter search for `GradientBoostingClassifier` (`best_cv_roc_auc = 0.9442`) and recorded per-class `classification_report` metrics (`test_macro_avg_f1`, `test_weighted_avg_f1`) in `all_models_summary.json`. |
| **QA-03** | **Medium (Consistency)** | `src/preprocessing/cleaner.py` | `CreditRiskImputer.transform()` masked `person_age > 100` and `person_emp_length > 60` as `NaN`, whereas `CreditRiskImputer.fit()` assumed its input had already been invalidated; also IEEE-754 float subtraction `0.38 - 0.37 = 0.010000000000000009` required a `1e-6` tolerance at the `0.01` rounding boundary. | Added explicit invalidation (`person_age > 100 -> NaN`, `person_emp_length > 60 -> NaN`) inside `CreditRiskImputer.fit()` and used `<= 0.01 + 1e-6` in `investigate_loan_to_income_mismatches()` so all `618` rounding-boundary and `387` `0.85x` net-income records classify deterministically. |
| **QA-04** | **Medium (Portability)** | `notebooks/data_preprocessing_validation.ipynb` | When opened directly inside `notebooks/` in JupyterLab (where `cwd` is `notebooks/`), relative paths `"data/raw/..."` would fail without moving up to the repository root. | Added working-directory normalization (`if Path.cwd().name == 'notebooks': os.chdir(Path.cwd().parent)`) in Cell 0 and verified execution from inside `notebooks/`. |
| **QA-05** | **Medium (Dashboard QA)** | `app/streamlit_app.py` | Sidebar and Page 1 / Page 8 summary cards used static string literals (`"32,416"`, `"21.87%"`, `"0.9553"`, `"0.39"`), unclosed `open()` file handles, and default macOS GUI Matplotlib backend (causing `RuntimeError` when Page 8 renders SHAP plots in worker threads). | Replaced all static metric strings with values computed dynamically from `df_raw`, `df_val`, `imp_data`, `model_data`, and `thresh_data`, added `with open(...)` context managers, configured `matplotlib.use("Agg")` + `plt.close(fig)`, and widened Page 8 input bounds so users can interactively trigger domain validation errors. |
| **QA-06** | **Low (Code Hygiene)** | Multiple `.py` files | AST audit identified 12 unused imports across 10 files (`SimpleImputer`, `WoEIVAnalyzer`, `joblib`, `Optional`, `Any`, `List`, `GRID`, `np`, `pd`), a dead `pd.qcut` assignment in `src/features/woe_iv.py`, a scikit-learn 1.8 `penalty='l1'` deprecation warning, and a matplotlib `fontweight='medium'` warning. | Removed all unused imports and dead code, updated L1 logistic regression to `l1_ratio=1.0`, and set `fontweight='normal'`. Confirmed 0 unused imports across the entire repository via AST check. |

---

## 3. Test Suite Execution Results (`pytest`)

Command executed:
```bash
PYTHONPATH=. .venv/bin/python -m pytest -v
```
- **Total Tests Executed:** `13`
- **Passed:** `13` (`100%`)
- **Failed:** `0`
- **Execution Time:** `~3.8s`

| Test File | Test Function | Purpose | Status |
| :--- | :--- | :--- | :--- |
| `tests/test_preprocessing.py` | `test_raw_and_validated_dataset_shapes` | Verify raw (`32,581 × 12`) and validated (`32,416 × 12`) shapes, 0 nulls, 0 duplicates | **PASSED** |
| `tests/test_preprocessing.py` | `test_deduplication_and_invalidation_exact_counts` | Verify `165` (`keep='first'`) vs `330` (`keep=False`) duplicates, `5` age `>100`, `2` emp `>60`, and exact match with Member 1's CSV | **PASSED** |
| `tests/test_preprocessing.py` | `test_loan_to_income_mismatch_decomposition` | Verify `1,005` mismatches = `618` rounding boundary (`<=0.01`) + `387` `0.85x` net-income scaled (`0` unexplained) | **PASSED** |
| `tests/test_preprocessing.py` | `test_imputers_no_leakage` | Verify `CreditRiskImputer` (`grade_median`, `global_median`, `knn`) fits on train only with 0 nulls on validation | **PASSED** |
| `tests/test_preprocessing.py` | `test_validate_schema_raises_on_empty_or_missing` | Verify schema guard raises `ValueError` on empty or malformed DataFrames | **PASSED** |
| `tests/test_woe_iv_features.py` | `test_statistical_analysis_outputs` | Verify Mann-Whitney U, KS, Welch $t$, Chi-Square, Cramér's V, and level-wise Wilson CI tables | **PASSED** |
| `tests/test_woe_iv_features.py` | `test_woe_iv_analyzer_zero_bin_laplace_smoothing` | Verify Laplace smoothing ($\alpha=0.5$) prevents $\pm\infty$ or `NaN` on zero-event bins | **PASSED** |
| `tests/test_woe_iv_features.py` | `test_feature_engineering_fit_on_train_only` | Verify `CreditRiskFeatureEngineer` and VIF computation fit strictly on training data | **PASSED** |
| `tests/test_models_prediction.py` | `test_validate_borrower_input_valid_and_invalid` | Verify borrower input validator catches out-of-range age, `emp_length > age - 14`, and invalid categories | **PASSED** |
| `tests/test_models_prediction.py` | `test_saved_pipeline_loads_and_predicts_accurately` | Verify `credit_risk_pipeline.joblib` loads, scores low/high risk profiles accurately, and produces identical repeatable SHAP outputs | **PASSED** |
| `tests/test_models_prediction.py` | `test_streamlit_dashboard_all_8_pages_and_invalid_input` | Execute all 8 Streamlit pages headlessly via `AppTest` and verify invalid borrower input error handling on Page 8 | **PASSED** |
| `tests/test_models_prediction.py` | `test_no_hardcoded_local_paths_in_code_or_notebooks` | Scan all `.py` and `.ipynb` files to guarantee zero hardcoded local paths | **PASSED** |
| `tests/test_models_prediction.py` | `test_all_16_figures_and_reports_exist` | Verify all 16 PNG charts (`fig01`–`fig16`) and markdown reports exist and are non-empty | **PASSED** |

---

## 4. Sequential Notebook Execution Status

All 9 notebooks in `notebooks/` were executed top-to-bottom from inside `notebooks/` using `nbclient`:

| Order | Notebook File | Cells | Execution Status | Key Verified Outputs |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `01_data_understanding.ipynb` | 6 | **PASSED (0 errors)** | Raw shape `(32581, 12)`, `895` missing `person_emp_length`, `3,116` missing `loan_int_rate`, `165` (`keep='first'`) vs `330` (`keep=False`) duplicates |
| 2 | `02_data_preprocessing.ipynb` | 6 | **PASSED (0 errors)** | Deduplicated `(32416, 12)`, Imputation RMSE (`Grade Median = 1.0184%`, `KNN = 1.1338%`, `Global Median = 3.2917%`), `1,005` mismatches (`618 + 387`) |
| 3 | `03_exploratory_data_analysis.ipynb` | 6 | **PASSED (0 errors)** | Regenerated `fig01`–`fig09`, verified `21.87%` default rate and `77.8%` low-income × high-burden interaction |
| 4 | `04_statistical_analysis.ipynb` | 4 | **PASSED (0 errors)** | Mann-Whitney U, KS, Welch $t$, Chi-Square, Cramér's V (`loan_grade V = 0.4152`), 95% Wilson CIs, Odds Ratios |
| 5 | `05_woe_iv_analysis.ipynb` | 4 | **PASSED (0 errors)** | Train-only Laplace-smoothed WoE/IV tables (`loan_percent_income IV = 0.8761`, `loan_grade IV = 0.8583`) |
| 6 | `06_feature_engineering.ipynb` | 5 | **PASSED (0 errors)** | 14 engineered domain features, VIF multicollinearity reduction, 26 selected features |
| 7 | `07_model_training.ipynb` | 4 | **PASSED (0 errors)** | Stratified 70/15/15 split (`22,691 / 4,862 / 4,863`), 5-Fold CV hyperparameter tuning across 5 model families |
| 8 | `08_model_evaluation.ipynb` | 6 | **PASSED (0 errors)** | Held-out test evaluation across 8 models, threshold optimization (`0.50`, `0.39`, `0.16`), global & local SHAP |
| 9 | `data_preprocessing_validation.ipynb` | 32 | **PASSED (0 errors)** | Member 1 (Darshan Patil)'s original validation workflow with portable relative paths |

---

## 5. Independently Verified Model Comparison & Champion Justification

All 8 models were trained strictly on `X_train` (`n = 22,691`), ranked by `Validation ROC-AUC` (`n = 4,862`), and evaluated once on the untouched held-out `Test Set` (`n = 4,863`):

| Rank (by Val AUC) | Model | 5-Fold CV ROC-AUC | Val ROC-AUC | Val PR-AUC | Test Accuracy | Test Precision | Test Recall | Test Specificity | Test F1 | Test Macro F1 | Test ROC-AUC | Test PR-AUC | Test KS | Test Brier | Confusion Matrix (`TN, FP, FN, TP`) |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **1** | **Gradient Boosting (Tuned) [Champion]** | **0.9442 ± 0.0061** | **0.9469** | **0.9073** | **0.9439** | 0.9759 | 0.7622 | 0.9947 | **0.8559** | **0.9102** | **0.9553** | **0.9199** | **0.7891** | **0.0466** | `3779, 20, 253, 811` |
| 2 | **XGBoost (Champion Tuned)** | 0.9438 ± 0.0062 | 0.9463 | 0.9045 | 0.9422 | **0.9769** | 0.7538 | **0.9950** | 0.8509 | 0.9073 | 0.9547 | 0.9182 | 0.7818 | 0.0473 | `3780, 19, 262, 802` |
| 3 | **XGBoost (Balanced)** | 0.9441 ± 0.0068 | 0.9451 | 0.9034 | 0.9350 | 0.8864 | 0.8064 | 0.9710 | 0.8445 | 0.9016 | 0.9529 | 0.9165 | 0.7834 | 0.0576 | `3689, 110, 206, 858` |
| 4 | **Random Forest (Tuned)** | 0.9239 ± 0.0077 | 0.9232 | 0.8759 | 0.9297 | 0.9011 | 0.7622 | 0.9766 | 0.8259 | 0.8908 | 0.9355 | 0.8946 | 0.7549 | 0.0661 | `3710, 89, 253, 811` |
| 5 | **Decision Tree (Tuned)** | 0.9060 ± 0.0105 | 0.9041 | 0.8473 | 0.9321 | 0.9520 | 0.7265 | 0.9897 | 0.8241 | 0.8909 | 0.9154 | 0.8626 | 0.7184 | 0.0591 | `3760, 39, 291, 773` |
| 6 | **Logistic Regression (Balanced)** | 0.8990 ± 0.0069 | 0.8997 | 0.7876 | 0.8468 | 0.6129 | **0.8139** | 0.8560 | 0.6992 | 0.7991 | 0.9017 | 0.7834 | 0.6895 | 0.1173 | `3252, 547, 198, 866` |
| 7 | **Logistic Regression (L2)** | 0.8968 ± 0.0073 | 0.8972 | 0.7997 | 0.8888 | 0.8030 | 0.6513 | 0.9553 | 0.7193 | 0.8248 | 0.8997 | 0.7983 | 0.6920 | 0.0856 | `3629, 170, 371, 693` |
| 8 | **Dummy Baseline (Stratified)** | 0.5046 ± 0.0080 | 0.4961 | 0.2173 | 0.6572 | 0.2153 | 0.2143 | 0.7813 | 0.2148 | 0.4979 | 0.4978 | 0.2180 | 0.0000 | 0.3428 | `2968, 831, 836, 228` |

### Verified Decision Threshold Regimes on Champion Model (`Gradient Boosting (Tuned)`)
- **Standard Cutoff (`0.50`):** Accuracy = `94.39%`, Precision = `97.59%`, Recall = `76.22%`, F1 = `0.8559` (`TP = 811, FP = 20, FN = 253, TN = 3,779`).
- **Validation F1-Optimal Cutoff (`0.39`):** Accuracy = `94.41%`, Precision = `95.00%`, Recall = `78.57%`, F1 = `0.8601` (`TP = 836, FP = 44, FN = 228, TN = 3,755`).
- **Validation 5:1 Cost-Optimal Cutoff (`0.16`):** Accuracy = `90.11%`, Precision = `73.15%`, Recall = `86.56%`, F1 = `0.7929` (`TP = 921, FP = 338, FN = 143, TN = 3,461`).

---

## 6. Pull Request & Branch Integration Assessment (PR #1 & PR #2)
- **PR #1 (`origin/Darshan` — `48d20e6` by Darshan Patil):** Added `data/processed/final_validated_dataset.csv` (`32,416 × 12`), `notebooks/data_preprocessing_validation.ipynb`, and `reports/data_quality_checklist.md`.
  - **Audit Status:** **100% Integrated & Improved** in `Main-TEAM-LEAD-ONLY`. `data/processed/final_validated_dataset.csv` and `reports/data_quality_checklist.md` are preserved byte-for-byte, while `notebooks/data_preprocessing_validation.ipynb` has been upgraded to use portable relative paths (`data/raw/credit_risk_dataset.csv`) instead of hardcoded Windows OneDrive paths (`C:\Users\darsh\OneDrive\...`).
- **PR #2 (`origin/Sriram` — `77a35ff` by Srirami Reddy Ailuri):** Added `member2_eda_visualization.ipynb`, `make_figures.py`, `EDA_observations.md`, and `fig01`–`fig09` inside a doubly-nested `member2_analysis_visualization/member2_analysis_visualization/` directory along with a duplicate copy of `final_validated_dataset.csv`.
  - **Audit Status:** **100% Integrated & Improved** in `Main-TEAM-LEAD-ONLY`. All 9 figures (`reports/figures/fig01_loan_status.png` – `fig09_correlation_heatmap.png`), plotting logic (`src/visualization/eda_plots.py`), and observations (`notebooks/03_exploratory_data_analysis.ipynb`, `reports/eda_report.md`) are integrated into the canonical repository structure without the duplicate nested folder.
- **Recommendation for PR #1 & PR #2:**
  - **Recommended Action:** **Close PR #1 and PR #2 with an integration comment** referencing `Main-TEAM-LEAD-ONLY` (or run `git merge -s ours origin/Darshan origin/Sriram` if you wish GitHub to mark the PR badges as `Merged` without re-introducing the hardcoded Windows paths from PR #1 or the nested `member2_analysis_visualization/member2_analysis_visualization/` directory from PR #2). Do **not** perform a naive default GitHub UI merge of PR #1 or PR #2, as that would re-introduce the nested duplicate directory and overwrite the portable notebook paths.

---

## 7. Remaining Limitations & Final Readiness Assessment
- **Remaining Limitations:** As documented in [docs/limitations.md](../docs/limitations.md), historical observational records establish statistical association rather than causation, include only originated loans (survivorship/reject bias), include endogenous lender-set variables (`loan_grade`, `loan_int_rate`), and have small sample sizes in `loan_grade = G` (`n = 64`) and `person_home_ownership = OTHER` (`n = 106`).
- **Final Readiness Assessment:** **COMPLETE & SUBMISSION-READY.**
