# Technical Methodology — Credit Risk Scoring & Loan Default Analysis

## 1. End-to-End Leakage-Free Pipeline Design
A critical flaw in many credit risk studies is **data leakage**—computing imputation medians, outlier caps, Weight of Evidence (WoE) bins, or feature scalers on the entire dataset before splitting into train and test sets. Our methodology strictly enforces **train-only fitting**:

1. **Stage 1 — Exact Deduplication & Domain Anomaly Flagging (`src/preprocessing/cleaner.py`):**
   - Exact row duplicates (`165` redundant copies) are removed (`32,581 -> 32,416` rows).
   - Impossible values (`person_age > 100` [5 rows] and `person_emp_length > 60` [2 rows]) are converted to `NaN` prior to splitting in `data/interim/cleaned_credit_risk.csv`.
2. **Stage 2 — Stratified 70 / 15 / 15 Partitioning (`src/data/load_data.py`):**
   - `df_train`: `22,691` rows (`70.0%`, default rate `21.87%`)
   - `df_val`: `4,862` rows (`15.0%`, default rate `21.86%`)
   - `df_test`: `4,863` rows (`15.0%`, default rate `21.88%`)
3. **Stage 3 — Missing-Value Imputation (`CreditRiskImputer`):**
   - Benchmarked Global Median, Loan-Grade Stratified Median, and Multivariate KNN Imputation (`k=5`, distance-weighted).
   - Loan-Grade Median (`loan_int_rate` filled via training grade median; `person_emp_length` and `person_age` via training median) reduces `loan_int_rate` reconstruction RMSE by **69.1%** (`1.0184%` vs `3.2917%`).
4. **Stage 4 — Parametric & Non-Parametric Statistical Testing (`src/features/statistical_tests.py`):**
   - Numerical features: Welch's two-sample $t$-test, Mann-Whitney $U$, Rank-Biserial correlation ($r_{rb}$), Two-Sample Kolmogorov-Smirnov ($D_{KS}$), Spearman ($r_s$), and Point-Biserial ($r_{pb}$) correlations.
   - Categorical features: Pearson's Chi-Square ($\chi^2$), Cramér's $V$, 95% Wilson Score Confidence Intervals, and level-wise Odds Ratios.
5. **Stage 5 — Weight of Evidence (WoE) & Information Value (IV) (`src/features/woe_iv.py`):**
   - Continuous features are binned into deciles (`q = 10`) on `df_train`.
   - Zero-frequency bins are protected via **Laplace smoothing ($\alpha = 0.5$)**:
     $$\text{WoE}_i = \ln\!\left(\frac{(E_i + 0.5) / \sum_j (E_j + 0.5)}{(NE_i + 0.5) / \sum_j (NE_j + 0.5)}\right), \qquad \text{IV} = \sum_{i=1}^{k} (\text{Dist}_{E,i} - \text{Dist}_{NE,i}) \cdot \text{WoE}_i$$
6. **Stage 6 — Domain Feature Engineering & VIF Selection (`src/features/feature_engineering.py`):**
   - Constructs `14` domain features (`calc_loan_to_income`, `ratio_discrepancy_flag`, `log_person_income`, `log_loan_amnt`, `est_annual_debt_service`, `debt_service_to_income`, `log_residual_income`, `emp_to_working_age_ratio`, `cred_hist_to_working_age_ratio`, `loan_grade_num`, `high_burden_flag`, `subprime_grade_flag`, `grade_x_burden`, `int_rate_spread_over_grade`).
   - Audits Variance Inflation Factor (VIF) and ranks features via a composite of ANOVA $F$, Mutual Information, L1 Logistic Regression, and Random Forest Gini importance.
7. **Stage 7 — Cross-Validated Model Training, Threshold Optimization & SHAP (`src/models/` & `src/evaluation/`):**
   - Compares 8 models tuned via 5-Fold Stratified Cross-Validation on `df_train`.
   - Optimizes decision thresholds on `df_val` under both **F1 maximization (`0.39`)** and **5:1 asymmetric credit loss minimization (`0.16`)**.
   - Computes global and single-borrower local **SHAP (`TreeExplainer`)** attributions.

