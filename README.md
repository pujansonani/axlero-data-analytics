# Axlero Solutions — Credit Risk Scoring & Loan Default Analysis

**Domain:** Finance & Banking — Consumer Credit Risk Analytics & Machine Learning  
**Repository:** `https://github.com/pujansonani/axlero-data-analytics`

---

## 1. Project Overview & Problem Statement
Consumer lending institutions face an asymmetric credit risk decision on every loan application: approving a borrower who defaults (**False Negative**) results in substantial principal loss, whereas declining a creditworthy borrower (**False Positive**) forfeits only interest margin. Furthermore, because the historical loan portfolio exhibits a **21.87% default rate** (`3.57 : 1` class imbalance across `32,416` validated loans), naive accuracy optimization is inadequate for underwriting.

This repository provides a complete, end-to-end, reproducible **Credit Risk Scoring and Loan Default Analysis** platform spanning data quality auditing, imputation benchmarking, exploratory data analysis, parametric and non-parametric hypothesis testing, Weight of Evidence (WoE) & Information Value (IV) credit scorecard modeling, leakage-free domain feature engineering, cross-validated machine learning (`8` models), asymmetric cost-threshold optimization, TreeSHAP explainability, automated unit testing, and an interactive 8-page **Streamlit** dashboard.

---

## 2. Business Objectives
1. **Data Governance & Quality Assurance:** Reconcile duplicate record definitions (`165` redundant copies vs `330` involved rows), remediate impossible demographic values (`person_age > 100`, `person_emp_length > 60`), benchmark Global Median vs Loan-Grade Median vs KNN Imputation (`k=5`), and mathematically resolve all `1,005` `loan_percent_income` consistency mismatches.
2. **Evidence-Based Risk Factor Discovery:** Quantify univariate and multi-factor default patterns while rigorously separating observational association from causation and identifying lender-set policy variables (`loan_grade`, `loan_int_rate`).
3. **Credit Scorecard WoE / IV Ranking:** Rank predictors using Laplace-smoothed ($\alpha = 0.5$) Weight of Evidence and Information Value fitted strictly on the training split to prevent target leakage.
4. **High-Precision Probability of Default (PD) Modeling:** Train, tune, and evaluate interpretable and ensemble classifiers on a stratified `70% / 15% / 15%` split (`train = 22,691`, `val = 4,862`, `test = 4,863`), optimize decision thresholds for both F1 (`0.39`) and 5:1 asymmetric credit loss (`0.16`), and provide local borrower-level **SHAP** attributions.

---

## 3. Dataset Description & Schema
- **Raw Dataset (`data/raw/credit_risk_dataset.csv`):** `32,581` rows × `12` columns (standard public Credit Risk benchmark dataset).
- **Validated Dataset (`data/processed/final_validated_dataset.csv`):** `32,416` rows × `12` columns (`25,327` non-defaults [`78.13%`] and `7,089` defaults [`21.87%`], `0` missing values, `0` duplicates).

| Column Name | Data Type | Domain Description | Summary / Range (Validated) |
| :--- | :--- | :--- | :--- |
| `person_age` | Numeric (`float`) | Applicant age in years | `20` – `94` yrs (Median: `26.0`) |
| `person_income` | Numeric (`int`) | Annual income | `$4,000` – `$6,000,000` (Median: `$55,000`) |
| `person_home_ownership` | Categorical (`str`) | Housing tenure status | `RENT` (50.5%), `MORTGAGE` (41.2%), `OWN` (7.9%), `OTHER` (0.3%) |
| `person_emp_length` | Numeric (`float`) | Employment tenure in years | `0.0` – `41.0` yrs (Median: `4.0`) |
| `loan_intent` | Categorical (`str`) | Stated purpose of loan | `EDUCATION`, `MEDICAL`, `VENTURE`, `PERSONAL`, `DEBTCONSOLIDATION`, `HOMEIMPROVEMENT` |
| `loan_grade` | Categorical (`str`) | Lender underwriting risk grade | `A` (best, 33.0%) through `G` (worst, 0.2%) |
| `loan_amnt` | Numeric (`int`) | Requested loan principal | `$500` – `$35,000` (Median: `$8,000`) |
| `loan_int_rate` | Numeric (`float`) | Annual interest rate (%) | `5.42%` – `23.22%` (Median: `10.99%`) |
| `loan_status` | Binary Target (`int`) | `1` = Default, `0` = Repaid | `0`: `25,327` (`78.13%`), `1`: `7,089` (`21.87%`) |
| `loan_percent_income` | Numeric (`float`) | Stated loan-to-income ratio | `0.00` – `0.83` (Median: `0.15`) |
| `cb_person_default_on_file` | Categorical (`str`) | Prior default in credit bureau file | `N` (`82.36%`), `Y` (`17.64%`) |
| `cb_person_cred_hist_length` | Numeric (`int`) | Credit history length in years | `2` – `30` yrs (Median: `4.0`) |

---

## 4. Methodology & Technologies Used
- **Languages & Core Libraries:** Python 3.13, `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `streamlit`, `pyyaml`, `joblib`, `pytest`, `nbformat`, `nbclient`.
- **Leakage-Free Workflow:**
  `Raw Dataset (32,581)` → `Exact Deduplication & Anomaly Flagging (32,416)` → `Stratified 70/15/15 Split` → `Train-Only Imputation (Grade Median & KNN)` → `Train-Only WoE/IV (Laplace alpha=0.5)` → `14 Engineered Domain Features & VIF Audit` → `5-Fold Stratified CV Tuning (8 Models)` → `Threshold Optimization & TreeSHAP` → `Streamlit Prediction App`.

---

## 5. Repository Structure

```text
axlero-data-analytics/
├── README.md
├── FINAL_STATUS.md
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   ├── README.md
│   │   └── credit_risk_dataset.csv
│   ├── interim/
│   │   ├── README.md
│   │   └── cleaned_credit_risk.csv
│   └── processed/
│       ├── README.md
│       ├── final_validated_dataset.csv
│       ├── train_processed.csv
│       ├── val_processed.csv
│       └── test_processed.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_statistical_analysis.ipynb
│   ├── 05_woe_iv_analysis.ipynb
│   ├── 06_feature_engineering.ipynb
│   ├── 07_model_training.ipynb
│   ├── 08_model_evaluation.ipynb
│   └── data_preprocessing_validation.ipynb
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── run_pipeline.py
│   ├── data/
│   │   └── load_data.py
│   ├── preprocessing/
│   │   └── cleaner.py
│   ├── visualization/
│   │   └── eda_plots.py
│   ├── features/
│   │   ├── statistical_tests.py
│   │   ├── woe_iv.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   └── train_model.py
│   └── evaluation/
│       └── evaluate_model.py
├── reports/
│   ├── figures/                   # 16 reproducible PNG charts (fig01 - fig16)
│   ├── data_quality_checklist.md
│   ├── data_quality_report.md
│   ├── eda_report.md
│   ├── statistical_analysis.md
│   ├── model_evaluation_report.md
│   ├── final_project_report.md
│   └── final_qa_report.md
├── models/
│   ├── README.md
│   ├── credit_risk_pipeline.joblib
│   ├── best_model.joblib
│   ├── all_models_summary.json
│   ├── threshold_analysis.json
│   ├── woe_iv_summary.json
│   ├── statistical_summary.json
│   └── imputation_comparison.json
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_woe_iv_features.py
│   └── test_models_prediction.py
└── docs/
    ├── project_overview.md
    ├── methodology.md
    ├── team_contributions.md
    ├── limitations.md
    └── work-distribution.md
```

---

## 6. Installation & Execution Instructions

### 6.1 Environment Setup
```bash
git clone https://github.com/pujansonani/axlero-data-analytics.git
cd axlero-data-analytics

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 6.2 Run the Complete End-to-End Pipeline
```bash
PYTHONPATH=. python src/run_pipeline.py
```

### 6.3 Run the Automated Unit Test Suite
```bash
PYTHONPATH=. pytest -v
```

### 6.4 Execute or Inspect the Jupyter Notebooks
```bash
jupyter lab
```
All notebooks in `notebooks/01_data_understanding.ipynb` through `notebooks/08_model_evaluation.ipynb` are pre-executed and can be re-run sequentially from top to bottom.

### 6.5 Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 7. Model Evaluation Results (Held-Out Test Set, `n = 4,863`)

| Model | 5-Fold CV ROC-AUC | Val ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC | Test KS | Test Brier |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Gradient Boosting (Tuned) [Champion]** | **0.9442** | **0.9469** | **0.9439** | **0.9759** | 0.7622 | **0.8559** | **0.9553** | **0.9199** | **0.7891** | **0.0466** |
| **XGBoost (Champion Tuned)** | 0.9438 | 0.9463 | 0.9422 | **0.9769** | 0.7538 | 0.8509 | 0.9547 | 0.9182 | 0.7818 | 0.0473 |
| **XGBoost (Balanced)** | 0.9441 | 0.9451 | 0.9350 | 0.8864 | 0.8064 | 0.8445 | 0.9529 | 0.9165 | 0.7834 | 0.0576 |
| **Random Forest (Tuned)** | 0.9239 | 0.9232 | 0.9297 | 0.9011 | 0.7622 | 0.8259 | 0.9355 | 0.8946 | 0.7549 | 0.0661 |
| **Decision Tree (Tuned)** | 0.9060 | 0.9041 | 0.9321 | 0.9520 | 0.7265 | 0.8241 | 0.9154 | 0.8626 | 0.7184 | 0.0591 |
| **Logistic Regression (Balanced)** | 0.8990 | 0.8997 | 0.8468 | 0.6129 | **0.8139** | 0.6992 | 0.9017 | 0.7834 | 0.6895 | 0.1173 |
| **Logistic Regression (L2)** | 0.8968 | 0.8972 | 0.8888 | 0.8030 | 0.6513 | 0.7193 | 0.8997 | 0.7983 | 0.6920 | 0.0856 |
| **Dummy Baseline (Stratified)** | 0.5046 | 0.4961 | 0.6572 | 0.2153 | 0.2143 | 0.2148 | 0.4978 | 0.2180 | 0.0000 | 0.3428 |

- **F1-Optimal Threshold (`0.39`):** Accuracy = **94.41%**, Precision = **95.00%**, Recall = **78.57%**, F1-Score = **0.8601**.
- **5:1 Asymmetric Cost-Optimal Threshold (`0.16`):** Recall = **86.56%** (`921 / 1,064` defaults detected), Precision = **73.15%**.

---

## 8. Key Analytical Findings
1. **Exact Root Cause of the 1,005 `loan_percent_income` Mismatches:**
   - **618 records** differ by `<= 0.01` due to floor truncation / round-half-up vs round-half-to-even at the second decimal place.
   - **387 records** have `person_income = 0.85 × gross_income` (a 15% tax/net deduction) while `loan_percent_income` was recorded against pre-deduction `gross_income`.
2. **Imputation Benchmark:** Loan-grade stratified median imputation reduces `loan_int_rate` reconstruction RMSE by **69.1%** over global median (`1.0184%` vs `3.2917%`) and slightly outperforms 5-NN imputation (`1.1338%`).
3. **Top Risk Drivers (WoE/IV & SHAP):** Debt-to-income burden (`IV = 0.8761`, default rate **70.4%** when `> 30%`), loan grade (`IV = 0.8583`, default rate **61.2%** in Grades `D–G`), interest rate (`IV = 0.7038`), annual income (`IV = 0.4557`), and housing status (`IV = 0.3801`, renters default at **31.6%** vs **7.5%** for owners).

---

## 9. Limitations
See [docs/limitations.md](docs/limitations.md) for full details:
- Observational data show **association, not causation**.
- `loan_grade` and `loan_int_rate` are lender-assigned underwriting variables that partly encode the institution's prior credit assessment.
- Only approved/originated loans are observed (**reject/survivorship bias**).
- Small sample sizes in `loan_grade = G` (`n = 64`) and `person_home_ownership = OTHER` (`n = 106`) carry wider confidence intervals.

---

## 10. Team Contributions (Verified Git History vs. Integration Work)
See [docs/team_contributions.md](docs/team_contributions.md) for complete commit and PR provenance:

| Team Member | Assigned Module | Verified Prior Submission (Git Branch / PR) | Work Added / Enhanced During Integration |
| :--- | :--- | :--- | :--- |
| **Darshan Patil** | Member 1: Data Preprocessing | `origin/Darshan` (`48d20e6`, PR #1): `final_validated_dataset.csv`, `data_preprocessing_validation.ipynb`, `data_quality_checklist.md` | Portable paths, `165` vs `330` duplicate reconciliation, KNN vs Grade Median imputation benchmark, exact decomposition of `1,005` ratio mismatches (`01` & `02` notebooks, `src/preprocessing/cleaner.py`, `reports/data_quality_report.md`). |
| **Srirami Reddy Ailuri** | Member 2: Exploratory Data Analysis | `origin/Sriram` (`77a35ff`, PR #2): `member2_eda_visualization.ipynb`, `make_figures.py`, `EDA_observations.md`, `fig01`–`fig09` | Flattened directory structure, integrated into `src/visualization/eda_plots.py`, `notebooks/03_exploratory_data_analysis.ipynb`, and `reports/eda_report.md`. |
| **Charukesh** | Member 3: Statistical Analysis & WoE/IV | `origin/charukesh` (`96b19e5` — no commits beyond initial setup) | Implemented from scratch during integration: `src/features/statistical_tests.py`, `src/features/woe_iv.py`, `fig10`–`fig11`, `04` & `05` notebooks, `reports/statistical_analysis.md`. |
| **Pavani** | Member 4: Feature Engineering & Selection | `origin/pavani` (`96b19e5` — no commits beyond initial setup) | Implemented from scratch during integration: `src/features/feature_engineering.py` (14 domain features, train-only caps, VIF, feature selection), `06_feature_engineering.ipynb`. |
| **Sowmya** | Member 5: Machine Learning & Evaluation | `origin/sowmya` (`96b19e5` — no commits beyond initial setup) | Implemented from scratch during integration: `src/models/train_model.py`, `src/evaluation/evaluate_model.py`, `fig12`–`fig16`, `07` & `08` notebooks, `reports/model_evaluation_report.md`. |
| **Pujan Sonani** | Team Lead: Architecture & Integration | `Main-TEAM-LEAD-ONLY` (`96b19e5`, `65374c6`): Initial repo layout, `.gitignore`, `README.md`, `requirements.txt` | Full multi-branch integration, `src/run_pipeline.py`, `app/streamlit_app.py`, `tests/`, `reports/final_project_report.md`, `FINAL_STATUS.md`. |
