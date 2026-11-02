# Project Overview — Axlero Credit Risk Scoring & Loan Default Analysis

## 1. Domain & Business Context
**Sector:** Finance & Banking — Consumer Lending & Credit Risk Management  
**Project Title:** Credit Risk Scoring and Loan Default Analysis  
**Organization:** Axlero Solutions

Consumer lending institutions must continuously balance portfolio growth against credit default losses. When a borrower defaults on an unsecured personal loan, the lender loses up to 100% of the unpaid principal balance (**Loss Given Default, LGD**), whereas declining a creditworthy borrower forfeits only the net interest margin (**NIM**). Consequently, accurate **Probability of Default (PD)** estimation, transparent risk factor attribution, and cost-sensitive decision thresholds are core requirements of modern credit risk analytics.

## 2. Dataset Provenance & Scope
- **Raw Dataset (`data/raw/credit_risk_dataset.csv`):** `32,581` loan records across `12` attributes (`7` numerical features, `4` categorical features, and `1` binary target `loan_status`, where `1 = Default` and `0 = Non-Default / Repaid`).
- **Validated Dataset (`data/processed/final_validated_dataset.csv`):** `32,416` unique validated records (`25,327` repaid loans [`78.13%`] and `7,089` defaulted loans [`21.87%`]).

## 3. Core Deliverables
1. **Reproducible Python Package (`src/`):** Modular data loading, preprocessing, imputation comparison, EDA & model plotting, statistical testing, WoE/IV scorecard calculation, feature engineering, model training, and evaluation.
2. **Eight Sequential Jupyter Notebooks (`notebooks/01`–`08`):** Fully executed and documented notebooks covering every analytical phase from data understanding to SHAP explainability, alongside Member 1's preserved `data_preprocessing_validation.ipynb`.
3. **Comprehensive Technical Reports (`reports/`):** Five markdown analytical reports plus `16` publication-grade figures (`reports/figures/fig01`–`fig16`).
4. **Interactive Streamlit Dashboard (`app/streamlit_app.py`):** Eight-page analytics and live credit risk scoring web application with real-time input validation and borrower-level SHAP explanations.
5. **Automated Test Suite (`tests/`):** Pytest verification suite covering schema checks, deduplication, ratio mismatch decomposition, leakage-free imputation, WoE/IV Laplace smoothing, feature engineering, and model inference.

