# Jupyter Notebooks Directory (`notebooks/`)

Sequential, self-contained notebooks executing the complete workflow:
1. `01_data_understanding.ipynb` — Raw data profiling, schema dictionary, duplicate & anomaly identification.
2. `02_data_preprocessing.ipynb` — Deduplication, Global/Grade/KNN imputation benchmark, and root-cause decomposition of the 1,005 `loan_percent_income` mismatches.
3. `03_exploratory_data_analysis.ipynb` — Generation and analysis of Figures 1–9 (`fig01`–`fig09`) and multi-factor risk interactions.
4. `04_statistical_analysis.ipynb` — Welch's $t$, Mann-Whitney $U$, KS statistic, Chi-Square ($\chi^2$), Cramér's $V$, Wilson 95% CIs, and Odds Ratios.
5. `05_woe_iv_analysis.ipynb` — Laplace-smoothed Weight of Evidence (WoE) and Information Value (IV) scorecard analysis (Figures 10–11).
6. `06_feature_engineering.ipynb` — Construction of 14 domain financial features, VIF multicollinearity audit, and feature selection.
7. `07_model_training.ipynb` — Stratified 70/15/15 split, 5-Fold Stratified CV hyperparameter tuning, and training of 8 models.
8. `08_model_evaluation.ipynb` — Held-out test evaluation, threshold optimization (`0.50` vs `0.39` vs `0.16`), calibration, and SHAP explainability (Figures 12–16).
9. `data_preprocessing_validation.ipynb` — Member 1 (Darshan Patil)'s original validation notebook preserved with portable relative file paths.

