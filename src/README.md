# Modular Source Code Package (`src/`)

- `src/utils.py` — Configuration loader (`config/config.yaml`), path resolver, and random seed management.
- `src/data/load_data.py` — Dataset loaders, schema validator, and stratified 70/15/15 train/val/test splitter.
- `src/preprocessing/cleaner.py` — Deduplication, anomaly invalidation, `CreditRiskImputer` (`grade_median`, `global_median`, `knn`), and 1,005 ratio mismatch root-cause analyzer.
- `src/visualization/eda_plots.py` — Reproducible generator for EDA figures (`fig01`–`fig09`) and WoE/IV charts (`fig10`–`fig11`).
- `src/features/statistical_tests.py` — Welch's $t$, Mann-Whitney $U$, KS statistic, Chi-Square, Cramér's $V$, Wilson 95% CIs, and Odds Ratios.
- `src/features/woe_iv.py` — `WoEIVAnalyzer` with Laplace smoothing ($\alpha = 0.5$) fitted strictly on training data.
- `src/features/feature_engineering.py` — `CreditRiskFeatureEngineer` (14 domain features, train-only outlier caps & scaling), VIF audit, and multi-criteria feature selection.
- `src/models/train_model.py` — Candidate models, 5-Fold Stratified CV hyperparameter tuning, input validator, and single-borrower SHAP prediction helper.
- `src/evaluation/evaluate_model.py` — Test evaluation metrics, decision threshold optimization, calibration, and evaluation plots (`fig12`–`fig16`).
- `src/run_pipeline.py` — Master end-to-end execution script.

