# Interim Data Directory (`data/interim/`)

Contains intermediate deduplicated data prior to missing-value imputation:
- `cleaned_credit_risk.csv`: `32,416` rows × `12` columns (`165` exact duplicates removed; `5` ages `> 100` and `2` employment lengths `> 60` set to `NaN`). Used as the input to `stratified_train_val_test_split` so that imputation is fitted strictly on the training set without data leakage.

