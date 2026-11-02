# Processed Data Directory (`data/processed/`)

Contains validated and feature-engineered datasets:
- `final_validated_dataset.csv`: `32,416` rows × `12` columns (`0` missing values, `0` duplicates). Validated dataset prepared by Member 1 (Darshan Patil) and used for EDA & statistical analysis.
- `train_processed.csv`: Stratified training split (`22,691` rows × `27` columns: `26` selected engineered features + `loan_status`).
- `val_processed.csv`: Stratified validation split (`4,862` rows × `27` columns).
- `test_processed.csv`: Stratified held-out test split (`4,863` rows × `27` columns).

