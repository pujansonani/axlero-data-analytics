import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "knn_imputed_corrected_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "preprocessed_dataset.csv"
)

# Required columns
required_columns = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length"
]

# Step 1: Load dataset
print("Loading corrected KNN dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully!")
print("Original Shape:", df.shape)

# Step 2: Check required columns
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Step 3: Check missing values
missing_count = df.isnull().sum().sum()

print("Missing Values:", missing_count)

if missing_count > 0:
    raise ValueError("Dataset contains missing values.")

# Step 4: Check duplicates
duplicate_count = df.duplicated().sum()

print("Duplicate Rows:", duplicate_count)

if duplicate_count > 0:
    raise ValueError("Dataset contains duplicate records.")

# Step 5: Validate numeric records
invalid_checks = {
    "person_age": df["person_age"] <= 0,
    "person_income": df["person_income"] <= 0,
    "person_emp_length": df["person_emp_length"] < 0,
    "loan_amnt": df["loan_amnt"] <= 0,
    "loan_int_rate": df["loan_int_rate"] <= 0,
    "loan_percent_income": (
        (df["loan_percent_income"] < 0) |
        (df["loan_percent_income"] > 1)
    ),
    "cb_person_cred_hist_length":
        df["cb_person_cred_hist_length"] < 0
}

for column, condition in invalid_checks.items():

    invalid_count = condition.sum()

    print(f"{column} invalid records: {invalid_count}")

    if invalid_count > 0:
        raise ValueError(f"Invalid records found in {column}")

# Step 6: Validate target column
if not set(df["loan_status"].unique()).issubset({0, 1}):
    raise ValueError("Invalid values found in loan_status.")

print("Target column validation: PASS")

# Step 7: Save preprocessed dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\n========== PREPROCESSING COMPLETED ==========")
print("Final Dataset Shape:", df.shape)
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())
print("Output File:", OUTPUT_PATH)