import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_validated_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "data_validation_report.csv"

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

# Load dataset
df = pd.read_csv(DATA_PATH)

if df.empty:
    raise ValueError("Dataset is empty. Validation cannot continue.")

validation_results = []

print("========== DATA VALIDATION REPORT ==========")

print("\nDataset Shape:", df.shape)

# 1. Required Columns Validation
print("\nRequired Columns Check:")

for column in required_columns:

    missing_column = column not in df.columns

    validation_results.append({
        "Check": "Required Column",
        "Column": column,
        "Count": int(missing_column),
        "Status": "FAIL" if missing_column else "PASS"
    })

    print(f"{column}: {'MISSING' if missing_column else 'AVAILABLE'}")

missing_required = [
    column for column in required_columns
    if column not in df.columns
]

if missing_required:
    raise ValueError(f"Required columns missing: {missing_required}")


# 2. Missing Values Check
print("\nMissing Values:")

for column in df.columns:

    missing_count = df[column].isnull().sum()

    validation_results.append({
        "Check": "Missing Values",
        "Column": column,
        "Count": missing_count,
        "Status": "PASS" if missing_count == 0 else "FAIL"
    })

    print(f"{column}: {missing_count}")


# 3. Duplicate Records Check
duplicate_count = df.duplicated().sum()

validation_results.append({
    "Check": "Duplicate Records",
    "Column": "All Columns",
    "Count": duplicate_count,
    "Status": "PASS" if duplicate_count == 0 else "FAIL"
})

print("\nDuplicate Records:", duplicate_count)


# 4. Invalid Records Check
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
    "cb_person_cred_hist_length": (
        df["cb_person_cred_hist_length"] < 0
    )
}

print("\nInvalid Records:")

for column, condition in invalid_checks.items():

    invalid_count = condition.sum()

    validation_results.append({
        "Check": "Invalid Records",
        "Column": column,
        "Count": invalid_count,
        "Status": "PASS" if invalid_count == 0 else "FAIL"
    })

    print(f"{column}: {invalid_count}")


# 5. Target Column Validation
target_values = set(df["loan_status"].dropna().unique())

invalid_target_count = (
    ~df["loan_status"].isin([0, 1])
).sum()

target_status = (
    "PASS"
    if target_values.issubset({0, 1})
    and invalid_target_count == 0
    else "FAIL"
)

validation_results.append({
    "Check": "Target Column Verification",
    "Column": "loan_status",
    "Count": invalid_target_count,
    "Status": target_status
})

print("\nTarget Column Validation:")
print("Unique Values:", target_values)
print("Invalid Target Records:", invalid_target_count)
print("Status:", target_status)


# 6. Data Types Verification
print("\nData Types:")

for column in df.columns:

    dtype = str(df[column].dtype)

    validation_results.append({
        "Check": "Data Types",
        "Column": column,
        "Count": dtype,
        "Status": "PASS"
    })

    print(f"{column}: {dtype}")


# 7. Generate CSV Report
report_df = pd.DataFrame(validation_results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

report_df.to_csv(OUTPUT_PATH, index=False)

print("\n============================================")
print("Data Validation Report Generated Successfully!")
print("Report Location:", OUTPUT_PATH)