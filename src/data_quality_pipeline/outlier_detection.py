import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_validated_dataset.csv"

OUTPUT_PATH = PROJECT_ROOT / "reports" / "outlier_report.csv"

df = pd.read_csv(DATA_PATH)

numerical_columns = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

outlier_results = []

print("========== OUTLIER DETECTION REPORT ==========")

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(f"\nColumn: {column}")
    print("Lower Limit:", round(lower_limit, 2))
    print("Upper Limit:", round(upper_limit, 2))
    print("Outlier Count:", len(outliers))

    outlier_results.append({
        "Column": column,
        "Lower_Limit": round(lower_limit, 2),
        "Upper_Limit": round(upper_limit, 2),
        "Outlier_Count": len(outliers)
    })

print("\n========== OUTLIER ANALYSIS COMPLETED ==========")

report_df = pd.DataFrame(outlier_results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

report_df.to_csv(OUTPUT_PATH, index=False)

print("\nOutlier report saved successfully!")
print("Saved Location:", OUTPUT_PATH)