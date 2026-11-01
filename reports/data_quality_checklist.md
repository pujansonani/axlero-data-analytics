
# Data Quality Checklist
## Project: Credit Risk Scoring and Loan Default Analysis
## Prepared by: Darshan Patil - Member 1

### 1. Dataset Overview
- Original records: 32,581
- Original columns: 12
- Final records: 32,416
- Final columns: 12

### 2. Duplicate Records
- Duplicate records found: 165
- Action: Removed duplicate records.
- Remaining duplicates: 0

### 3. Missing Values
- person_age: 5 invalid values replaced using median (26).
- person_emp_length: 889 missing/invalid values filled using median (4).
- loan_int_rate: 3,095 missing values filled using loan-grade-wise median.
- Final missing values: 0

### 4. Invalid Records
- Age values above 100: 5 records treated as invalid.
- Employment length above 60: 2 records treated as invalid.
- Invalid values were replaced with missing values before imputation.

### 5. Logical Consistency
- Employment length greater than age: 0
- Credit history longer than age: 0
- Zero income records: 0

### 6. Categorical Validation
- Checked unique values in categorical columns.
- No obvious spelling or case inconsistencies identified.

### 7. Numeric Validation
- Checked numeric ranges and negative values.
- No negative values found in the checked numeric columns.

### 8. Target Column Validation
- Target column: loan_status
- Unique values: 0 and 1
- Non-default records (0): 25,327
- Default records (1): 7,089
- Missing target values: 0

### 9. Loan-to-Income Ratio Validation
- Compared loan_percent_income with loan_amnt / person_income.
- Total records: 32,416
- Matching records after rounding: 31,411
- Mismatched records after rounding: 1,005
- Original loan_percent_income values were preserved.
- Mismatches documented for further investigation.

### 10. Final Validation
- Final dataset shape: 32,416 rows × 12 columns
- Duplicate rows: 0
- Missing values: 0

### 11. Output File
Data/processed/final_validated_dataset.csv

### 12. Status
Data cleaning and validation completed.
Loan-to-income ratio inconsistencies documented for review.
