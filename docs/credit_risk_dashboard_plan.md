# Credit Risk Dashboard – Analysis Plan

## 1. Project Objective
The objective of this dashboard is to analyze loan applications, identify default patterns, and understand the factors associated with loan defaults.

## 2. KPI Cards

| KPI | Calculation | Purpose |
|---|---|---|
| Total Loan Applications | Count of records | Total applications |
| Total Defaults | Count where loan_status = 1 | Number of defaults |
| Non-Defaults | Count where loan_status = 0 | Number of non-defaults |
| Default Rate | Average of loan_status × 100 | Overall default percentage |
| Average Loan Amount | Average of loan_amnt | Average loan value |
| Average Interest Rate | Average of loan_int_rate | Average interest rate |

## 3. Recommended Charts

### Chart 1 – Loan Status Distribution
- Chart Type: Donut Chart
- Category: loan_status
- Values: Count of loan_status
- Purpose: Compare default and non-default applications.

### Chart 2 – Default Rate by Loan Grade
- Chart Type: Column Chart
- X-axis: loan_grade
- Y-axis: Average of loan_status
- Purpose: Identify loan grades with higher default rates.

### Chart 3 – Loan Intent Analysis
- Chart Type: Bar Chart
- X-axis: loan_intent
- Y-axis: Count of loan applications
- Purpose: Understand the distribution of loans by purpose.

### Chart 4 – Default Rate by Home Ownership
- Chart Type: Column Chart
- X-axis: person_home_ownership
- Y-axis: Average of loan_status
- Purpose: Compare default rates across home ownership categories.

### Chart 5 – Age-wise Loan Default Analysis
- Chart Type: Column Chart
- X-axis: Age Groups
- Y-axis: Average of loan_status
- Purpose: Analyze default patterns across different age groups.
- Note: Create age groups using person_age.

### Chart 6 – Loan Amount Analysis
- Chart Type: Bar Chart
- X-axis: loan_grade
- Y-axis: Average of loan_amnt
- Purpose: Compare average loan amounts across loan grades.

### Chart 7 – Loan-to-Income Analysis
- Chart Type: Scatter Chart
- X-axis: person_income
- Y-axis: loan_percent_income
- Purpose: Understand the relationship between income and loan burden.

## 4. Important Dataset Columns

- loan_status: Target variable
- loan_grade: Loan risk grade
- loan_intent: Loan purpose
- person_age: Applicant age
- person_income: Applicant income
- person_home_ownership: Home ownership category
- loan_amnt: Loan amount
- loan_int_rate: Interest rate
- loan_percent_income: Loan amount as a percentage of income

## 5. Target Variable Mapping

- 0 = Non-default
- 1 = Default

## 6. Expected Dashboard Outcome

The dashboard should help users understand loan distribution, default patterns, loan grades, applicant categories, and financial risk indicators.