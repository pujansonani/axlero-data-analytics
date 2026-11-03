# Machine Learning (ML) Pipeline – Simple Explanation

**Project:** Finance and Banking – Credit Risk Scoring  
**Prepared by:** Member 5 – Documentation and Integration

# 1.What is an ML Pipeline?

An ML pipeline is a sequence of steps used to prepare data for machine learning.

In simple words, it explains how we take the original data, clean it, and prepare useful information for a machine learning model.

This document focuses only on the five stages that have been completed or reviewed in the team's notebooks.

# 2. Completed / Reviewed Pipeline

1. Original Loan Dataset
2. Data Cleaning
3. Missing Value Handling
4. Feature Engineering
5. Feature Selection + VIF

## 3. Simple Explanation of Each Stage

 Step 1: Original Loan Dataset

We start with the original loan dataset. It contains information about loan applicants and their loans, such as income, loan amount, employment details, and credit-related information.

**Purpose:** This dataset is the starting point for credit risk analysis.

 Step 2: Data Cleaning

The dataset may contain duplicate records, invalid values, or other data-quality problems.

In this stage, we:
- Check for duplicate rows.
- Identify invalid values.
- Review data-quality issues.
- Remove duplicate records where required.


 Step 3: Missing Value Handling

Some columns may contain empty or missing values.

The reviewed preprocessing notebook includes these methods:
- **Median Imputation:** Fill a missing value using the median of a column.
- **Grade-wise Median Imputation:** Use the median for the relevant loan grade.
- **KNN Imputation:** Estimate a missing value using information from similar records.

The notebook compares imputation methods using **MAE** and **RMSE**. These metrics help measure the difference between estimated and expected values.

**Purpose:** Handle missing information while keeping useful data.

 Step 4: Feature Engineering

Feature engineering means creating useful features from the existing dataset columns.

 Step 5: Feature Selection + VIF

Not every column in a dataset is equally useful. Feature selection helps identify relevant features for the next stage.

The reviewed notebook also includes **VIF (Variance Inflation Factor)** analysis.

VIF helps identify multicollinearity. This happens when two or more input features are strongly related to each other.

**Purpose:** Review important features and identify highly related features.

### 4. Completed / Reviewed ML Pipeline Flow


Original Loan Dataset
        |
        v
Data Cleaning
        |
        v
Missing Value Handling
        |
        v
Feature Engineering
        |
        v
Feature Selection + VIF


## 5. What Has Been Reviewed in the Team's Notebooks?

### Data Preprocessing Notebook

The reviewed notebook includes:
- Duplicate removal
- Invalid-value identification
- Missing-value handling
- Median, grade-wise median, and KNN imputation
- Imputation comparison using MAE and RMSE
- Loan-to-income mismatch analysis
- Final dataset validation

### Feature Engineering Notebook

The reviewed notebook includes:
- Creating useful features
- Feature selection
- VIF analysis to check multicollinearity

These notebook reviews support the five stages listed in the completed/reviewed pipeline above.


