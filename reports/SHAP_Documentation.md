# Explainable AI (XAI) Using SHAP – Simple Explanation

**Project:** Finance and Banking – Credit Risk Scoring and Prediction  
**Prepared by:** Member 5 – Documentation and Integration

## 1. What is Explainable AI (XAI)?

Explainable AI (XAI) refers to methods that help us understand how a machine learning model makes its predictions.

In simple words, a machine learning model predicts whether a loan applicant is high-risk or low-risk. XAI helps us understand why the model made that prediction.

**Purpose:**Make machine learning predictions easier to understand and interpret.

## 2. What is SHAP?

SHAP stands for **SHapley Additive exPlanations**.

It is an Explainable AI technique used to understand how individual input features influence a machine learning model's prediction.

In our Credit Risk Scoring project, SHAP can help explain why a particular loan applicant is predicted as high-risk or low-risk.

**Example:**

Consider a loan applicant with the following details:

- Annual Income: ₹3,00,000
- Loan Amount: ₹8,00,000
- Credit Score: 580
- Interest Rate: 14%
- Previous Loan Defaults: Yes

Suppose the model predicts that this applicant has a high credit risk.

SHAP helps identify which features contributed towards that prediction.

**Note:** These values are only an illustrative example. Actual explanations will depend on our trained model and dataset.

## 3. How Does SHAP Work?

SHAP calculates the contribution of each feature towards a model prediction.

It helps us understand whether a feature increases or decreases the predicted risk.

The main concepts are:

- **Base Value:** The model's average prediction before considering the individual applicant's features.
- **SHAP Value:** The contribution of an individual feature towards the prediction.
- **Final Prediction:** The model prediction after considering the contributions of the features.

**Example:**

Suppose an applicant receives a high-risk prediction.

- Previous loan defaults may increase the predicted risk.
- A low credit score may increase the predicted risk.
- A high loan amount may increase the predicted risk.
- A higher income may reduce the predicted risk.

SHAP helps explain these individual contributions.

## 4. Types of SHAP Explanations

### A. Global Explanation

Global explanation helps us understand how features influence predictions across the entire dataset.

It identifies which features are generally important for predicting credit risk.

**Example:** Credit score, loan amount, income and previous defaults may be important features across many applicants.

### B. Local Explanation

Local explanation focuses on one particular applicant.

It explains why the model predicted that specific applicant as high-risk or low-risk.

**Example:** SHAP can explain why a particular applicant received a high-risk prediction based on their credit score, income and loan details.

## 5. Important SHAP Visualizations

SHAP provides different visualizations to understand model predictions.

### 1. SHAP Summary Plot

A summary plot shows how different features influence predictions across multiple applicants.

**Purpose:** Understand the overall impact of features on model predictions.

### 2. SHAP Bar Plot

A bar plot displays the importance of features based on their average absolute SHAP values.

**Purpose:** Identify the most influential features in the model.

### 3. SHAP Waterfall Plot

A waterfall plot explains how individual features contribute to one applicant's prediction.

**Purpose:** Understand why a particular applicant received a high-risk or low-risk prediction.

## 6. Application of SHAP in Our Credit Risk Project

In our Finance and Banking – Credit Risk Scoring project, SHAP will be used to explain the predictions made by the trained machine learning model.

The planned implementation includes:

1. Load the trained credit risk prediction model.
2. Prepare the input data using the same preprocessing steps used during model training.
3. Apply SHAP to explain model predictions.
4. Generate global feature importance visualizations.
5. Generate individual borrower-level explanations.
6. Interpret the factors that increase or decrease predicted credit risk.

**Purpose:** Make credit risk predictions more transparent and understandable.

## 7. Planned Dashboard Enhancement

The existing Streamlit dashboard will be enhanced to display model predictions and their explanations.

The planned dashboard features include:

- **Risk Prediction:** Display whether a borrower is classified as high-risk or low-risk.
- **Risk Probability:** Display the model's predicted risk probability, where supported.
- **Feature Importance:** Show the important features influencing predictions.
- **SHAP Visualization:** Display SHAP summary and waterfall plots.
- **Borrower-Level Interpretation:** Explain why an individual borrower received a particular prediction.

**Purpose:** Help users understand model predictions through a simple and interactive dashboard.

## 8. Tools and Technologies

The planned implementation will use the following tools and technologies:

- **Python:** Programming language for implementation.
- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical operations.
- **Scikit-learn:** Machine learning model support and preprocessing.
- **SHAP:** Explainable AI and feature contribution analysis.
- **Streamlit:** Interactive dashboard development.
- **Matplotlib:** Data visualization.

## 9. Expected Outcomes

After completing the SHAP implementation and dashboard enhancement, the project is expected to provide:

1. Better understanding of machine learning predictions.
2. Identification of important credit risk features.
3. Individual borrower-level explanations.
4. Visual representation of feature contributions.
5. Improved transparency in credit risk assessment.
6. An interactive Streamlit dashboard with explainable predictions.



## 11. Conclusion

SHAP is an important Explainable AI technique that helps us understand how machine learning models make predictions.

In our Credit Risk Scoring project, SHAP will help explain why a loan applicant is predicted as high-risk or low-risk.

By integrating SHAP with the Streamlit dashboard, we aim to make credit risk predictions more transparent, understandable and useful.

