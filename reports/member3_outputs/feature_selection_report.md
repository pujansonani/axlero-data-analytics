# Feature Selection and Statistical Validation Report

## 1. Objective

Evaluate feature relationships, multicollinearity,
statistical significance, and predictive performance
to identify a suitable feature set for credit risk prediction.

## 2. Dataset

- Total observations: 32,416
- Training observations: 25,932
- Test observations: 6,484
- Target: loan_status
- Original predictors: 11
- Selected candidate predictors: 8

## 3. Correlation Analysis

Pearson and Spearman correlation analysis identified
a strong relationship between borrower age and credit
history length.

- Pearson correlation: 0.878
- Spearman correlation: 0.806

Loan amount and loan percent income also showed
a moderate positive correlation.

These findings motivated further multicollinearity checks.

## 4. Multicollinearity Analysis

Variance Inflation Factor (VIF) was calculated
using the training dataset.

Highest VIF values:

- person_age: 4.378
- cb_person_cred_hist_length: 4.349
- loan_amnt: 2.058
- loan_percent_income: 2.019

All seven numerical predictors had VIF values below 5.

No predictors were removed based solely on VIF.

## 5. Statistical Significance

Welch's t-tests showed statistically significant mean
differences for all seven numerical predictors after
multiple-testing adjustment.

Chi-square tests showed statistically significant
associations for all four categorical predictors.

In the multivariable logistic regression:

- person_age was not significant after FDR correction.
- cb_person_cred_hist_length was not significant.
- person_income was marginally nonsignificant
  after correction (adjusted p = 0.05105).
- cb_person_default_on_file_Y was not significant
  after correction (adjusted p = 0.91175).

These results informed candidate feature selection.

## 6. Feature Selection

The baseline model included 11 features.

The reduced model retained:

1. person_income
2. person_emp_length
3. loan_amnt
4. loan_int_rate
5. loan_percent_income
6. person_home_ownership
7. loan_intent
8. loan_grade

Candidate features excluded:

- person_age
- cb_person_cred_hist_length
- cb_person_default_on_file

These exclusions were evaluated through validation
performance rather than statistical tests alone.

## 7. Validation Model Comparison

| Metric | Baseline | Reduced |
|---|---:|---:|
| ROC-AUC | 0.8775 | 0.8776 |
| Accuracy | 0.8684 | 0.8683 |
| Precision | 0.7623 | 0.7631 |
| Recall | 0.5790 | 0.5769 |
| F1-score | 0.6581 | 0.6570 |

Both models achieved comparable validation performance.

The reduced model used three fewer original predictors
while maintaining approximately the same ROC-AUC.

## 8. Held-Out Test Evaluation

The reduced Logistic Regression model was refitted
using all 25,932 training observations.

It was evaluated on 6,484 held-out test observations.

| Metric | Test Result |
|---|---:|
| ROC-AUC | 0.8681 |
| Accuracy | 0.8689 |
| Precision | 0.7715 |
| Recall | 0.5691 |
| F1-score | 0.6550 |

Confusion matrix:

| Actual / Predicted | Non-default | Default |
|---|---:|---:|
| Non-default | 4827 | 239 |
| Default | 611 | 807 |

At the default 0.5 classification threshold,
the model missed 611 of 1,418 actual defaults.

## 9. Limitations

- The originally calculated IV scores were used as
  reference evidence and were not independently
  recomputed using training-only data.
- Prediction-time availability of loan_grade and
  loan_int_rate requires verification.
- Model performance depends on the classification
  threshold and relative costs of prediction errors.
- Correlation, VIF, and significance alone do not
  establish causal relationships or predictive value.
- The reduced model's similar validation performance
  does not prove it is statistically superior.

## 10. Conclusion

The eight-feature Logistic Regression model
demonstrated performance comparable to the
11-feature baseline on validation data.

The reduced model achieved a test ROC-AUC of 0.8681.

It is recommended as a candidate feature set
for further review, subject to prediction-time
feature availability and deployment requirements.

## 11. Recommendations

- Verify that all selected variables are available
  at the intended prediction time.
- Consider validation-based classification threshold
  tuning to address missed defaults.
- Investigate model calibration and fairness.
- Monitor performance and data drift if deployed.
