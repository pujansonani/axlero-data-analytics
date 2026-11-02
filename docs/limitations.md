# Methodological, Data & Operational Limitations

To ensure responsible deployment and transparent academic/industry review, the following limitations are documented:

## 1. Observational Data & Causal Inference Limits
- All findings in EDA, statistical hypothesis testing, WoE/IV, and SHAP analysis reflect **observational associations** within historical loan records—not causal effects. For example, the elevated default rate among renters (`31.6%` vs `7.5%` for homeowners) does not imply that renting causes loan default; rather, housing tenure serves as a proxy for unobserved wealth, liquidity reserves, and balance-sheet stability.

## 2. Endogeneity of Lender-Assigned Underwriting Variables
- Both `loan_grade` (`A`–`G`) and `loan_int_rate` (`5.42%`–`23.22%`) are assigned by the originating lender after evaluating the applicant's credit bureau file and application details.
- Consequently, their high Information Value (`IV = 0.8583` and `0.7038`) partly reflects the lender's existing risk scorecard. When deploying this model for **de novo** underwriting where a lender grade has not yet been assigned, a stage-1 applicant-only model (using `calc_loan_to_income`, `person_income`, `person_home_ownership`, `cb_person_default_on_file`, `loan_intent`, `person_emp_length`, and `loan_amnt`) should be used prior to pricing.

## 3. Sample Selection & Reject Bias (Survivorship Bias)
- The dataset contains only **approved and funded loans** (`32,416` records). Applicants who were rejected during initial underwriting do not appear in the dataset.
- Applying the model to an unselected population of raw applicants without reject-inference adjustments may underestimate default risk in demographic segments that were historically declined.

## 4. Small-Sample Granularity in Extreme Subgroups
- Certain categorical slices contain limited observations:
  - `loan_grade = G`: `n = 64` loans (`63` defaults, `98.4%` default rate, 95% Wilson CI: `[91.7%, 99.7%]`).
  - `person_home_ownership = OTHER`: `n = 106` loans (`33` defaults, `31.1%` default rate, 95% Wilson CI: `[23.1%, 40.5%]`).
- Point estimates in these small cells carry wider uncertainty, which is why Laplace smoothing ($\alpha = 0.5$) and tree leaf regularization (`min_samples_leaf >= 15`) are enforced in our pipeline.

## 5. Unspecified Currency, Vintage & Macroeconomic Regime
- The public benchmark dataset does not specify the exact origination calendar years, geographic region, or currency unit (assumed USD `$` based on US consumer credit bureau conventions).
- Consumer default rates are sensitive to macroeconomic interest rate cycles, inflation, and unemployment shocks. Production deployment requires periodic **Population Stability Index (PSI)** monitoring and quarterly probability recalibration.

