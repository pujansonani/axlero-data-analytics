# Week 3 Report – Advanced EDA & Business Intelligence

**Project:** Finance & Banking – Credit Risk Scoring and Loan Default Analysis
**Member:** Sriram (Member 2) – Advanced EDA & Business Intelligence
**Dataset:** `knn_imputed_corrected_dataset.csv` (32,416 records, 12 features)
**Target:** `loan_status` (0 = Non-default, 1 = Default)
**Notebook:** `member2_week3_advanced_eda_bi/notebooks/week3_advanced_eda_bi.ipynb`

All figures below come from the notebook. "pp" means percentage points. Figures marked *(calculated)* were derived from the notebook tables. All results are observed associations, not causal effects.

---

## 1. Overview

Overall, 7,089 of 32,416 borrowers defaulted (21.87%), and 25,327 did not (78.13%). The dataset has no missing values and no duplicate records. This week's work segmented borrowers by age, income, loan burden, loan grade, home ownership, employment length, loan purpose and prior default history, and compared default rates across them.

## 2. Default Rate by Segment

| Segment | Group: default rate (%) |
|---|---|
| Age group | 18–25: 23.13 · 26–35: 20.69 · 36–45: 20.68 · 46–55: 21.83 · Above 55: 23.88 |
| Income group | Low: 39.75 · Lower-Middle: 21.41 · Upper-Middle: 17.14 · High: 9.13 |
| Loan burden | Low: 11.75 · Moderate: 15.13 · High: 22.02 · Very High: 70.44 |
| Loan grade | A: 9.96 · B: 16.32 · C: 20.75 · D: 59.06 · E: 64.49 · F: 70.54 · G: 98.44 |
| Home ownership | OWN: 7.49 · MORTGAGE: 12.62 · OTHER: 31.13 · RENT: 31.61 |
| Employment length | 0–2 yrs: 27.54 · 3–5: 20.50 · 6–10: 18.17 · 11–15: 16.23 · 16+: 16.12 |
| Loan purpose | Debt consolidation: 28.68 · Medical: 26.76 · Home improvement: 26.15 · Personal: 19.90 · Education: 17.25 · Venture: 14.85 |
| Prior default on file | Yes: 37.87 · No: 18.43 |

**Size of the gap between highest- and lowest-risk groups** (groups with at least 100 borrowers): loan grade 60.58 pp, loan burden 58.68 pp, income 30.61 pp, home ownership 24.12 pp, employment length 11.42 pp, age 3.20 pp.

## 3. Key Findings

- **Loan burden:** The Very High group (3,819 borrowers) has a 70.44% default rate. It is 11.78% of borrowers but accounts for 2,690 defaults (37.95% of all defaults). Defaulted borrowers have an average loan burden of 24.7%, against 14.9% for non-defaulted borrowers.
- **Loan grade:** Default rates rise from 9.96% in grade A to 98.44% in grade G, with the largest jump between C (20.75%) and D (59.06%). Grades D–G together are 4,888 borrowers with 2,992 defaults and a 61.21% default rate *(calculated)*.
- **Grade and burden together:** Burden matters inside every grade. Grade A rises from 3.83% (Low burden) to 60.93% (Very High), B from 7.49% to 66.18%, and C from 11.33% to 75.47%.
- **Income and burden together:** Low Income with Very High burden has a 77.77% default rate (1,943 borrowers). Very High burden stays high at every income level, from 66.95% (Lower-Middle) down to 45.08% (High Income, 122 borrowers).
- **Prior default:** Borrowers with a prior default have a 37.87% default rate (2,170 of 5,730), against 18.43% (4,919 of 26,686). The gap is 19.44 pp.
- **Home ownership:** RENT borrowers account for 5,177 defaults (73.03% of all defaults) *(calculated)*. Renters also have lower average income (55,019.63) than mortgage holders (81,150.98).
- **Age:** The default rate range across age groups is only 3.20 pp, so age alone separates risk poorly.

## 4. Business Recommendations

1. **Make very high loan burden (above 30% of income) a priority review trigger.** The default rate is 70.44% in that group. However, 1,129 of its 3,819 borrowers (29.56%) did not default *(calculated)*, so use it as a review signal rather than a stand-alone rejection rule.
2. **Review loan grade and loan burden together.** Apply enhanced review to grades D–G (61.21% default rate) and to any grade combined with Very High burden.
3. **Monitor low-income borrowers with very high burden as the highest-risk segment.** Their default rate is 77.77% (1,943 borrowers). Low Income overall is 25.0% of borrowers but 45.44% of all defaults *(calculated)*.
4. **Add prior default on file as a review factor.** The rate is 37.87% against 18.43%. Most defaults (69.39%) come from borrowers without a prior default *(calculated)*, so a clean record is not a guarantee.
5. **Treat loan purpose, home ownership and employment length as secondary signals.** Use them for monitoring and ordering review queues, and test whether they add information beyond burden, grade and income.
6. **Do not use age as a stand-alone risk criterion.** The spread is 3.20 pp, and age is a personal characteristic that would need a fairness review.
7. **Monitor and validate before operational use.** Track default rate and counts by segment each period, apply a minimum segment size, re-check on a later time period, and test outcomes across relevant borrower groups before any policy change.

## 5. Limitations

- **Association, not causation.** Each segment view was analysed on its own, with no control for other variables, so overlapping factors are not separated.
- **Small segments.** Above 55 (134 borrowers), grade G (64), grade F (241), home ownership OTHER (106) and High Income with Very High burden (122) are directional only.
- **Overlapping factors.** Income, home ownership, employment length and age are related, so differences by one factor may partly reflect another.
- **Imputed data.** The dataset is KNN-imputed, and the notebook does not show which values were imputed.
- **Cut-offs are analyst choices.** The burden bands (10%, 20%, 30%), age bands, income quartiles and employment bands affect the reported rates.
- **Sample and time.** The 21.87% default rate applies to this dataset only. Its source, time period and class balance are not documented here, and no later-period validation was done.
- **No predictive model.** This is descriptive analysis only. It makes no claim about predictive accuracy and does not analyse `loan_int_rate`.

## 6. Figures Produced

Saved by the notebook in `member2_week3_advanced_eda_bi/figures/`:

- `borrower_segmentation.png` (default rate by age group)
- `loan_burden_default_rate.png`
- `loan_grade_default_rate.png`
- `home_ownership_default_rate.png`
- `employment_length_default_rate.png`
- `grade_by_loan_burden_default_rate.png`
- `income_by_loan_burden_default_rate.png`

## 7. Conclusion

Loan burden, loan grade, income and prior default history show the clearest gaps in observed default rates. They are the best candidates for review rules and monitoring. Home ownership, employment length and loan purpose are useful secondary signals, and age adds little on its own. These findings support investigation and monitoring, not production credit decisions.
