"""Feature Engineering, Encoding, Multicollinearity (VIF), and Feature Selection (Module 4 - Pavani).

All stateful transformations (imputation, outlier caps, WoE mappings, scalers, encoders)
are fitted strictly on the training set and applied deterministically to validation/test sets.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.preprocessing.cleaner import GRADE_MAP, CreditRiskImputer

ENGINEERED_FEATURE_DOCS = [
    {
        "feature": "calc_loan_to_income",
        "formula": "loan_amnt / person_income",
        "business_relevance": "Exact debt-to-income ratio removing the 1,005 rounding and 0.85 net-income discrepancies in raw loan_percent_income.",
    },
    {
        "feature": "ratio_discrepancy_flag",
        "formula": "1 if |loan_amnt / person_income - loan_percent_income| > 0.01 else 0",
        "business_relevance": "Identifies the 387 records where person_income reflects post-deduction (0.85x) net income relative to gross underwriting income.",
    },
    {
        "feature": "log_person_income",
        "formula": "ln(1 + person_income)",
        "business_relevance": "Compresses extreme right skewness in annual income (skew = 32.9, max = 6,000,000) to stabilize linear decision boundaries.",
    },
    {
        "feature": "log_loan_amnt",
        "formula": "ln(1 + loan_amnt)",
        "business_relevance": "Normalizes loan principal distribution across micro-loans (500) and maximum personal loans (35,000).",
    },
    {
        "feature": "est_annual_debt_service",
        "formula": "loan_amnt * (1 + loan_int_rate / 100) / 3.0",
        "business_relevance": "Estimates annual principal + interest repayment obligation under a 3-year consumer loan horizon.",
    },
    {
        "feature": "debt_service_to_income",
        "formula": "est_annual_debt_service / person_income",
        "business_relevance": "Captures total interest-adjusted payment shock relative to annual income.",
    },
    {
        "feature": "log_residual_income",
        "formula": "ln(1 + max(0, person_income - est_annual_debt_service))",
        "business_relevance": "Measures the post-debt-service disposable income cushion available for living expenses.",
    },
    {
        "feature": "emp_to_working_age_ratio",
        "formula": "person_emp_length / max(1, person_age - 18)",
        "business_relevance": "Measures career tenure stability relative to total adult working years since age 18.",
    },
    {
        "feature": "cred_hist_to_working_age_ratio",
        "formula": "cb_person_cred_hist_length / max(1, person_age - 18)",
        "business_relevance": "Measures credit file maturity relative to adult age.",
    },
    {
        "feature": "loan_grade_num",
        "formula": "Ordinal mapping A=1, B=2, C=3, D=4, E=5, F=6, G=7",
        "business_relevance": "Preserves monotonic credit risk rank order across lender-assigned underwriting grades.",
    },
    {
        "feature": "high_burden_flag",
        "formula": "1 if loan_percent_income > 0.30 else 0",
        "business_relevance": "Captures the structural threshold above 30% loan-to-income where default rate jumps to 70.4%.",
    },
    {
        "feature": "subprime_grade_flag",
        "formula": "1 if loan_grade in {D, E, F, G} else 0",
        "business_relevance": "Captures the structural jump at Grade D+ where default rate surges from 20.8% (Grade C) to 59.1%+.",
    },
    {
        "feature": "grade_x_burden",
        "formula": "loan_grade_num * loan_percent_income",
        "business_relevance": "Interaction term capturing the compounding risk of a low credit grade combined with a high loan-to-income burden.",
    },
    {
        "feature": "int_rate_spread_over_grade",
        "formula": "loan_int_rate - median(loan_int_rate | loan_grade)_train",
        "business_relevance": "Isolates borrower-specific pricing premium above or below the typical interest rate for their assigned grade.",
    },
]


class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """End-to-end leakage-free feature engineering pipeline for Credit Risk Modeling."""

    def __init__(
        self,
        imputation_strategy: str = "grade_median",
        age_cap: float = 80.0,
        emp_cap: float = 40.0,
        income_cap_quantile: float = 0.995,
        scale_numeric: bool = True,
    ):
        self.imputation_strategy = imputation_strategy
        self.age_cap = age_cap
        self.emp_cap = emp_cap
        self.income_cap_quantile = income_cap_quantile
        self.scale_numeric = scale_numeric

        self.imputer_: Optional[CreditRiskImputer] = None
        self.income_cap_: float = 500000.0
        self.grade_int_medians_: Dict[str, float] = {}
        self.scaler_: Optional[StandardScaler] = None
        self.feature_names_: List[str] = []
        self.numeric_scale_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CreditRiskFeatureEngineer":
        """Fit imputer, outlier caps, grade medians, and standard scaler strictly on training data."""
        self.imputer_ = CreditRiskImputer(strategy=self.imputation_strategy).fit(X)
        X_imp = self.imputer_.transform(X)

        self.income_cap_ = float(X_imp["person_income"].quantile(self.income_cap_quantile))
        self.grade_int_medians_ = (
            X_imp.groupby("loan_grade")["loan_int_rate"].median().to_dict()
        )

        X_feat = self._build_features(X_imp)
        self.feature_names_ = list(X_feat.columns)
        self.numeric_scale_cols_ = [
            c
            for c in self.feature_names_
            if not c.startswith(("home_", "intent_", "prior_default_", "high_burden_", "subprime_", "ratio_discrepancy_"))
        ]
        if self.scale_numeric:
            self.scaler_ = StandardScaler().fit(X_feat[self.numeric_scale_cols_])
        return self

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct domain-engineered features and deterministic categorical encodings."""
        out = pd.DataFrame(index=df.index)

        # Capped continuous base variables
        age = df["person_age"].clip(lower=18.0, upper=self.age_cap)
        emp = df["person_emp_length"].clip(lower=0.0, upper=self.emp_cap)
        income_raw = df["person_income"].clip(lower=1000.0)
        income_capped = income_raw.clip(upper=self.income_cap_)
        loan_amnt = df["loan_amnt"].clip(lower=500.0)
        int_rate = df["loan_int_rate"].clip(lower=5.0, upper=25.0)
        loan_pct_inc = df["loan_percent_income"].clip(lower=0.0, upper=1.0)
        cred_hist = df["cb_person_cred_hist_length"].clip(lower=0.0, upper=50.0)

        out["person_age"] = age
        out["person_income_capped"] = income_capped
        out["person_emp_length"] = emp
        out["loan_amnt"] = loan_amnt
        out["loan_int_rate"] = int_rate
        out["loan_percent_income"] = loan_pct_inc
        out["cb_person_cred_hist_length"] = cred_hist

        # Engineered domain features
        calc_lti = loan_amnt / income_raw
        out["calc_loan_to_income"] = calc_lti
        out["ratio_discrepancy_flag"] = ((calc_lti - loan_pct_inc).abs() > 0.01).astype(int)
        out["log_person_income"] = np.log1p(income_raw)
        out["log_loan_amnt"] = np.log1p(loan_amnt)

        est_debt_service = loan_amnt * (1.0 + int_rate / 100.0) / 3.0
        out["est_annual_debt_service"] = est_debt_service
        out["debt_service_to_income"] = est_debt_service / income_raw
        out["log_residual_income"] = np.log1p(np.maximum(0.0, income_raw - est_debt_service))

        working_years = np.maximum(age - 18.0, 1.0)
        out["emp_to_working_age_ratio"] = emp / working_years
        out["cred_hist_to_working_age_ratio"] = cred_hist / working_years

        grade_num = df["loan_grade"].map(GRADE_MAP).fillna(3).astype(float)
        out["loan_grade_num"] = grade_num
        out["high_burden_flag"] = (loan_pct_inc > 0.30).astype(int)
        out["subprime_grade_flag"] = (grade_num >= 4).astype(int)
        out["grade_x_burden"] = grade_num * loan_pct_inc

        grade_med_rate = df["loan_grade"].map(self.grade_int_medians_).fillna(11.0)
        out["int_rate_spread_over_grade"] = int_rate - grade_med_rate

        # Binary encoding for prior default
        out["prior_default_Y"] = (df["cb_person_default_on_file"] == "Y").astype(int)

        # One-hot encoding for person_home_ownership (OWN as low-risk reference category)
        for cat in ["RENT", "MORTGAGE", "OTHER"]:
            out[f"home_{cat}"] = (df["person_home_ownership"] == cat).astype(int)

        # One-hot encoding for loan_intent (VENTURE as low-risk reference category)
        for intent in [
            "DEBTCONSOLIDATION",
            "MEDICAL",
            "HOMEIMPROVEMENT",
            "PERSONAL",
            "EDUCATION",
        ]:
            out[f"intent_{intent}"] = (df["loan_intent"] == intent).astype(int)

        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform dataset into final engineered feature matrix."""
        if self.imputer_ is None:
            raise RuntimeError("CreditRiskFeatureEngineer must be fitted before transform().")
        X_imp = self.imputer_.transform(X)
        X_feat = self._build_features(X_imp)
        if self.scale_numeric and self.scaler_ is not None:
            X_feat_scaled = X_feat.copy()
            X_feat_scaled[self.numeric_scale_cols_] = self.scaler_.transform(
                X_feat[self.numeric_scale_cols_]
            )
            return X_feat_scaled
        return X_feat


def compute_vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) across feature columns to audit multicollinearity."""
    X_clean = X.astype(float).copy()
    # Add constant column for proper intercept-adjusted VIF calculation
    X_with_const = X_clean.copy()
    X_with_const["_const_"] = 1.0
    cols = list(X_clean.columns)
    vif_rows = []
    vals = X_with_const.values
    for i, col in enumerate(cols):
        vif_val = variance_inflation_factor(vals, i)
        vif_rows.append(
            {
                "feature": col,
                "vif": round(float(vif_val), 2),
                "multicollinearity_status": (
                    "Low (< 5)"
                    if vif_val < 5.0
                    else "Moderate (5 - 10)"
                    if vif_val < 10.0
                    else "High (>= 10, Collinear)"
                ),
            }
        )
    return pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(drop=True)


def rank_and_select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """Evaluate and rank engineered features using ANOVA F-test, Mutual Information, L1 Logistic, and Random Forest."""
    f_vals, f_pvals = f_classif(X_train, y_train)
    mi_vals = mutual_info_classif(X_train, y_train, random_state=random_state)

    l1_clf = LogisticRegression(
        l1_ratio=1.0, C=0.1, solver="liblinear", random_state=random_state, max_iter=500
    )
    l1_clf.fit(X_train, y_train)
    l1_coefs = np.abs(l1_clf.coef_[0])

    rf_clf = RandomForestClassifier(
        n_estimators=150, max_depth=10, random_state=random_state, n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    rf_imps = rf_clf.feature_importances_

    ranking_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "anova_f_stat": np.round(f_vals, 2),
            "anova_p_value": f_pvals,
            "mutual_info": np.round(mi_vals, 4),
            "l1_abs_coef": np.round(l1_coefs, 4),
            "rf_gini_importance": np.round(rf_imps, 4),
        }
    )

    # Normalized composite score across statistical + model-based criteria
    for col in ["anova_f_stat", "mutual_info", "l1_abs_coef", "rf_gini_importance"]:
        mx = ranking_df[col].max()
        ranking_df[f"{col}_norm"] = ranking_df[col] / mx if mx > 0 else 0.0

    ranking_df["composite_score"] = (
        0.20 * ranking_df["anova_f_stat_norm"]
        + 0.25 * ranking_df["mutual_info_norm"]
        + 0.25 * ranking_df["l1_abs_coef_norm"]
        + 0.30 * ranking_df["rf_gini_importance_norm"]
    ).round(4)

    ranking_df = (
        ranking_df.drop(columns=[c for c in ranking_df.columns if c.endswith("_norm")])
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
    ranking_df["rank"] = np.arange(1, len(ranking_df) + 1)

    # Filter out exact redundant collinear duplicates for parsimonious linear modeling while keeping informative set
    redundant_collinear = {"est_annual_debt_service", "person_income_capped"}
    selected = [f for f in ranking_df["feature"].tolist() if f not in redundant_collinear]
    ranking_df["selected"] = ranking_df["feature"].isin(selected)
    return ranking_df, selected
