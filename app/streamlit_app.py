"""Axlero Solutions — Credit Risk Scoring & Loan Default Analysis Streamlit Dashboard."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_raw_data, load_validated_data
from src.models.train_model import predict_single_borrower

st.set_page_config(
    page_title="Axlero Solutions | Credit Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_datasets():
    df_raw = load_raw_data()
    df_val = load_validated_data()
    return df_raw, df_val


@st.cache_data
def get_json_artifacts():
    models_dir = PROJECT_ROOT / "models"
    required_files = [
        "imputation_comparison.json",
        "statistical_summary.json",
        "woe_iv_summary.json",
        "all_models_summary.json",
        "threshold_analysis.json",
    ]
    loaded = []
    for fname in required_files:
        fpath = models_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Required artifact '{fname}' not found in 'models/'. Run 'PYTHONPATH=. python src/run_pipeline.py' first."
            )
        with open(fpath, "r", encoding="utf-8") as f:
            loaded.append(json.load(f))
    return tuple(loaded)


@st.cache_resource
def get_prediction_bundle():
    pipeline_path = PROJECT_ROOT / "models" / "credit_risk_pipeline.joblib"
    if not pipeline_path.exists():
        return None
    return joblib.load(pipeline_path)


def show_figure(filename: str, caption: str = ""):
    fig_path = PROJECT_ROOT / "reports" / "figures" / filename
    if fig_path.exists():
        st.image(str(fig_path), caption=caption, width="stretch")
    else:
        st.warning(f"Figure `{filename}` not found in `reports/figures/`.")


try:
    df_raw, df_val = get_datasets()
    imp_data, stat_data, woe_data, model_data, thresh_data = get_json_artifacts()
except Exception as exc:
    st.error(f"Initialization Error: {exc}")
    st.stop()

bundle = get_prediction_bundle()

# Dynamic portfolio & model metrics derived from actual artifacts
champ_row = model_data["model_comparison"][0]
champ_name = model_data["champion_model_name"]
default_count = int(df_val["loan_status"].sum())
default_rate_pct = float(df_val["loan_status"].mean() * 100.0)
f1_opt_thresh = float(thresh_data["f1_optimal_threshold"])
f1_opt_metrics = thresh_data["f1_optimal_test_metrics"]
cost_opt_thresh = float(thresh_data["cost_optimal_threshold_5_to_1"])

# Sidebar Navigation
st.sidebar.title("🏦 Axlero Credit Risk")
st.sidebar.caption("Finance & Banking — Credit Risk Scoring & Loan Default Analysis")

PAGES = [
    "1. Project Overview",
    "2. Dataset Overview",
    "3. Data Quality & Preprocessing",
    "4. Exploratory Data Analysis",
    "5. Risk Factors & Statistical Analysis",
    "6. WoE & IV Analysis",
    "7. Model Performance Comparison",
    "8. Credit Risk Prediction",
]
selected_page = st.sidebar.radio("Navigate Module", PAGES)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
**Verified Portfolio Metrics**
- **Raw Records:** `{len(df_raw):,}`
- **Validated Records:** `{len(df_val):,}`
- **Default Rate:** `{default_rate_pct:.2f}%` (`{default_count:,}` loans)
- **Champion Model:** `{champ_name}`
- **Test ROC-AUC:** `{champ_row['test_roc_auc']:.4f}`
- **Test PR-AUC:** `{champ_row['test_pr_auc']:.4f}`
"""
)

# =========================================================================
# PAGE 1: PROJECT OVERVIEW
# =========================================================================
if selected_page == "1. Project Overview":
    st.title("🏦 Credit Risk Scoring & Loan Default Analysis")
    st.subheader("Axlero Solutions — End-to-End Data Analytics & Machine Learning Platform")

    removed_dups = len(df_raw) - len(df_val)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validated Portfolio", f"{len(df_val):,} Loans", f"-{removed_dups} exact duplicates")
    c2.metric("Portfolio Default Rate", f"{default_rate_pct:.2f}%", f"{default_count:,} Defaults")
    c3.metric("Champion Test ROC-AUC", f"{champ_row['test_roc_auc']:.4f}", champ_name)
    c4.metric(
        "F1-Optimal Precision / Recall",
        f"{f1_opt_metrics['precision']*100:.1f}% / {f1_opt_metrics['recall']*100:.1f}%",
        f"Threshold = {f1_opt_thresh:.2f}",
    )

    st.markdown(
        f"""
### 1. Problem Statement & Business Context
Financial institutions face asymmetric credit risk when underwriting consumer loans: approving a borrower who defaults (**False Negative**) results in substantial principal charge-offs, whereas declining a creditworthy applicant (**False Positive**) incurs foregone interest income. Because the historical portfolio exhibits a **{default_rate_pct:.2f}% default rate** (`3.57 : 1` class imbalance), naive accuracy-driven decisioning fails to protect loan book capital.

### 2. Business & Technical Objectives
1. **Data Governance & Quality Auditing:** Reconcile duplicate record definitions (`{removed_dups}` redundant copies removed), remediate impossible demographic anomalies (`age > 100`, `emp_length > 60`), benchmark missing-value imputation strategies (Global Median vs Grade-Stratified Median vs KNN), and resolve the `{imp_data['mismatch_audit']['total_mismatches']:,}` `loan_percent_income` consistency discrepancies.
2. **Evidence-Based EDA & Statistical Testing:** Quantify univariate and multi-factor risk patterns while rigorously distinguishing observational association from causation and identifying lender-assigned policy variables (`loan_grade`, `loan_int_rate`).
3. **Credit Scorecard WoE & IV Modeling:** Compute Laplace-smoothed Weight of Evidence (WoE) and Information Value (IV) strictly on training data to rank feature predictive power without target leakage.
4. **Leakage-Free Feature Engineering & ML Pipeline:** Engineer domain financial ratios, audit multicollinearity via VIF, compare 8 models across Stratified 5-Fold CV and a held-out test set (`n = {model_data['split_sizes']['test_rows']:,}`), optimize decision thresholds under asymmetric loss (`5:1` FN:FP cost ratio), and explain predictions with **SHAP**.
"""
    )

    st.markdown("### 3. Team Module Architecture & Contributions")
    team_df = pd.DataFrame(
        [
            {
                "Member": "Darshan Patil (Member 1)",
                "Role": "Data Preprocessing & Validation",
                "Deliverables": "Deduplication (32,581 -> 32,416), outlier/invalid value rules, Grade-Stratified & KNN imputation benchmark, 1,005 ratio mismatch root-cause audit",
            },
            {
                "Member": "Srirami Reddy Ailuri (Member 2)",
                "Role": "Exploratory Data Analysis & Visualization",
                "Deliverables": "9 core publication-grade figures (fig01-fig09), statistical observations, multi-factor interaction insights, correlation vs causation analysis",
            },
            {
                "Member": "Charukesh (Member 3)",
                "Role": "Statistical Analysis & WoE/IV",
                "Deliverables": "Mann-Whitney U, KS, Welch t-test, Chi-Square, Cramér's V, Wilson 95% CIs, Laplace-smoothed WoE/IV scorecard tables & plots (fig10-fig11)",
            },
            {
                "Member": "Pavani (Member 4)",
                "Role": "Feature Engineering & Selection",
                "Deliverables": "14 domain financial & borrower features, train-only outlier capping & scaling, VIF multicollinearity audit, multi-criteria feature selection",
            },
            {
                "Member": "Sowmya (Member 5)",
                "Role": "Machine Learning, Tuning & Explainability",
                "Deliverables": "Stratified 70/15/15 split, 8-model comparison, 5-Fold CV tuning, ROC/PR/Calibration/Confusion plots (fig12-fig16), Threshold optimization & SHAP",
            },
            {
                "Member": "Pujan Sonani (Team Lead)",
                "Role": "Architecture, Integration & Streamlit App",
                "Deliverables": "Repository integration, modular src/ package, automated test suite, interactive Streamlit dashboard, end-to-end documentation & reports",
            },
        ]
    )
    st.dataframe(team_df, width="stretch", hide_index=True)

# =========================================================================
# PAGE 2: DATASET OVERVIEW
# =========================================================================
elif selected_page == "2. Dataset Overview":
    st.title("📊 Dataset Overview & Schema Explorer")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Dataset Rows", f"{len(df_raw):,}")
    c2.metric("Validated Dataset Rows", f"{len(df_val):,}")
    c3.metric("Features / Columns", f"{df_val.shape[1] - 1} Predictors + 1 Target")
    c4.metric("Missing Values (Validated)", f"{int(df_val.isnull().sum().sum())} (100% Complete)")

    tab1, tab2, tab3 = st.tabs(
        ["📋 Variable Dictionary & Summary Stats", "🔍 Interactive Data Explorer", "⚖️ Raw vs Validated Comparison"]
    )

    with tab1:
        st.subheader(f"Numerical Variables Summary (Validated Dataset, n = {len(df_val):,})")
        st.dataframe(df_val.describe().T.round(3), width="stretch")

        st.subheader("Categorical Variables Distribution")
        cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
        cols_ui = st.columns(2)
        for idx, col in enumerate(cat_cols):
            vc = df_val[col].value_counts().reset_index()
            vc.columns = [col, "Loan Count"]
            vc["Share (%)"] = (vc["Loan Count"] / len(df_val) * 100).round(2)
            cols_ui[idx % 2].write(f"**{col}**")
            cols_ui[idx % 2].dataframe(vc, width="stretch", hide_index=True)

    with tab2:
        st.subheader("Filter & Inspect Validated Loan Records")
        fc1, fc2, fc3 = st.columns(3)
        sel_grades = fc1.multiselect(
            "Filter by Loan Grade", sorted(df_val["loan_grade"].unique()), default=sorted(df_val["loan_grade"].unique())
        )
        sel_home = fc2.multiselect(
            "Filter by Home Ownership",
            sorted(df_val["person_home_ownership"].unique()),
            default=sorted(df_val["person_home_ownership"].unique()),
        )
        sel_status = fc3.selectbox("Filter by Loan Status", ["All", "Repaid (0)", "Default (1)"])

        filtered = df_val[
            df_val["loan_grade"].isin(sel_grades) & df_val["person_home_ownership"].isin(sel_home)
        ]
        if sel_status == "Repaid (0)":
            filtered = filtered[filtered["loan_status"] == 0]
        elif sel_status == "Default (1)":
            filtered = filtered[filtered["loan_status"] == 1]

        filt_rate = float(filtered["loan_status"].mean() * 100.0) if len(filtered) > 0 else 0.0
        st.caption(
            f"Showing first 200 of **{len(filtered):,}** matching records (Filtered Default Rate: **{filt_rate:.2f}%**)"
        )
        st.dataframe(filtered.head(200), width="stretch")

    with tab3:
        st.subheader("Raw vs Validated Dataset Comparison")
        ca = imp_data["cleaning_audit"]
        comp_summary = pd.DataFrame(
            [
                {
                    "Metric": "Total Rows",
                    "Raw Dataset": f"{ca['raw_rows']:,}",
                    "Validated Dataset": f"{ca['dedup_rows']:,}",
                    "Notes": f"{ca['duplicates_removed_keep_first']} exact duplicate rows removed",
                },
                {
                    "Metric": "Missing `person_emp_length`",
                    "Raw Dataset": f"{ca['raw_missing']['person_emp_length']:,} (2.75%)",
                    "Validated Dataset": "0 (0.00%)",
                    "Notes": f"8 duplicates removed + {ca['invalid_records_flagged']['person_emp_length_gt_60_count']} invalid (>60) -> {ca['post_invalidation_missing']['person_emp_length']} imputed with median (4.0 yrs)",
                },
                {
                    "Metric": "Missing `loan_int_rate`",
                    "Raw Dataset": f"{ca['raw_missing']['loan_int_rate']:,} (9.56%)",
                    "Validated Dataset": "0 (0.00%)",
                    "Notes": f"21 duplicates removed -> {ca['post_invalidation_missing']['loan_int_rate']:,} imputed via loan-grade median",
                },
                {
                    "Metric": "Invalid `person_age > 100`",
                    "Raw Dataset": f"{ca['invalid_records_flagged']['person_age_gt_100_count']} records (max 144)",
                    "Validated Dataset": f"0 records (max {int(df_val['person_age'].max())})",
                    "Notes": "Ages 123 and 144 set to NaN and imputed with median (26.0 yrs)",
                },
                {
                    "Metric": "Invalid `person_emp_length > 60`",
                    "Raw Dataset": f"{ca['invalid_records_flagged']['person_emp_length_gt_60_count']} records (max 123)",
                    "Validated Dataset": f"0 records (max {int(df_val['person_emp_length'].max())})",
                    "Notes": "Employment lengths of 123 yrs (ages 21, 22) replaced with median (4.0 yrs)",
                },
            ]
        )
        st.dataframe(comp_summary, width="stretch", hide_index=True)

# =========================================================================
# PAGE 3: DATA QUALITY & PREPROCESSING
# =========================================================================
elif selected_page == "3. Data Quality & Preprocessing":
    st.title("🧹 Data Quality Audit, Imputation Benchmark & Ratio Reconciliation")
    st.caption("Member 1 (Darshan Patil) Deliverables + Integration Audit Enhancements")

    st.subheader("1. Resolution of Duplicate Record Count Discrepancy")
    st.info(imp_data["cleaning_audit"]["duplicate_explanation"])

    st.subheader("2. Missing-Value Imputation Benchmark (Global Median vs Grade Median vs KNN)")
    st.markdown(
        f"""
To evaluate whether **KNN Imputation (`k=5`, distance-weighted)** or **Loan-Grade Stratified Median Imputation** better reconstructs missing financial variables without data leakage, we masked `{imp_data['imputation_comparison']['evaluation_sample_size']:,}` complete validation observations and measured reconstruction error:
"""
    )
    imp_methods_df = pd.DataFrame(imp_data["imputation_comparison"]["methods"]).T.reset_index()
    imp_methods_df.columns = [
        "Imputation Strategy",
        "Interest Rate RMSE (%)",
        "Interest Rate MAE (%)",
        "Emp. Length RMSE (Yrs)",
        "Emp. Length MAE (Yrs)",
    ]
    st.dataframe(imp_methods_df, width="stretch", hide_index=True)
    st.success(f"**Selected Pipeline Strategy:** {imp_data['imputation_comparison']['justification']}")

    st.subheader("3. Root-Cause Investigation of the 1,005 `loan_percent_income` Mismatches")
    mism_audit = imp_data["mismatch_audit"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Exact Round(2) Matches", f"{mism_audit['exact_round2_matches']:,} ({100 - mism_audit['mismatch_pct']:.1f}%)")
    mc2.metric("Total Mismatches", f"{mism_audit['total_mismatches']:,} ({mism_audit['mismatch_pct']:.1f}%)")
    mc3.metric(
        "Category 1: Rounding Boundary (<=0.01)",
        f"{mism_audit['decomposition']['rounding_or_truncation_boundary_le_0_01']:,}",
    )
    mc4.metric(
        "Category 2: 0.85x Net Income Scaled",
        f"{mism_audit['decomposition']['net_income_0_85_scaled_denominator_gt_0_01']:,}",
    )

    st.markdown(f"**Root-Cause Finding:** {mism_audit['root_cause_explanation']}")
    st.write("**Sample Records from Category 2 (`person_income = 0.85 × implied_gross_income`):**")
    st.dataframe(pd.DataFrame(mism_audit["sample_scaled_income_records"]), width="stretch", hide_index=True)

# =========================================================================
# PAGE 4: EXPLORATORY DATA ANALYSIS
# =========================================================================
elif selected_page == "4. Exploratory Data Analysis":
    st.title("📈 Exploratory Data Analysis (EDA)")
    st.caption("Member 2 (Srirami Reddy Ailuri) Visualisations & Evidence-Based Observations")

    st.warning(
        "**Methodological Caution (Correlation vs. Causation):** All visualised patterns represent observational associations in historical data, not causal effects. Furthermore, `loan_grade` and `loan_int_rate` are assigned by the lender after evaluating applicant credit risk."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        show_figure("fig01_loan_status.png", "Figure 1: Target distribution (21.9% default rate, n = 32,416).")
        show_figure("fig03_default_rate_by_loan_burden.png", "Figure 3: Loans exceeding 30% of annual income default at 70.4%.")
        show_figure("fig05_default_rate_by_loan_intent.png", "Figure 5: Debt consolidation (28.7%) and medical (26.8%) carry highest risk.")
    with col_b:
        show_figure("fig02_default_rate_by_grade.png", "Figure 2: Default rate rises sharply from Grade C (20.8%) to Grade D (59.1%).")
        show_figure("fig04_prior_default_and_home_ownership.png", "Figure 4: Prior default (37.9%) and renting (31.6%) strongly stratify default risk.")
        show_figure("fig06_income.png", "Figure 6: Bottom income quartile defaults at 39.7% vs 9.1% in top quartile.")

    st.markdown("---")
    col_c, col_d = st.columns(2)
    with col_c:
        show_figure("fig07_loan_amount_and_interest_rate.png", "Figure 7: Interest rate shows a sharper monotonic risk pattern than loan amount.")
        show_figure("fig09_correlation_heatmap.png", "Figure 9: Spearman rank correlation matrix across numerical variables.")
    with col_d:
        show_figure("fig08_age_and_employment.png", "Figure 8: Shorter employment tenure (0-1 yrs: 27.9%) is associated with higher default risk.")

# =========================================================================
# PAGE 5: RISK FACTORS & STATISTICAL ANALYSIS
# =========================================================================
elif selected_page == "5. Risk Factors & Statistical Analysis":
    st.title("📐 Parametric & Non-Parametric Statistical Analysis")
    st.caption("Module 3 (Charukesh) — Hypothesis Testing, Effect Sizes & Confidence Intervals")

    st.subheader("1. Numerical Variables: Mann-Whitney U, Kolmogorov-Smirnov (KS), Welch's t & Correlations")
    num_tests_df = pd.DataFrame(stat_data["numerical_tests"])
    st.dataframe(num_tests_df, width="stretch", hide_index=True)

    st.subheader("2. Categorical Variables: Chi-Square Test of Independence & Cramér's V")
    cat_tests_df = pd.DataFrame(stat_data["categorical_tests"])
    st.dataframe(cat_tests_df, width="stretch", hide_index=True)

    st.subheader("3. Category-Level Default Rates, 95% Wilson CIs & Odds Ratios")
    sel_cat = st.selectbox(
        "Select Categorical Feature to Inspect Level-Wise Odds Ratios",
        list(stat_data["categorical_level_tables"].keys()),
    )
    lvl_df = pd.DataFrame(stat_data["categorical_level_tables"][sel_cat])
    st.dataframe(lvl_df, width="stretch", hide_index=True)

# =========================================================================
# PAGE 6: WOE & IV ANALYSIS
# =========================================================================
elif selected_page == "6. WoE & IV Analysis":
    st.title("⚖️ Weight of Evidence (WoE) & Information Value (IV)")
    st.caption("Module 3 (Charukesh) — Credit Scorecard Predictive Power & Monotonic Binning")

    st.markdown(
        r"""
### Mathematical Formulation (Fitted Strictly on Training Split, $n = 22,691$)
To prevent target leakage and zero-frequency division errors, bin edges are learned strictly on the training split and smoothed with **Laplace parameter $\alpha = 0.5$**:
$$\text{WoE}_i = \ln\!\left(\frac{\text{Dist}_{\text{Event}, i}}{\text{Dist}_{\text{NonEvent}, i}}\right), \qquad \text{IV} = \sum_{i=1}^{k} \left(\text{Dist}_{\text{Event}, i} - \text{Dist}_{\text{NonEvent}, i}\right) \cdot \text{WoE}_i$$
"""
    )

    c1, c2 = st.columns([1.1, 1.0])
    with c1:
        st.subheader("Feature Ranking by Information Value (IV)")
        iv_df = pd.DataFrame(woe_data["iv_ranking"])
        st.dataframe(iv_df, width="stretch", hide_index=True)
    with c2:
        show_figure("fig10_woe_iv_summary.png", "Figure 10: Information Value (IV) ranking with Siddiqi credit scoring thresholds.")

    st.markdown("---")
    show_figure("fig11_woe_curves.png", "Figure 11: Weight of Evidence (WoE) and Default Rate curves across top 4 predictors.")

    st.subheader("Inspect Bin-Level WoE & IV Table by Feature")
    sel_woe_feat = st.selectbox("Select Feature", [r["feature"] for r in woe_data["iv_ranking"]])
    st.dataframe(pd.DataFrame(woe_data["bin_tables"][sel_woe_feat]), width="stretch", hide_index=True)

# =========================================================================
# PAGE 7: MODEL PERFORMANCE COMPARISON
# =========================================================================
elif selected_page == "7. Model Performance Comparison":
    st.title("🤖 Machine Learning Model Comparison & Explainability")
    st.caption("Module 4 (Pavani — Feature Engineering) & Module 5 (Sowmya — ML Training, Tuning & SHAP)")

    st.subheader(
        f"1. Held-Out Test Set Evaluation Summary (n = {model_data['split_sizes']['test_rows']:,}; Selected by Validation ROC-AUC)"
    )
    eval_df = pd.DataFrame(model_data["model_comparison"])
    st.dataframe(eval_df, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        show_figure("fig12_roc_pr_curves.png", "Figure 12: ROC and Precision-Recall curves across all 8 models on the test set.")
        show_figure(
            "fig14_threshold_optimization.png",
            f"Figure 14: Decision threshold optimization (F1-optimal = {f1_opt_thresh:.2f}, 5:1 Cost-optimal = {cost_opt_thresh:.2f}).",
        )
    with c2:
        show_figure("fig13_confusion_matrices.png", "Figure 13: Row-normalized confusion matrices on the held-out test set.")
        show_figure("fig16_calibration_curve.png", "Figure 16: Probability calibration curves and Brier scores.")

    st.markdown("---")
    c3, c4 = st.columns([1.1, 1.0])
    with c3:
        show_figure("fig15_shap_summary.png", "Figure 15: Global SHAP feature importance for the champion model.")
    with c4:
        st.subheader("Decision Threshold Regime Analysis (Test Set)")
        thresh_df = pd.DataFrame(
            [
                {"Operating Regime": "Default Threshold (0.50)", **thresh_data["default_threshold_0_50_test_metrics"]},
                {"Operating Regime": f"F1-Optimal Threshold ({f1_opt_thresh:.2f})", **thresh_data["f1_optimal_test_metrics"]},
                {"Operating Regime": f"5:1 Cost-Optimal Threshold ({cost_opt_thresh:.2f})", **thresh_data["cost_optimal_test_metrics"]},
            ]
        )
        st.dataframe(thresh_df, width="stretch", hide_index=True)

        st.subheader("Engineered Features & VIF Multicollinearity Audit")
        st.dataframe(pd.DataFrame(model_data["vif_after_selection"]), width="stretch", hide_index=True)

# =========================================================================
# PAGE 8: CREDIT RISK PREDICTION
# =========================================================================
elif selected_page == "8. Credit Risk Prediction":
    st.title("🎯 Interactive Credit Risk Scorer & SHAP Explainer")
    st.markdown(
        f"Enter borrower and loan attributes below to run live inference using the saved preprocessing pipeline and **{champ_name}** model, including local **SHAP** factor attributions."
    )

    if bundle is None:
        st.error(
            "Trained model bundle (`models/credit_risk_pipeline.joblib`) is unavailable. Please run `PYTHONPATH=. python src/run_pipeline.py` first."
        )
        st.stop()

    # Preset buttons for instant demonstration
    st.write("**Quick-Fill Sample Applicant Profiles:**")
    pc1, pc2, pc3 = st.columns(3)
    if "preset" not in st.session_state:
        st.session_state.preset = "medium"

    if pc1.button("🟢 Load Low-Risk Profile (Prime Grade A)"):
        st.session_state.preset = "low"
    if pc2.button("🟡 Load Medium-Risk Profile (Grade C Consolidation)"):
        st.session_state.preset = "medium"
    if pc3.button("🔴 Load High-Risk Profile (High Burden + Grade E)"):
        st.session_state.preset = "high"

    presets = {
        "low": dict(age=31, inc=95000, home="OWN", emp=8.0, intent="VENTURE", grade="A", amnt=7000, rate=7.49, prior="N", cred=8),
        "medium": dict(age=26, inc=48000, home="RENT", emp=3.0, intent="DEBTCONSOLIDATION", grade="C", amnt=11500, rate=13.48, prior="N", cred=4),
        "high": dict(age=23, inc=28000, home="RENT", emp=0.5, intent="DEBTCONSOLIDATION", grade="E", amnt=12000, rate=16.82, prior="Y", cred=2),
    }
    p = presets[st.session_state.preset]

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            person_age = st.number_input("Applicant Age (Years)", min_value=18, max_value=120, value=int(p["age"]), step=1)
            person_income = st.number_input("Annual Income ($)", min_value=1000, max_value=10000000, value=int(p["inc"]), step=1000)
            person_emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=70.0, value=float(p["emp"]), step=0.5)
            cb_cred_hist = st.number_input("Credit History Length (Years)", min_value=0, max_value=60, value=int(p["cred"]), step=1)

        with col2:
            person_home = st.selectbox(
                "Home Ownership Status",
                ["RENT", "MORTGAGE", "OWN", "OTHER"],
                index=["RENT", "MORTGAGE", "OWN", "OTHER"].index(p["home"]),
            )
            loan_intent = st.selectbox(
                "Loan Purpose (Intent)",
                ["DEBTCONSOLIDATION", "MEDICAL", "HOMEIMPROVEMENT", "PERSONAL", "EDUCATION", "VENTURE"],
                index=["DEBTCONSOLIDATION", "MEDICAL", "HOMEIMPROVEMENT", "PERSONAL", "EDUCATION", "VENTURE"].index(p["intent"]),
            )
            loan_grade = st.selectbox(
                "Loan Grade (A = Best, G = Worst)",
                ["A", "B", "C", "D", "E", "F", "G"],
                index=["A", "B", "C", "D", "E", "F", "G"].index(p["grade"]),
            )
            cb_prior = st.selectbox(
                "Prior Default on Credit File",
                ["N", "Y"],
                index=["N", "Y"].index(p["prior"]),
            )

        with col3:
            loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=500, max_value=100000, value=int(p["amnt"]), step=500)
            loan_int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=float(p["rate"]), step=0.1)
            calc_lti = round(loan_amnt / max(person_income, 1), 2)
            st.metric("Calculated Loan-to-Income Ratio", f"{calc_lti*100:.1f}% ({calc_lti:.2f})")

        submitted = st.form_submit_button("🔍 Evaluate Borrower Credit Risk", width="stretch")

    payload = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": calc_lti,
        "cb_person_default_on_file": cb_prior,
        "cb_person_cred_hist_length": cb_cred_hist,
    }
    try:
        res = predict_single_borrower(bundle, payload)
        st.markdown("---")
        st.subheader("Underwriting Risk Assessment Result")

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Predicted Default Probability", f"{res['default_probability_pct']:.2f}%")
        rc2.metric("Assigned Risk Tier", res["risk_category"])
        rc3.metric(
            f"F1-Optimal Decision ({res['decision_threshold']:.2f})",
            "DEFAULT FLAG (1)" if res["predicted_default"] == 1 else "PERFORMING (0)",
        )
        rc4.metric(
            f"5:1 Cost-Sensitive Flag ({cost_opt_thresh:.2f})",
            "REVIEW / HIGH RISK" if res["default_probability"] >= cost_opt_thresh else "LOW LOSS RISK",
        )

        if res["risk_category"] == "Low Risk":
            st.success(f"**Underwriting Recommendation:** {res['underwriting_recommendation']}")
        elif res["risk_category"] == "Medium Risk":
            st.warning(f"**Underwriting Recommendation:** {res['underwriting_recommendation']}")
        else:
            st.error(f"**Underwriting Recommendation:** {res['underwriting_recommendation']}")

        st.subheader("Local SHAP Explanation — Top Drivers of This Prediction")
        top_df = pd.DataFrame(res["top_factors"])

        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        top_plot = top_df.iloc[::-1]
        bars_color = ["#eb6834" if v > 0 else "#2a78d6" for v in top_plot["shap_value"]]
        ax.barh(top_plot["feature"], top_plot["shap_value"], color=bars_color, height=0.6)
        ax.axvline(0, color="#0b0b0b", lw=1.0)
        ax.set_xlabel("SHAP Contribution to Log-Odds of Default (+ Increases Risk, - Decreases Risk)")
        ax.set_title("Borrower-Specific SHAP Feature Attributions")
        fig.tight_layout()

        sc1, sc2 = st.columns([1.15, 1.0])
        with sc1:
            st.pyplot(fig)
            plt.close(fig)
        with sc2:
            st.dataframe(
                top_df[["feature", "shap_value", "direction"]].round(4),
                width="stretch",
                hide_index=True,
            )
    except ValueError as err:
        st.error(f"**Input Validation Error:** {err}")
