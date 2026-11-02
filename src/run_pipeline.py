"""End-to-end execution script for the Axlero Credit Risk Scoring & Loan Default Analysis pipeline."""

import joblib

from src.data.load_data import (
    load_raw_data,
    stratified_train_val_test_split,
    summarize_dataset,
)
from src.evaluation.evaluate_model import (
    evaluate_all_models,
    generate_evaluation_figures,
    optimize_decision_threshold,
)
from src.features.feature_engineering import (
    CreditRiskFeatureEngineer,
    ENGINEERED_FEATURE_DOCS,
    compute_vif_table,
    rank_and_select_features,
)
from src.features.statistical_tests import (
    analyze_categorical_features,
    analyze_numerical_features,
)
from src.features.woe_iv import WoEIVAnalyzer
from src.models.train_model import (
    predict_single_borrower,
    train_all_models,
    tune_hyperparameters_cv,
)
from src.preprocessing.cleaner import (
    analyze_loan_to_income_mismatches,
    apply_member1_imputation,
    compare_imputation_methods,
    deduplicate_and_invalidate,
)
from src.utils import load_config, resolve_path, save_json, set_random_seed
from src.visualization.eda_plots import generate_eda_figures, plot_woe_iv_figures


def run_complete_pipeline() -> None:
    """Execute the full credit risk analytics pipeline from raw data to serialized models."""
    cfg = load_config()
    seed = int(cfg["project"]["random_seed"])
    set_random_seed(seed)

    print("1. Loading raw dataset...")
    df_raw = load_raw_data()
    raw_summary = summarize_dataset(df_raw)

    print("2. Running deduplication, invalid record handling, and mismatch audit...")
    df_interim, cleaning_audit = deduplicate_and_invalidate(
        df_raw,
        max_valid_age=cfg["preprocessing"]["max_valid_age"],
        max_valid_emp_length=cfg["preprocessing"]["max_valid_emp_length"],
    )
    interim_path = resolve_path(cfg["paths"]["interim_cleaned_data"])
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    df_interim.to_csv(interim_path, index=False)

    print("3. Comparing missing value imputation strategies (Global Median vs Grade Median vs KNN)...")
    imputation_comparison = compare_imputation_methods(df_interim, random_state=seed)

    print("4. Generating Member 1 compatible validated dataset and auditing 1,005 ratio mismatches...")
    df_validated = apply_member1_imputation(df_interim)
    validated_path = resolve_path(cfg["paths"]["processed_validated_data"])
    validated_path.parent.mkdir(parents=True, exist_ok=True)
    df_validated.to_csv(validated_path, index=False)
    mismatch_audit = analyze_loan_to_income_mismatches(df_validated)

    save_json(
        {
            "raw_summary": raw_summary,
            "cleaning_audit": cleaning_audit,
            "imputation_comparison": imputation_comparison,
            "mismatch_audit": mismatch_audit,
        },
        resolve_path(cfg["paths"]["imputation_comparison_path"]),
    )

    print("5. Generating EDA figures (fig01-fig09)...")
    figures_dir = resolve_path(cfg["paths"]["figures_dir"])
    eda_summary = generate_eda_figures(df_validated, figures_dir)

    print("6. Performing Statistical Analysis (Mann-Whitney, KS, Welch t, Chi-Square, Cramer's V)...")
    num_cols = cfg["data_schema"]["numeric_columns"]
    cat_cols = cfg["data_schema"]["categorical_columns"]
    num_stats_df = analyze_numerical_features(df_validated, num_cols)
    cat_stats_df, cat_level_tables = analyze_categorical_features(df_validated, cat_cols)

    save_json(
        {
            "eda_summary": eda_summary,
            "numerical_tests": num_stats_df.to_dict(orient="records"),
            "categorical_tests": cat_stats_df.to_dict(orient="records"),
            "categorical_level_tables": {
                k: v.to_dict(orient="records") for k, v in cat_level_tables.items()
            },
        },
        resolve_path(cfg["paths"]["statistical_summary_path"]),
    )

    print("7. Performing Stratified Train / Validation / Test split (70% / 15% / 15%) on interim data...")
    # Split on df_interim (prior to imputation) so that all imputation, WoE, and feature scaling
    # are fitted strictly on df_train without any data leakage!
    df_train, df_val, df_test = stratified_train_val_test_split(
        df_interim,
        target_col="loan_status",
        train_size=cfg["splitting"]["train_size"],
        val_size=cfg["splitting"]["val_size"],
        test_size=cfg["splitting"]["test_size"],
        random_state=seed,
    )

    print("8. Computing Weight of Evidence (WoE) and Information Value (IV) on training split...")
    # Use training set imputed with grade_median for WoE/IV analysis
    fe_temp = CreditRiskFeatureEngineer(imputation_strategy="grade_median", scale_numeric=False).fit(df_train)
    df_train_imp = fe_temp.imputer_.transform(df_train)
    woe_analyzer = WoEIVAnalyzer(
        num_bins=cfg["woe_iv"]["num_bins"],
        laplace_alpha=cfg["woe_iv"]["laplace_smoothing"],
    ).fit(df_train_imp, num_cols, cat_cols, target_col="loan_status")

    plot_woe_iv_figures(woe_analyzer.iv_summary_, woe_analyzer.bin_tables_, figures_dir)
    save_json(
        {
            "iv_ranking": woe_analyzer.iv_summary_.to_dict(orient="records"),
            "bin_tables": {
                k: v.round(4).to_dict(orient="records")
                for k, v in woe_analyzer.bin_tables_.items()
            },
        },
        resolve_path(cfg["paths"]["woe_iv_summary_path"]),
    )

    print("9. Fitting Feature Engineering Pipeline on training set and checking VIF...")
    fe = CreditRiskFeatureEngineer(imputation_strategy="grade_median", scale_numeric=True)
    fe.fit(df_train.drop(columns=["loan_status"]), df_train["loan_status"])

    X_train_full = fe.transform(df_train.drop(columns=["loan_status"]))
    y_train = df_train["loan_status"].astype(int)
    X_val_full = fe.transform(df_val.drop(columns=["loan_status"]))
    y_val = df_val["loan_status"].astype(int)
    X_test_full = fe.transform(df_test.drop(columns=["loan_status"]))
    y_test = df_test["loan_status"].astype(int)

    vif_before = compute_vif_table(X_train_full)
    feature_ranking_df, selected_features = rank_and_select_features(
        X_train_full, y_train, random_state=seed
    )
    vif_after = compute_vif_table(X_train_full[selected_features])

    X_train = X_train_full[selected_features].copy()
    X_val = X_val_full[selected_features].copy()
    X_test = X_test_full[selected_features].copy()

    # Save processed splits
    train_out = X_train.copy()
    train_out["loan_status"] = y_train.values
    val_out = X_val.copy()
    val_out["loan_status"] = y_val.values
    test_out = X_test.copy()
    test_out["loan_status"] = y_test.values

    train_out.to_csv(resolve_path(cfg["paths"]["train_data"]), index=False)
    val_out.to_csv(resolve_path(cfg["paths"]["val_data"]), index=False)
    test_out.to_csv(resolve_path(cfg["paths"]["test_data"]), index=False)

    print("10. Tuning hyperparameters via 5-Fold Stratified CV and training all models...")
    tuning_results = tune_hyperparameters_cv(X_train, y_train, random_state=seed, cv_folds=5)
    models, cv_summary = train_all_models(X_train, y_train, random_state=seed, cv_folds=5)

    print("11. Evaluating models on Validation and Held-Out Test Sets...")
    eval_df, val_probs, test_probs = evaluate_all_models(
        models, X_val, y_val, X_test, y_test, cv_summary
    )
    champion_name = str(eval_df.iloc[0]["model"])
    champion_model = models[champion_name]
    print(f"    Champion Model: {champion_name} (Test ROC-AUC = {eval_df.iloc[0]['test_roc_auc']:.4f})")

    print("12. Performing Threshold Analysis & Generating Evaluation Figures (fig12-fig16)...")
    threshold_analysis = optimize_decision_threshold(
        y_val.values,
        val_probs[champion_name],
        y_test.values,
        test_probs[champion_name],
    )
    shap_importance_df = generate_evaluation_figures(
        models,
        y_test,
        test_probs,
        threshold_analysis,
        X_test,
        champion_name,
        figures_dir,
    )

    save_json(threshold_analysis, resolve_path(cfg["paths"]["threshold_analysis_path"]))
    save_json(
        {
            "champion_model_name": champion_name,
            "split_sizes": {
                "train_rows": len(df_train),
                "val_rows": len(df_val),
                "test_rows": len(df_test),
                "train_default_rate_pct": round(float(y_train.mean() * 100), 2),
                "val_default_rate_pct": round(float(y_val.mean() * 100), 2),
                "test_default_rate_pct": round(float(y_test.mean() * 100), 2),
            },
            "engineered_features_docs": ENGINEERED_FEATURE_DOCS,
            "feature_ranking": feature_ranking_df.to_dict(orient="records"),
            "selected_features": selected_features,
            "vif_before_selection": vif_before.to_dict(orient="records"),
            "vif_after_selection": vif_after.to_dict(orient="records"),
            "hyperparameter_tuning": tuning_results,
            "model_comparison": eval_df.to_dict(orient="records"),
            "shap_global_importance": shap_importance_df.to_dict(orient="records"),
        },
        resolve_path(cfg["paths"]["metrics_summary_path"]),
    )

    print("13. Saving serialized model and pipeline bundles...")
    bundle = {
        "feature_engineer": fe,
        "selected_features": selected_features,
        "champion_model_name": champion_name,
        "champion_model": champion_model,
        "optimal_threshold": threshold_analysis["f1_optimal_threshold"],
        "cost_optimal_threshold": threshold_analysis["cost_optimal_threshold_5_to_1"],
        "model_comparison": eval_df.to_dict(orient="records"),
    }
    joblib.dump(bundle, resolve_path(cfg["paths"]["pipeline_path"]))
    joblib.dump(champion_model, resolve_path(cfg["paths"]["best_model_path"]))

    # Smoke-test single borrower prediction
    sample_pred = predict_single_borrower(
        bundle,
        {
            "person_age": 27,
            "person_income": 55000,
            "person_home_ownership": "RENT",
            "person_emp_length": 4.0,
            "loan_intent": "DEBTCONSOLIDATION",
            "loan_grade": "C",
            "loan_amnt": 12000,
            "loan_int_rate": 13.48,
            "loan_percent_income": 0.22,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 5,
        },
    )
    print(
        f"14. Pipeline complete! Sample prediction: PD={sample_pred['default_probability_pct']}% ({sample_pred['risk_category']})"
    )


if __name__ == "__main__":
    run_complete_pipeline()

