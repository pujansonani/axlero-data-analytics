"""Model training, cross-validated hyperparameter tuning, and prediction bundle serialization (Module 5 - Sowmya)."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.features.feature_engineering import CreditRiskFeatureEngineer


def build_candidate_models(scale_pos_weight: float, random_state: int = 42) -> Dict[str, Any]:
    """Instantiate baseline, linear, tree, bagging, and boosting models for comparison."""
    return {
        "Dummy Baseline (Stratified)": DummyClassifier(
            strategy="stratified", random_state=random_state
        ),
        "Logistic Regression (L2)": LogisticRegression(
            C=1.0, max_iter=1000, random_state=random_state
        ),
        "Logistic Regression (Balanced)": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "Decision Tree (Tuned)": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=25, random_state=random_state
        ),
        "Random Forest (Tuned)": RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting (Tuned)": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=5,
            min_samples_leaf=15,
            subsample=0.85,
            random_state=random_state,
        ),
        "XGBoost (Balanced)": XGBClassifier(
            n_estimators=300,
            learning_rate=0.06,
            max_depth=6,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost (Champion Tuned)": XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.5,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def tune_hyperparameters_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """Perform Stratified K-Fold hyperparameter tuning for Logistic Regression, Decision Tree, Random Forest, and XGBoost."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    tuning_results = {}

    # 1. Logistic Regression tuning
    lr_search = RandomizedSearchCV(
        LogisticRegression(max_iter=1000, random_state=random_state),
        param_distributions={
            "C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
            "class_weight": [None, "balanced"],
        },
        n_iter=8,
        scoring="roc_auc",
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
    )
    lr_search.fit(X_train, y_train)
    tuning_results["Logistic Regression"] = {
        "best_params": lr_search.best_params_,
        "best_cv_roc_auc": round(float(lr_search.best_score_), 4),
    }

    # 2. Decision Tree tuning
    dt_search = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=random_state),
        param_distributions={
            "max_depth": [5, 7, 8, 10, 12],
            "min_samples_leaf": [10, 20, 25, 40, 50],
            "criterion": ["gini", "entropy"],
        },
        n_iter=10,
        scoring="roc_auc",
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
    )
    dt_search.fit(X_train, y_train)
    tuning_results["Decision Tree"] = {
        "best_params": dt_search.best_params_,
        "best_cv_roc_auc": round(float(dt_search.best_score_), 4),
    }

    # 3. Random Forest tuning
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_distributions={
            "n_estimators": [150, 250],
            "max_depth": [10, 14, 18],
            "min_samples_leaf": [3, 5, 10],
            "class_weight": [None, "balanced_subsample"],
        },
        n_iter=6,
        scoring="roc_auc",
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    tuning_results["Random Forest"] = {
        "best_params": rf_search.best_params_,
        "best_cv_roc_auc": round(float(rf_search.best_score_), 4),
    }

    # 4. Gradient Boosting tuning
    gb_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=random_state),
        param_distributions={
            "n_estimators": [150, 250],
            "learning_rate": [0.05, 0.08, 0.1],
            "max_depth": [4, 5],
            "min_samples_leaf": [10, 15, 25],
            "subsample": [0.85, 1.0],
        },
        n_iter=6,
        scoring="roc_auc",
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
    )
    gb_search.fit(X_train, y_train)
    tuning_results["Gradient Boosting"] = {
        "best_params": gb_search.best_params_,
        "best_cv_roc_auc": round(float(gb_search.best_score_), 4),
    }

    # 5. XGBoost tuning
    xgb_search = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=random_state, n_jobs=-1),
        param_distributions={
            "n_estimators": [250, 350],
            "learning_rate": [0.04, 0.05, 0.08],
            "max_depth": [5, 6, 7],
            "min_child_weight": [3, 4, 6],
            "subsample": [0.8, 0.85, 0.9],
            "colsample_bytree": [0.8, 0.85, 0.9],
        },
        n_iter=6,
        scoring="roc_auc",
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)
    tuning_results["XGBoost"] = {
        "best_params": xgb_search.best_params_,
        "best_cv_roc_auc": round(float(xgb_search.best_score_), 4),
    }

    return tuning_results


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    cv_folds: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Train all candidate models on X_train and compute 5-fold cross-validation ROC-AUC on X_train."""
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = round(neg_count / max(pos_count, 1), 4)

    models = build_candidate_models(scale_pos_weight=scale_pos_weight, random_state=random_state)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_summary: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
        )
        model.fit(X_train, y_train)
        cv_summary[name] = {
            "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        }

    return models, cv_summary


def validate_borrower_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize a single borrower's input dictionary for credit risk prediction."""
    required = {
        "person_age": (18, 100),
        "person_income": (1000, 10_000_000),
        "person_emp_length": (0, 60),
        "loan_amnt": (500, 100_000),
        "loan_int_rate": (1.0, 40.0),
        "cb_person_cred_hist_length": (0, 60),
    }
    cleaned: Dict[str, Any] = {}
    for field, (low, high) in required.items():
        if field not in data or data[field] is None:
            raise ValueError(f"Missing required numeric field: {field}")
        val = float(data[field])
        if np.isnan(val) or val < low or val > high:
            raise ValueError(f"Field '{field}'={val} is outside valid domain [{low}, {high}].")
        cleaned[field] = val

    if cleaned["person_emp_length"] > cleaned["person_age"] - 14:
        raise ValueError(
            f"Employment length ({cleaned['person_emp_length']} yrs) cannot exceed working age ({cleaned['person_age'] - 14} yrs)."
        )
    if cleaned["cb_person_cred_hist_length"] > cleaned["person_age"] - 14:
        raise ValueError(
            f"Credit history length ({cleaned['cb_person_cred_hist_length']} yrs) cannot exceed adult age."
        )

    valid_home = {"RENT", "OWN", "MORTGAGE", "OTHER"}
    home = str(data.get("person_home_ownership", "")).upper()
    if home not in valid_home:
        raise ValueError(f"Invalid person_home_ownership '{home}'. Must be one of {valid_home}")
    cleaned["person_home_ownership"] = home

    valid_intent = {
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    }
    intent = str(data.get("loan_intent", "")).upper()
    if intent not in valid_intent:
        raise ValueError(f"Invalid loan_intent '{intent}'. Must be one of {valid_intent}")
    cleaned["loan_intent"] = intent

    valid_grade = {"A", "B", "C", "D", "E", "F", "G"}
    grade = str(data.get("loan_grade", "")).upper()
    if grade not in valid_grade:
        raise ValueError(f"Invalid loan_grade '{grade}'. Must be one of {valid_grade}")
    cleaned["loan_grade"] = grade

    valid_prior = {"Y", "N"}
    prior = str(data.get("cb_person_default_on_file", "")).upper()
    if prior not in valid_prior:
        raise ValueError(f"Invalid cb_person_default_on_file '{prior}'. Must be 'Y' or 'N'.")
    cleaned["cb_person_default_on_file"] = prior

    # Compute loan_percent_income if not provided or reconcile
    if "loan_percent_income" in data and data["loan_percent_income"] is not None:
        cleaned["loan_percent_income"] = float(data["loan_percent_income"])
    else:
        cleaned["loan_percent_income"] = round(
            cleaned["loan_amnt"] / cleaned["person_income"], 2
        )

    return cleaned


def predict_single_borrower(
    bundle: Dict[str, Any], borrower_input: Dict[str, Any]
) -> Dict[str, Any]:
    """Run validated inference and local SHAP explanation for a single borrower."""
    cleaned = validate_borrower_input(borrower_input)
    df_single = pd.DataFrame([cleaned])

    fe: CreditRiskFeatureEngineer = bundle["feature_engineer"]
    selected_features: List[str] = bundle["selected_features"]
    model = bundle["champion_model"]
    opt_thresh: float = float(bundle.get("optimal_threshold", 0.35))

    X_feat = fe.transform(df_single)[selected_features]
    prob = float(model.predict_proba(X_feat)[0, 1])
    pred_class = int(prob >= opt_thresh)

    if prob < 0.20:
        risk_category = "Low Risk"
        recommendation = "Approve — Standard Underwriting Terms"
    elif prob < 0.50:
        risk_category = "Medium Risk"
        recommendation = "Manual Review / Risk-Adjusted Pricing or Collateral"
    else:
        risk_category = "High Risk"
        recommendation = "Decline or Require Co-Signer & Principal Reduction"

    # Compute local SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_feat)
    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[1])[0]
    else:
        sv = np.array(shap_vals)[0]

    contrib_df = pd.DataFrame(
        {
            "feature": selected_features,
            "scaled_value": X_feat.iloc[0].values,
            "shap_value": sv,
            "abs_shap": np.abs(sv),
            "direction": [
                "Increases Default Risk (+)" if v > 0 else "Decreases Default Risk (-)"
                for v in sv
            ],
        }
    ).sort_values("abs_shap", ascending=False).reset_index(drop=True)

    return {
        "validated_input": cleaned,
        "default_probability": round(prob, 4),
        "default_probability_pct": round(prob * 100.0, 2),
        "decision_threshold": round(opt_thresh, 4),
        "predicted_default": pred_class,
        "risk_category": risk_category,
        "underwriting_recommendation": recommendation,
        "top_factors": contrib_df.head(8).to_dict(orient="records"),
        "all_factors": contrib_df.to_dict(orient="records"),
    }

