"""Model evaluation, threshold analysis, calibration, and SHAP explainability (Module 5 - Sowmya)."""

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.visualization.eda_plots import BLUE, INK, MUTED, ORANGE, PURPLE, TEAL, _save_fig, configure_plot_style


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Compute comprehensive binary classification and credit risk scorecard metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks_stat = float(np.max(tpr - fpr))
    gini = 2.0 * roc_auc - 1.0

    specificity = tn / max(tn + fp, 1)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "gini_coefficient": round(gini, 4),
        "ks_statistic": round(ks_stat, 4),
        "brier_score": round(brier, 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_all_models(
    models: Dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_summary: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Evaluate all trained models on validation and held-out test sets."""
    rows = []
    val_probs: Dict[str, np.ndarray] = {}
    test_probs: Dict[str, np.ndarray] = {}

    y_val_arr = y_val.values.astype(int)
    y_test_arr = y_test.values.astype(int)

    for name, model in models.items():
        p_val = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]
        val_probs[name] = p_val
        test_probs[name] = p_test

        m_val = compute_classification_metrics(y_val_arr, p_val, threshold=0.50)
        m_test = compute_classification_metrics(y_test_arr, p_test, threshold=0.50)
        cls_rep = classification_report(
            y_test_arr, (p_test >= 0.50).astype(int), output_dict=True, zero_division=0
        )

        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": cv_summary[name]["cv_roc_auc_mean"],
                "cv_roc_auc_std": cv_summary[name]["cv_roc_auc_std"],
                "val_roc_auc": m_val["roc_auc"],
                "val_pr_auc": m_val["pr_auc"],
                "val_f1": m_val["f1_score"],
                "test_accuracy": m_test["accuracy"],
                "test_precision": m_test["precision"],
                "test_recall": m_test["recall"],
                "test_specificity": m_test["specificity"],
                "test_f1": m_test["f1_score"],
                "test_roc_auc": m_test["roc_auc"],
                "test_pr_auc": m_test["pr_auc"],
                "test_ks_stat": m_test["ks_statistic"],
                "test_gini": m_test["gini_coefficient"],
                "test_brier": m_test["brier_score"],
                "test_tn": m_test["tn"],
                "test_fp": m_test["fp"],
                "test_fn": m_test["fn"],
                "test_tp": m_test["tp"],
                "test_macro_avg_f1": round(float(cls_rep["macro avg"]["f1-score"]), 4),
                "test_weighted_avg_f1": round(float(cls_rep["weighted avg"]["f1-score"]), 4),
            }
        )

    # Sort strictly by validation ROC-AUC (with 5-fold CV ROC-AUC as tie-breaker) to prevent test-set selection leakage
    df_eval = (
        pd.DataFrame(rows)
        .sort_values(["val_roc_auc", "cv_roc_auc_mean"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return df_eval, val_probs, test_probs


def optimize_decision_threshold(
    y_val: np.ndarray,
    val_prob: np.ndarray,
    y_test: np.ndarray,
    test_prob: np.ndarray,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> Dict[str, Any]:
    """Perform threshold sweep on validation probabilities and evaluate optimal thresholds on test set.

    In credit risk lending, missing an actual default (False Negative) incurs principal loss
    typically ~5x larger than the foregone interest margin of declining a good loan (False Positive).
    """
    thresholds = np.linspace(0.05, 0.90, 86)
    sweep_rows = []
    for t in thresholds:
        pred = (val_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        prec = precision_score(y_val, pred, zero_division=0)
        rec = recall_score(y_val, pred, zero_division=0)
        f1 = f1_score(y_val, pred, zero_division=0)
        spec = tn / max(tn + fp, 1)
        youden_j = rec + spec - 1.0
        expected_cost_per_loan = (cost_fn * fn + cost_fp * fp) / len(y_val)
        sweep_rows.append(
            {
                "threshold": round(float(t), 2),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "specificity": round(float(spec), 4),
                "f1_score": round(float(f1), 4),
                "youden_j": round(float(youden_j), 4),
                "expected_cost_per_loan": round(float(expected_cost_per_loan), 4),
            }
        )

    sweep_df = pd.DataFrame(sweep_rows)
    best_f1_row = sweep_df.loc[sweep_df["f1_score"].idxmax()]
    best_j_row = sweep_df.loc[sweep_df["youden_j"].idxmax()]
    best_cost_row = sweep_df.loc[sweep_df["expected_cost_per_loan"].idxmin()]

    t_f1 = float(best_f1_row["threshold"])
    t_cost = float(best_cost_row["threshold"])

    return {
        "default_threshold_0_50_test_metrics": compute_classification_metrics(y_test, test_prob, 0.50),
        "f1_optimal_threshold": t_f1,
        "f1_optimal_val_f1": float(best_f1_row["f1_score"]),
        "f1_optimal_test_metrics": compute_classification_metrics(y_test, test_prob, t_f1),
        "youden_optimal_threshold": float(best_j_row["threshold"]),
        "cost_optimal_threshold_5_to_1": t_cost,
        "cost_optimal_test_metrics": compute_classification_metrics(y_test, test_prob, t_cost),
        "threshold_sweep_table": sweep_df.to_dict(orient="records"),
    }


def generate_evaluation_figures(
    models: Dict[str, Any],
    y_test: pd.Series,
    test_probs: Dict[str, np.ndarray],
    threshold_analysis: Dict[str, Any],
    X_test: pd.DataFrame,
    champion_name: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Generate Figures 12–16: ROC/PR curves, Confusion Matrices, Threshold Optimization, SHAP, and Calibration."""
    configure_plot_style()
    y_true = y_test.values.astype(int)

    # Figure 12: ROC and Precision-Recall Curves
    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.8))
    palette = [ORANGE, BLUE, TEAL, PURPLE, "#e7298a", "#66a61e", "#a6761d", MUTED]
    for (name, probs), col in zip(test_probs.items(), palette):
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = roc_auc_score(y_true, probs)
        lw = 2.4 if name == champion_name else 1.4
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=col, lw=lw)

        prec, rec, _ = precision_recall_curve(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
        ax[1].plot(rec, prec, label=f"{name} (PR-AUC={pr_auc:.3f})", color=col, lw=lw)

    ax[0].plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1.0)
    ax[0].set_title("Figure 12a: Receiver Operating Characteristic (ROC) Curve")
    ax[0].set_xlabel("False Positive Rate (1 - Specificity)")
    ax[0].set_ylabel("True Positive Rate (Recall)")
    ax[0].legend(loc="lower right", fontsize=7.8, frameon=True, facecolor="white")

    baseline_rate = float(y_true.mean())
    ax[1].axhline(baseline_rate, ls="--", color=MUTED, lw=1.0, label=f"No-Skill ({baseline_rate:.2f})")
    ax[1].set_title("Figure 12b: Precision-Recall (PR) Curve on Imbalanced Test Set")
    ax[1].set_xlabel("Recall (Sensitivity)")
    ax[1].set_ylabel("Precision (Positive Predictive Value)")
    ax[1].legend(loc="lower left", fontsize=7.8, frameon=True, facecolor="white")
    _save_fig(fig, out_dir, "fig12_roc_pr_curves.png")

    # Figure 13: Confusion Matrices for Top 4 Models
    top_models = [
        champion_name,
        "Gradient Boosting (Tuned)",
        "Random Forest (Tuned)",
        "Logistic Regression (Balanced)",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.6))
    for ax_cm, m_name in zip(axes, top_models):
        p = test_probs[m_name]
        pred = (p >= 0.50).astype(int)
        cm = confusion_matrix(y_true, pred)
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100.0
        annot = np.array(
            [
                [f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)" for j in range(2)]
                for i in range(2)
            ]
        )
        sns.heatmap(
            cm_pct,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred Repaid (0)", "Pred Default (1)"],
            yticklabels=["Actual Repaid (0)", "Actual Default (1)"],
            ax=ax_cm,
        )
        ax_cm.set_title(m_name, fontsize=9.5)
        ax_cm.grid(False)
    fig.suptitle(
        "Figure 13: Held-Out Test Set Confusion Matrices (n = 4,863; Row-Normalized % & Counts)",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    _save_fig(fig, out_dir, "fig13_confusion_matrices.png")

    # Figure 14: Threshold Optimization Curve for Champion Model
    sweep_df = pd.DataFrame(threshold_analysis["threshold_sweep_table"])
    t_f1 = threshold_analysis["f1_optimal_threshold"]
    t_cost = threshold_analysis["cost_optimal_threshold_5_to_1"]

    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.4))
    ax[0].plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", color=BLUE, lw=2.0)
    ax[0].plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", color=ORANGE, lw=2.0)
    ax[0].plot(sweep_df["threshold"], sweep_df["f1_score"], label="F1-Score", color=TEAL, lw=2.4)
    ax[0].axvline(t_f1, color=TEAL, ls="--", lw=1.2, label=f"F1-Optimal ({t_f1:.2f})")
    ax[0].axvline(0.50, color=MUTED, ls=":", lw=1.1, label="Default (0.50)")
    ax[0].set_title("Figure 14a: Precision, Recall & F1-Score vs Decision Threshold")
    ax[0].set_xlabel("Default Probability Decision Threshold")
    ax[0].set_ylabel("Metric Value")
    ax[0].legend(loc="lower left", fontsize=8.5, frameon=True, facecolor="white")

    ax[1].plot(
        sweep_df["threshold"],
        sweep_df["expected_cost_per_loan"],
        color=PURPLE,
        lw=2.3,
        label="Expected Cost (FN=5x, FP=1x)",
    )
    ax[1].axvline(t_cost, color=ORANGE, ls="--", lw=1.3, label=f"Cost-Optimal ({t_cost:.2f})")
    ax[1].set_title("Figure 14b: Asymmetric Credit Loss Minimization (FN:FP = 5:1)")
    ax[1].set_xlabel("Default Probability Decision Threshold")
    ax[1].set_ylabel("Expected Relative Loss per Applicant")
    ax[1].legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="white")
    _save_fig(fig, out_dir, "fig14_threshold_optimization.png")

    # Figure 15: SHAP Explainability on Test Sample
    champ_model = models[champion_name]
    X_sample = X_test.iloc[:800].copy()
    explainer = shap.TreeExplainer(champ_model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1])
    else:
        sv = np.array(shap_values)

    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_df = (
        pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": np.round(mean_abs_shap, 4)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    top_shap = shap_df.head(15).sort_values("mean_abs_shap", ascending=True)
    ax.barh(top_shap["feature"], top_shap["mean_abs_shap"], color=ORANGE, height=0.62)
    for idx, val in enumerate(top_shap["mean_abs_shap"]):
        ax.text(val + 0.01, idx, f"{val:.4f}", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, top_shap["mean_abs_shap"].max() * 1.18)
    ax.set_xlabel("Mean |SHAP Value| (Average Impact on Log-Odds of Default)")
    ax.set_title(f"Figure 15: Global SHAP Feature Importance — {champion_name}")
    ax.grid(axis="y", visible=False)
    _save_fig(fig, out_dir, "fig15_shap_summary.png")

    # Figure 16: Probability Calibration Curve
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for m_name, col in [
        (champion_name, ORANGE),
        ("Gradient Boosting (Tuned)", TEAL),
        ("Random Forest (Tuned)", BLUE),
        ("Logistic Regression (L2)", PURPLE),
    ]:
        prob_pos = test_probs[m_name]
        frac_pos, mean_pred = calibration_curve(y_true, prob_pos, n_bins=10)
        brier = brier_score_loss(y_true, prob_pos)
        ax.plot(mean_pred, frac_pos, marker="o", lw=1.8, color=col, label=f"{m_name} (Brier={brier:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1.1, label="Perfectly Calibrated")
    ax.set_title("Figure 16: Probability Reliability & Calibration Curve (Test Set)")
    ax.set_xlabel("Mean Predicted Probability of Default")
    ax.set_ylabel("Observed Fraction of Defaults")
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="white")
    _save_fig(fig, out_dir, "fig16_calibration_curve.png")

    return shap_df

