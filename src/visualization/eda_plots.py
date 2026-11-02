"""Reproducible visualization generator for EDA (Member 2 - Sriram) and advanced analytics.

Preserves and improves Srirami Reddy Ailuri's 9 core EDA figures while adding
WoE/IV charts, ROC/PR curves, confusion matrices, threshold optimization, calibration, and SHAP plots.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEAL = "#1b9e77"
PURPLE = "#7570b3"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def configure_plot_style() -> None:
    """Apply clean publication-grade matplotlib styling across all project charts."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
        }
    )


def _save_fig(fig: plt.Figure, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _rate_table(df: pd.DataFrame, col: str, order: Optional[List[str]] = None) -> pd.DataFrame:
    g = df.groupby(col, observed=True)["is_default"].agg(["mean", "count"])
    g["rate"] = g["mean"] * 100.0
    if order is not None:
        g = g.reindex(order)
    return g[["rate", "count"]]


def _rate_bars(
    ax: plt.Axes,
    tbl: pd.DataFrame,
    overall_rate: float,
    title: str,
    xlabel: str,
    color: str = ORANGE,
    rotate: int = 0,
) -> None:
    x = np.arange(len(tbl))
    ax.bar(x, tbl["rate"], color=color, width=0.62)
    ax.axhline(
        overall_rate,
        color=MUTED,
        lw=1.1,
        ls="--",
        label=f"Portfolio baseline: {overall_rate:.1f}%",
    )
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        fontsize=8.5,
        handlelength=1.6,
        borderaxespad=0.2,
    )
    for i, (r, n) in enumerate(zip(tbl["rate"], tbl["count"])):
        ax.text(
            i,
            r + 0.8,
            f"{r:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85),
        )
        ax.text(
            i,
            0.8,
            f"n={int(n):,}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="white" if r > 6 else MUTED,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(i) for i in tbl.index],
        rotation=rotate,
        ha="right" if rotate else "center",
    )
    ax.set_ylim(0, max(tbl["rate"].max() * 1.18, overall_rate * 1.3))
    ax.set_ylabel("Default rate (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)


def generate_eda_figures(df_raw_val: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """Generate all 9 core EDA figures from the validated dataset and return verified summary tables."""
    configure_plot_style()
    df = df_raw_val.copy()
    df["is_default"] = df["loan_status"].astype(int)
    overall = float(df["is_default"].mean() * 100.0)

    # Figure 1: Loan status distribution
    counts = df["is_default"].value_counts().reindex([0, 1])
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    labels = ["Repaid / Non-Default (0)", "Default (1)"]
    ax.barh(labels, counts.values, color=[BLUE, ORANGE], height=0.5)
    for i, v in enumerate(counts.values):
        ax.text(
            v + 300,
            i,
            f"{v:,}  ({v / len(df) * 100:.1f}%)",
            va="center",
            color=INK,
            fontweight="medium",
        )
    ax.set_xlim(0, counts.max() * 1.26)
    ax.set_xlabel("Number of loans (n = 32,416)")
    ax.set_title("Figure 1: Target Class Distribution — 21.9% Default Rate (3.57 : 1 Imbalance)")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    _save_fig(fig, out_dir, "fig01_loan_status.png")

    # Figure 2: Default rate by loan grade
    t_grade = _rate_table(df, "loan_grade", list("ABCDEFG"))
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    _rate_bars(
        ax,
        t_grade,
        overall,
        "Figure 2: Default Rate Climbs Monotonically by Grade, Surging at Grade D+",
        "Lender-Assigned Loan Grade (A = Lowest Risk, G = Highest Risk)",
    )
    _save_fig(fig, out_dir, "fig02_default_rate_by_grade.png")

    # Figure 3: Loan burden (loan as % of income)
    df["burden_band"] = pd.cut(
        df["loan_percent_income"],
        [-0.001, 0.10, 0.20, 0.30, 1.0],
        labels=["<= 10%", "10-20%", "20-30%", "> 30%"],
    )
    t_burden = _rate_table(df, "burden_band")
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    _rate_bars(
        ax,
        t_burden,
        overall,
        "Figure 3: Debt-to-Income Burden Above 30% Exhibits a 70.4% Default Rate",
        "Loan Amount as Share of Annual Income (loan_percent_income)",
    )
    _save_fig(fig, out_dir, "fig03_default_rate_by_loan_burden.png")

    # Figure 4: Prior default + home ownership
    t_prior = _rate_table(df, "cb_person_default_on_file", ["N", "Y"])
    t_prior.index = ["No Prior Default (N)", "Prior Default (Y)"]
    t_home = _rate_table(df, "person_home_ownership", ["OWN", "MORTGAGE", "RENT", "OTHER"])
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw={"width_ratios": [2.1, 3.9]})
    _rate_bars(ax[0], t_prior, overall, "Prior Default on File", "Credit Bureau History")
    _rate_bars(ax[1], t_home, overall, "Home Ownership Tenure", "Housing Status")
    fig.suptitle(
        "Figure 4: Credit Bureau History and Home Ownership Strongly Stratify Default Risk",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    _save_fig(fig, out_dir, "fig04_prior_default_and_home_ownership.png")

    # Figure 5: Loan intent
    t_intent = _rate_table(df, "loan_intent").sort_values("rate", ascending=False)
    t_intent.index = [
        i.title()
        .replace("Debtconsolidation", "Debt Consol.")
        .replace("Homeimprovement", "Home Improv.")
        for i in t_intent.index
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    _rate_bars(
        ax,
        t_intent,
        overall,
        "Figure 5: Debt Consolidation and Medical Loans Carry Highest Default Rates",
        "Stated Loan Purpose (Sorted Highest to Lowest Risk)",
    )
    _save_fig(fig, out_dir, "fig05_default_rate_by_loan_intent.png")

    # Figure 6: Income distribution & quartiles
    df["income_q"] = pd.qcut(df["person_income"], 4)
    q_edges = df["income_q"].cat.categories
    q_labels = [
        f"Q1 Lowest\n(<={int(round(q_edges[0].right / 1000))}k)",
        f"Q2\n({int(round(q_edges[1].left / 1000))}-{int(round(q_edges[1].right / 1000))}k)",
        f"Q3\n({int(round(q_edges[2].left / 1000))}-{int(round(q_edges[2].right / 1000))}k)",
        f"Q4 Highest\n(>{int(round(q_edges[3].left / 1000))}k)",
    ]
    t_inc = _rate_table(df, "income_q")
    t_inc.index = q_labels
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.4))
    data = [
        np.log10(df.loc[df.is_default == 0, "person_income"]),
        np.log10(df.loc[df.is_default == 1, "person_income"]),
    ]
    bp = ax[0].boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        medianprops=dict(color=INK, lw=1.5),
    )
    for patch, c in zip(bp["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
        patch.set_edgecolor(c)
    ax[0].set_xticks([1, 2])
    ax[0].set_xticklabels(["Repaid (0)", "Default (1)"])
    ticks = [4000, 10000, 20000, 50000, 100000, 200000]
    ax[0].set_yticks(np.log10(ticks))
    ax[0].set_yticklabels([f"{int(t / 1000)}k" for t in ticks])
    ax[0].set_title("Figure 6a: Annual Income by Outcome (Log10 Scale)")
    ax[0].set_ylabel("Annual Income (Thousands)")
    ax[0].grid(axis="x", visible=False)
    med = df.groupby("is_default")["person_income"].median()
    for i, k in enumerate([0, 1]):
        ax[0].text(
            i + 1.32,
            np.log10(med[k]),
            f"median\n{med[k] / 1000:.1f}k",
            fontsize=8.5,
            color=MUTED,
            va="center",
        )
    _rate_bars(
        ax[1],
        t_inc,
        overall,
        "Figure 6b: Default Rate Across Annual Income Quartiles",
        "Annual Income Quartile Group",
    )
    _save_fig(fig, out_dir, "fig06_income.png")

    # Figure 7: Loan amount + interest rate
    df["amt_band"] = pd.cut(
        df["loan_amnt"],
        [0, 5000, 10000, 15000, 35000],
        labels=["<= 5k", "5-10k", "10-15k", "> 15k"],
    )
    df["rate_band"] = pd.cut(
        df["loan_int_rate"],
        [0, 8, 11, 14, 30],
        labels=["< 8%", "8-11%", "11-14%", "> 14%"],
    )
    t_amt = _rate_table(df, "amt_band")
    t_rate = _rate_table(df, "rate_band")
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3))
    _rate_bars(ax[0], t_amt, overall, "By Loan Principal Amount", "Loan Amount Band")
    _rate_bars(ax[1], t_rate, overall, "By Interest Rate Band", "Interest Rate (%)")
    fig.suptitle(
        "Figure 7: Interest Rate Exhibits a Sharper Monotonic Gradient Than Loan Principal",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    _save_fig(fig, out_dir, "fig07_loan_amount_and_interest_rate.png")

    # Figure 8: Age + employment length
    df["age_band"] = pd.cut(
        df["person_age"], [0, 24, 29, 39, 120], labels=["20-24", "25-29", "30-39", "40+"]
    )
    df["emp_band"] = pd.cut(
        df["person_emp_length"],
        [-1, 1, 4, 9, 100],
        labels=["0-1 yrs", "2-4 yrs", "5-9 yrs", "10+ yrs"],
    )
    t_age = _rate_table(df, "age_band")
    t_emp = _rate_table(df, "emp_band")
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3))
    _rate_bars(ax[0], t_age, overall, "By Applicant Age Band", "Applicant Age (Years)")
    _rate_bars(ax[1], t_emp, overall, "By Employment Tenure", "Years in Current Employment")
    fig.suptitle(
        "Figure 8: Longer Employment Tenure Reduces Default Risk; Age Shows Minimal Variation",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    _save_fig(fig, out_dir, "fig08_age_and_employment.png")

    # Figure 9: Spearman correlation heatmap
    num = df[
        [
            "person_age",
            "person_income",
            "person_emp_length",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "is_default",
        ]
    ].copy()
    num.columns = [
        "Age",
        "Income",
        "Emp. Length",
        "Loan Amount",
        "Interest Rate",
        "Loan % Income",
        "Credit Hist.",
        "Default",
    ]
    corr = num.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=sns.diverging_palette(250, 30, as_cmap=True),
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=1,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Figure 9: Spearman Rank Correlation Matrix Across Numerical Features")
    ax.grid(False)
    _save_fig(fig, out_dir, "fig09_correlation_heatmap.png")

    return {
        "overall_default_rate_pct": round(overall, 2),
        "grade_table": t_grade.round(2).to_dict(orient="index"),
        "burden_table": t_burden.round(2).to_dict(orient="index"),
        "prior_default_table": t_prior.round(2).to_dict(orient="index"),
        "home_ownership_table": t_home.round(2).to_dict(orient="index"),
        "intent_table": t_intent.round(2).to_dict(orient="index"),
        "income_quartile_table": t_inc.round(2).to_dict(orient="index"),
        "loan_amount_table": t_amt.round(2).to_dict(orient="index"),
        "interest_rate_table": t_rate.round(2).to_dict(orient="index"),
        "age_table": t_age.round(2).to_dict(orient="index"),
        "emp_length_table": t_emp.round(2).to_dict(orient="index"),
    }


def plot_woe_iv_figures(
    iv_summary: pd.DataFrame,
    bin_tables: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Generate Figure 10 (IV Feature Ranking) and Figure 11 (WoE Monotonic Curves)."""
    configure_plot_style()

    # Figure 10: Information Value Ranking
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    iv_sorted = iv_summary.sort_values("iv", ascending=True)
    colors = [
        ORANGE if v >= 0.50 else TEAL if v >= 0.30 else BLUE if v >= 0.10 else MUTED
        for v in iv_sorted["iv"]
    ]
    ax.barh(iv_sorted["feature"], iv_sorted["iv"], color=colors, height=0.6)
    for idx, v in enumerate(iv_sorted["iv"]):
        ax.text(v + 0.015, idx, f"{v:.4f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(0.02, color=MUTED, ls=":", lw=1.0, label="Weak (0.02)")
    ax.axvline(0.10, color=BLUE, ls="--", lw=1.0, label="Medium (0.10)")
    ax.axvline(0.30, color=TEAL, ls="--", lw=1.1, label="Strong (0.30)")
    ax.axvline(0.50, color=ORANGE, ls="--", lw=1.1, label="Very Strong / Policy (0.50)")
    ax.set_xlim(0, iv_sorted["iv"].max() * 1.18)
    ax.set_xlabel("Information Value (IV) — Fitted on Training Set (Laplace alpha = 0.5)")
    ax.set_title("Figure 10: Feature Predictive Power Ranking by Information Value (IV)")
    ax.legend(loc="lower right", frameon=True, facecolor="white")
    ax.grid(axis="y", visible=False)
    _save_fig(fig, out_dir, "fig10_woe_iv_summary.png")

    # Figure 11: WoE Profiles for top 4 features
    top_feats = ["loan_grade", "loan_int_rate", "loan_percent_income", "person_income"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    for ax, feat in zip(axes.ravel(), top_feats):
        tbl = bin_tables[feat].copy()
        x = np.arange(len(tbl))
        ax.bar(x, tbl["woe"], color=[ORANGE if w > 0 else BLUE for w in tbl["woe"]], width=0.58, alpha=0.85)
        ax.axhline(0.0, color=INK, lw=0.9, ls="-")
        ax2 = ax.twinx()
        ax2.plot(x, tbl["event_rate_pct"], color=INK, marker="o", lw=1.6, label="Default Rate (%)")
        ax2.set_ylabel("Default Rate (%)", color=INK, fontsize=8.5)
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)

        labels = [str(b)[:14] for b in tbl["bin"]]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8)
        ax.set_ylabel("Weight of Evidence (WoE)")
        ax.set_title(f"{feat} (IV = {tbl['iv_component'].sum():.3f})")
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Figure 11: Weight of Evidence (WoE) & Default Rate Profiles Across Top 4 Risk Drivers",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    _save_fig(fig, out_dir, "fig11_woe_curves.png")

