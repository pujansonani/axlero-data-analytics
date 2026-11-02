"""Statistical analysis of numerical and categorical credit risk variables (Module 3 - Charukesh)."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def wilson_ci(successes: int, total: int, z: float = 1.95996) -> Tuple[float, float]:
    """Compute 95% Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def analyze_numerical_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: str = "loan_status",
) -> pd.DataFrame:
    """Perform parametric and non-parametric statistical tests on numerical features against loan_status."""
    rows = []
    y = df[target_col].astype(int)
    for col in numeric_cols:
        s = df[col].dropna()
        g0 = df.loc[y == 0, col].dropna()
        g1 = df.loc[y == 1, col].dropna()

        # Mann-Whitney U test
        u_stat, mw_p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        # Rank-biserial correlation effect size: r_rb = 2 * U / (n1 * n0) - 1
        r_rb = (2.0 * u_stat) / (len(g1) * len(g0)) - 1.0

        # Two-sample Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(g1, g0)

        # Welch's t-test
        t_stat, t_p = stats.ttest_ind(g1, g0, equal_var=False)

        # Point-biserial & Spearman correlation
        pb_r, pb_p = stats.pointbiserialr(y.loc[s.index], s)
        sp_r, sp_p = stats.spearmanr(s, y.loc[s.index])

        rows.append(
            {
                "feature": col,
                "overall_mean": round(float(s.mean()), 4),
                "overall_sd": round(float(s.std()), 4),
                "overall_median": round(float(s.median()), 4),
                "overall_iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
                "skewness": round(float(stats.skew(s)), 4),
                "kurtosis": round(float(stats.kurtosis(s)), 4),
                "repaid_mean": round(float(g0.mean()), 4),
                "repaid_median": round(float(g0.median()), 4),
                "default_mean": round(float(g1.mean()), 4),
                "default_median": round(float(g1.median()), 4),
                "welch_t_stat": round(float(t_stat), 4),
                "welch_p_value": float(t_p),
                "mann_whitney_u": float(u_stat),
                "mann_whitney_p_value": float(mw_p),
                "rank_biserial_r": round(float(r_rb), 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_p_value": float(ks_p),
                "spearman_r": round(float(sp_r), 4),
                "point_biserial_r": round(float(pb_r), 4),
                "significant_at_0_01": bool(mw_p < 0.01 and ks_p < 0.01),
            }
        )
    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False).reset_index(drop=True)


def analyze_categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str],
    target_col: str = "loan_status",
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Perform Chi-Square, Cramér's V, and level-wise odds ratio / Wilson CI analysis on categorical variables."""
    summary_rows = []
    level_tables: Dict[str, pd.DataFrame] = {}
    n = len(df)

    for col in categorical_cols:
        ct = pd.crosstab(df[col], df[target_col])
        chi2, p_val, dof, _ = stats.chi2_contingency(ct)
        min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * max(min_dim, 1)))

        summary_rows.append(
            {
                "feature": col,
                "num_categories": int(ct.shape[0]),
                "chi2_statistic": round(float(chi2), 4),
                "degrees_of_freedom": int(dof),
                "p_value": float(p_val),
                "cramers_v": round(float(cramers_v), 4),
                "effect_interpretation": (
                    "Strong (>0.30)"
                    if cramers_v >= 0.30
                    else "Moderate (0.15-0.30)"
                    if cramers_v >= 0.15
                    else "Weak (<0.15)"
                ),
                "significant_at_0_01": bool(p_val < 0.01),
            }
        )

        # Build level-wise table with 95% Wilson CI and odds ratio relative to lowest-risk level
        lvl_df = pd.DataFrame(
            {
                "category": ct.index.astype(str),
                "non_default": ct[0].values,
                "default": ct[1].values,
            }
        )
        lvl_df["total"] = lvl_df["non_default"] + lvl_df["default"]
        lvl_df["share_pct"] = (lvl_df["total"] / n * 100.0).round(2)
        lvl_df["default_rate_pct"] = (lvl_df["default"] / lvl_df["total"] * 100.0).round(2)

        cis = [wilson_ci(int(d), int(t)) for d, t in zip(lvl_df["default"], lvl_df["total"])]
        lvl_df["ci_95_low_pct"] = [round(c[0] * 100.0, 2) for c in cis]
        lvl_df["ci_95_high_pct"] = [round(c[1] * 100.0, 2) for c in cis]

        # Reference category: lowest default rate category
        lvl_df = lvl_df.sort_values("default_rate_pct", ascending=True).reset_index(drop=True)
        ref_odds = (lvl_df.loc[0, "default"] + 0.5) / (lvl_df.loc[0, "non_default"] + 0.5)
        lvl_df["odds_ratio_vs_lowest"] = (
            ((lvl_df["default"] + 0.5) / (lvl_df["non_default"] + 0.5)) / ref_odds
        ).round(2)
        level_tables[col] = lvl_df

    summary_df = (
        pd.DataFrame(summary_rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)
    )
    return summary_df, level_tables
