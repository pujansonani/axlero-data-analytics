"""Weight of Evidence (WoE) and Information Value (IV) module (Module 3 - Charukesh).

Implements leakage-free binning and Laplace-smoothed WoE/IV computation fitted strictly on training data.

Formulas:
  Dist_Event_i    = (Events_i + alpha) / sum_j(Events_j + alpha)
  Dist_NonEvent_i = (NonEvents_i + alpha) / sum_j(NonEvents_j + alpha)
  WoE_i           = ln(Dist_Event_i / Dist_NonEvent_i)
  IV_i            = (Dist_Event_i - Dist_NonEvent_i) * WoE_i
  IV              = sum_i IV_i

Note on sign convention:
  Using Dist_Event / Dist_NonEvent, positive WoE indicates higher default risk than average,
  and negative WoE indicates lower default risk (higher creditworthiness). Both conventions yield
  the exact same Information Value (IV) because (Dist_Event - Dist_NonEvent) * ln(Dist_Event / Dist_NonEvent) >= 0.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def interpret_iv(iv: float) -> str:
    """Interpret Information Value (IV) according to Siddiqi (2006) credit risk scoring benchmarks."""
    if iv < 0.02:
        return "Not Predictive (< 0.02)"
    if iv < 0.10:
        return "Weak Predictor (0.02 - 0.10)"
    if iv < 0.30:
        return "Medium Predictor (0.10 - 0.30)"
    if iv < 0.50:
        return "Strong Predictor (0.30 - 0.50)"
    return "Very Strong / Check Policy Variable (> 0.50)"


class WoEIVAnalyzer:
    """Fits Weight of Evidence (WoE) bins and Information Value (IV) strictly on training data."""

    def __init__(self, num_bins: int = 10, laplace_alpha: float = 0.5):
        self.num_bins = num_bins
        self.laplace_alpha = laplace_alpha
        self.bin_edges_: Dict[str, np.ndarray] = {}
        self.woe_maps_: Dict[str, Dict[str, float]] = {}
        self.bin_tables_: Dict[str, pd.DataFrame] = {}
        self.iv_summary_: Optional[pd.DataFrame] = None
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []

    def fit(
        self,
        df_train: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
        target_col: str = "loan_status",
    ) -> "WoEIVAnalyzer":
        """Fit quantile bin edges and category WoE mappings strictly on df_train."""
        self.numeric_cols_ = list(numeric_cols)
        self.categorical_cols_ = list(categorical_cols)
        y = df_train[target_col].astype(int)

        iv_records = []

        for col in self.numeric_cols_:
            s = df_train[col]
            # Derive actual value quantile edges cleanly
            quantiles = np.linspace(0.0, 1.0, self.num_bins + 1)
            raw_edges = np.unique(np.nanquantile(s, quantiles))
            if len(raw_edges) < 2:
                raw_edges = np.array([s.min() - 1.0, s.max() + 1.0])
            raw_edges[0] = -np.inf
            raw_edges[-1] = np.inf
            self.bin_edges_[col] = raw_edges

            binned = pd.cut(s, bins=raw_edges, include_lowest=True).astype(str)
            tbl, iv_val, woe_map = self._compute_table(binned, y, feature_name=col, is_numeric=True)
            self.bin_tables_[col] = tbl
            self.woe_maps_[col] = woe_map
            iv_records.append(
                {
                    "feature": col,
                    "type": "Numeric",
                    "num_bins": len(tbl),
                    "iv": round(float(iv_val), 4),
                    "predictive_power": interpret_iv(iv_val),
                }
            )

        for col in self.categorical_cols_:
            binned = df_train[col].fillna("MISSING").astype(str)
            tbl, iv_val, woe_map = self._compute_table(binned, y, feature_name=col, is_numeric=False)
            self.bin_tables_[col] = tbl
            self.woe_maps_[col] = woe_map
            iv_records.append(
                {
                    "feature": col,
                    "type": "Categorical",
                    "num_bins": len(tbl),
                    "iv": round(float(iv_val), 4),
                    "predictive_power": interpret_iv(iv_val),
                }
            )

        self.iv_summary_ = (
            pd.DataFrame(iv_records).sort_values("iv", ascending=False).reset_index(drop=True)
        )
        self.iv_summary_["rank"] = np.arange(1, len(self.iv_summary_) + 1)
        return self

    def _compute_table(
        self, binned_series: pd.Series, y: pd.Series, feature_name: str, is_numeric: bool
    ) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
        """Compute Laplace-smoothed WoE and IV table for a binned feature."""
        df_tmp = pd.DataFrame({"bin": binned_series, "target": y})
        grp = df_tmp.groupby("bin", observed=False)["target"].agg(
            total="count", events="sum"
        ).reset_index()
        grp["non_events"] = grp["total"] - grp["events"]
        grp["event_rate_pct"] = (grp["events"] / grp["total"].clip(lower=1) * 100.0).round(2)
        grp["share_pct"] = (grp["total"] / len(df_tmp) * 100.0).round(2)

        alpha = self.laplace_alpha
        total_events_smoothed = (grp["events"] + alpha).sum()
        total_nonevents_smoothed = (grp["non_events"] + alpha).sum()

        grp["dist_event"] = (grp["events"] + alpha) / total_events_smoothed
        grp["dist_non_event"] = (grp["non_events"] + alpha) / total_nonevents_smoothed
        grp["woe"] = np.log(grp["dist_event"] / grp["dist_non_event"])
        grp["iv_component"] = (grp["dist_event"] - grp["dist_non_event"]) * grp["woe"]
        grp["feature"] = feature_name

        if not is_numeric:
            grp = grp.sort_values("woe", ascending=True).reset_index(drop=True)

        total_iv = float(grp["iv_component"].sum())
        woe_map = dict(zip(grp["bin"].astype(str), grp["woe"].astype(float)))
        return grp, total_iv, woe_map

    def transform_woe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features into their fitted Weight of Evidence (WoE) values."""
        out = pd.DataFrame(index=df.index)
        for col in self.numeric_cols_:
            edges = self.bin_edges_[col]
            binned = pd.cut(df[col], bins=edges, include_lowest=True).astype(str)
            out[f"{col}_woe"] = binned.map(self.woe_maps_[col]).fillna(0.0)
        for col in self.categorical_cols_:
            binned = df[col].fillna("MISSING").astype(str)
            out[f"{col}_woe"] = binned.map(self.woe_maps_[col]).fillna(0.0)
        return out

