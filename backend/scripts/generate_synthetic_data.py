"""Generate a SYNTHETIC loan dataset for development and testing only.

No real dataset was supplied in the workspace. This data is simulated from a
hand-written risk formula; metrics from models trained on it say nothing about
real-world credit performance and must never be presented as such.

The file deliberately contains realistic data problems so the pipeline can be
exercised: missing values, duplicate rows, numbers stored as text, an ID column,
a post-outcome leakage column and an imbalanced, text-labelled target.

Usage:  python scripts/generate_synthetic_data.py [--rows 3000] [--seed 7] [--out data/synthetic_loans.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n_rows: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n_rows)
    income = np.round(rng.lognormal(mean=10.9, sigma=0.5, size=n_rows), -2)
    emp_len = np.clip(rng.normal(6, 4, n_rows), 0, 40).round(1)
    loan_amount = np.round(rng.uniform(1000, 40000, n_rows), -2)
    term = rng.choice([36, 60], n_rows, p=[0.7, 0.3])
    credit_score = np.clip(rng.normal(690, 55, n_rows), 300, 850).round()
    dti = np.clip(rng.normal(18, 8, n_rows), 0, 60).round(2)
    delinq = rng.poisson(0.3, n_rows)
    open_acc = rng.poisson(9, n_rows)
    utilisation = np.clip(rng.beta(2, 3, n_rows) * 100, 0, 100).round(1)
    home = rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], n_rows, p=[0.42, 0.42, 0.14, 0.02])
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "small_business", "car", "medical", "other"],
        n_rows, p=[0.45, 0.2, 0.1, 0.05, 0.08, 0.05, 0.07],
    )
    emp_type = rng.choice(["salaried", "self_employed", "contract", "unemployed"], n_rows, p=[0.65, 0.2, 0.12, 0.03])
    interest = np.clip(28 - (credit_score - 600) * 0.06 + rng.normal(0, 1.5, n_rows), 5, 30).round(2)

    # Hidden "true" risk formula (log-odds) used only for simulation.
    loan_to_income = loan_amount / income
    logit = (
        -2.4
        - 0.012 * (credit_score - 690)
        + 0.035 * (dti - 18)
        + 1.6 * (loan_to_income - 0.3)
        + 0.45 * delinq
        + 0.35 * (term == 60)
        + 0.012 * (utilisation - 40)
        + 0.6 * (purpose == "small_business")
        + 0.25 * (home == "RENT")
        + 1.0 * (emp_type == "unemployed")
        + 0.3 * (emp_type == "contract")
        - 0.02 * (emp_len - 6)
        + rng.normal(0, 0.6, n_rows)
    )
    p_default = 1 / (1 + np.exp(-logit))
    defaulted = rng.random(n_rows) < p_default

    df = pd.DataFrame({
        "loan_id": [f"SYN-{i:06d}" for i in range(n_rows)],
        "age": age,
        "annual_income": income,
        "employment_length_years": emp_len,
        "employment_type": emp_type,
        "home_ownership": home,
        "loan_amount": loan_amount,
        "loan_term_months": term,
        "interest_rate": interest,
        "loan_purpose": purpose,
        "credit_score": credit_score,
        "debt_to_income": dti,
        "credit_utilization_pct": utilisation,
        "num_open_accounts": open_acc,
        "delinquencies_2y": delinq,
        # Post-outcome field: only known after default. Must be excluded from training.
        "total_recovered_amount": np.where(defaulted, np.round(loan_amount * rng.uniform(0.05, 0.4, n_rows), 2), 0.0),
        "loan_outcome": np.where(defaulted, "Defaulted", "Fully Paid"),
    })

    # Inject data-quality problems.
    for col, frac in [("employment_length_years", 0.06), ("annual_income", 0.02), ("credit_utilization_pct", 0.04), ("home_ownership", 0.02)]:
        idx = rng.choice(n_rows, int(frac * n_rows), replace=False)
        df.loc[idx, col] = np.nan
    df["annual_income"] = df["annual_income"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")  # numbers stored as text
    dupes = df.sample(n=max(1, n_rows // 100), random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "synthetic_loans.csv")
    args = parser.parse_args()
    df = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} SYNTHETIC rows to {args.out} (default rate {(df['loan_outcome'] == 'Defaulted').mean():.1%})")


if __name__ == "__main__":
    main()
