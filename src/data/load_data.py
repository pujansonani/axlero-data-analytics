"""Data loading, schema verification, and stratified train/validation/test splitting."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_config, resolve_path

REQUIRED_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that all required credit risk dataset columns exist and dataframe is non-empty."""
    if df is None or df.empty:
        raise ValueError("Loaded DataFrame is empty.")
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the immutable raw credit risk dataset (32,581 rows x 12 columns)."""
    if path is None:
        cfg = load_config()
        path = resolve_path(cfg["paths"]["raw_data"])
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def load_validated_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the cleaned and validated dataset (32,416 rows x 12 columns)."""
    if path is None:
        cfg = load_config()
        path = resolve_path(cfg["paths"]["processed_validated_data"])
    if not Path(path).exists():
        raise FileNotFoundError(f"Validated dataset not found at {path}")
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a structured summary of dataset shape, missing values, duplicates, and target distribution."""
    validate_schema(df)
    target_counts = df["loan_status"].value_counts().to_dict()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "exact_duplicates_keep_first": int(df.duplicated(keep="first").sum()),
        "all_involved_duplicates_keep_false": int(df.duplicated(keep=False).sum()),
        "missing_counts": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        "missing_pct": {
            k: round(float(v) / len(df) * 100, 4)
            for k, v in df.isnull().sum().to_dict().items()
        },
        "target_distribution": {
            "non_default_0": int(target_counts.get(0, 0)),
            "default_1": int(target_counts.get(1, 0)),
            "default_rate_pct": round(float(df["loan_status"].mean() * 100), 4),
        },
    }


def stratified_train_val_test_split(
    df: pd.DataFrame,
    target_col: str = "loan_status",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into stratified train, validation, and test sets without data leakage."""
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    df_train, df_temp = train_test_split(
        df,
        train_size=train_size,
        stratify=df[target_col],
        random_state=random_state,
    )
    relative_val_size = val_size / (val_size + test_size)
    df_val, df_test = train_test_split(
        df_temp,
        train_size=relative_val_size,
        stratify=df_temp[target_col],
        random_state=random_state,
    )
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )

