"""Data ingestion and validation for the loan approval pipeline."""

from pathlib import Path

import pandas as pd

TARGET = "loan_status"
ID_COL = "id"

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


def load_raw_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test CSVs and run basic schema validation."""
    data_dir = Path(data_dir)
    train_path, test_path = data_dir / "train.csv", data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Run `python data/generate_synthetic_data.py` "
            "for a demo dataset, or download the real Kaggle dataset (see README)."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()

    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"train.csv is missing expected columns: {missing}")

    return train, test


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return dtypes, missingness, and cardinality per column."""
    report = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(),
        }
    )
    return report.sort_values("pct_missing", ascending=False)
