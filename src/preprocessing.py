"""Preprocessing and feature engineering.

Everything runs through a single sklearn ColumnTransformer so the same
fitted pipeline object is applied to validation, test, and production data.
No logic gets reimplemented by hand in more than one place.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that capture repayment capacity."""
    out = df.copy()

    # loan_percent_income should already be in the raw data; recompute it
    # defensively in case a future data pull drops the column.
    if "loan_percent_income" not in out.columns:
        out["loan_percent_income"] = out["loan_amnt"] / out["person_income"].replace(0, np.nan)

    # Income headroom left after servicing the loan, as a fraction of income.
    out["income_after_loan_pct"] = 1 - out["loan_percent_income"]

    # Interest burden relative to loan size (rate normalized by amount band).
    out["rate_per_1k_loan"] = out["loan_int_rate"] / (out["loan_amnt"] / 1000).replace(0, np.nan)

    # Credit history relative to age, a proxy for financial maturity.
    out["cred_hist_to_age"] = out["cb_person_cred_hist_length"] / out["person_age"].replace(0, np.nan)

    return out


def build_preprocessing_pipeline(numeric_features, categorical_features) -> ColumnTransformer:
    """Median-impute + scale numeric features, mode-impute + one-hot encode
    categoricals. Imputation always runs before scaling/encoding, and
    unseen categories at inference time are handled via handle_unknown='ignore'
    instead of raising.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
