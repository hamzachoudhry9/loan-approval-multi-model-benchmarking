"""Smoke tests for the loan approval pipeline.

Run with: pytest tests/ -v

These check that each pipeline stage produces output of the expected
shape and type. A refactor that breaks the notebook should fail here in
seconds instead of after a full notebook execution.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loading import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, data_quality_report
from src.preprocessing import build_preprocessing_pipeline, engineer_features
from src.profit_analysis import best_threshold, profit_threshold_sweep
from src.train_models import get_candidate_models

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def sample_data():
    from data.generate_synthetic_data import generate_loan_data
    return generate_loan_data(n_rows=500, seed=1)


def test_engineer_features_adds_expected_columns(sample_data):
    out = engineer_features(sample_data)
    for col in ["income_after_loan_pct", "rate_per_1k_loan", "cred_hist_to_age"]:
        assert col in out.columns
    assert len(out) == len(sample_data)


def test_data_quality_report_shape(sample_data):
    report = data_quality_report(sample_data)
    assert set(["dtype", "n_missing", "pct_missing", "n_unique"]).issubset(report.columns)
    assert len(report) == sample_data.shape[1]


def test_preprocessing_pipeline_fits_and_transforms(sample_data):
    df = engineer_features(sample_data)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    extra = ["income_after_loan_pct", "rate_per_1k_loan", "cred_hist_to_age"]
    pre = build_preprocessing_pipeline(NUMERIC_FEATURES + extra, CATEGORICAL_FEATURES)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1
    )
    transformed = pre.fit_transform(X_train, y_train)
    assert transformed.shape[0] == len(X_train)
    assert not pd.DataFrame(transformed).isna().any().any()


def test_candidate_models_train_and_predict_proba(sample_data):
    df = engineer_features(sample_data)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    extra = ["income_after_loan_pct", "rate_per_1k_loan", "cred_hist_to_age"]
    pre = build_preprocessing_pipeline(NUMERIC_FEATURES + extra, CATEGORICAL_FEATURES)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = get_candidate_models(scale_pos_weight)
    assert len(models) >= 8  # 9 if xgboost is installed

    pipe = Pipeline([("preprocess", pre), ("clf", models["Logistic Regression"])])
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_valid)[:, 1]
    assert probs.shape[0] == len(X_valid)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_profit_threshold_sweep_finds_valid_optimum():
    import numpy as np
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_prob = np.clip(y_true * 0.6 + rng.normal(0, 0.25, 200) + 0.2, 0, 1)
    sweep = profit_threshold_sweep(y_true, y_prob)
    best = best_threshold(sweep)
    assert 0 < best["threshold"] < 1
    assert best["profit"] == sweep["profit"].max()
