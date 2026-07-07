"""Evaluation utilities: metrics, cross-validation, and model comparison."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_model(name, fitted_pipeline, X_val, y_val, threshold=0.5) -> dict:
    """Score a fitted pipeline on a held-out split. ROC-AUC and PR-AUC give
    threshold-independent ranking quality; precision/recall/F1 and the
    confusion matrix are computed at the actual operating threshold used
    for decisions.
    """
    y_prob = fitted_pipeline.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    prec_curve, rec_curve, _ = precision_recall_curve(y_val, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    return {
        "model": name,
        "roc_auc": roc_auc_score(y_val, y_prob),
        "pr_auc": auc(rec_curve, prec_curve),
        "accuracy": accuracy_score(y_val, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def cross_validated_auc(pipeline, X, y, n_splits=5, random_state=42) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean(), scores.std()


def compare_models(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results).set_index("model")
    return df.sort_values("roc_auc", ascending=False).round(4)


def plot_roc_comparison(fitted_pipelines: dict, X_val, y_val, save_path=None):
    """fitted_pipelines: {model_name: fitted_pipeline_or_estimator}"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in fitted_pipelines.items():
        RocCurveDisplay.from_estimator(model, X_val, y_val, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curves: Model Comparison")
    ax.legend(fontsize=8, loc="lower right")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison_bar(comparison_df: pd.DataFrame, metric="roc_auc", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    comparison_df[metric].sort_values().plot(kind="barh", ax=ax, color="#2E86AB")
    ax.set_xlabel(metric.upper().replace("_", " "))
    ax.set_title(f"Model Comparison: {metric.upper()}")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
