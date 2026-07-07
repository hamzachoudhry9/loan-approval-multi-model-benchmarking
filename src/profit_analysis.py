"""Cost-sensitive, profit-based threshold selection and segment analysis.

Translates model scores into an approve/reject policy by attaching a cost
to each confusion-matrix cell, then sweeping the decision threshold to find
the point that maximizes expected profit instead of accuracy or F1.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def profit_threshold_sweep(
    y_true,
    y_prob,
    gain_tp: float = 1.0,
    loss_fp: float = -5.0,
    loss_fn: float = -0.5,
    gain_tn: float = 0.0,
    n_thresholds: int = 99,
) -> pd.DataFrame:
    """Sweep decision thresholds and compute expected profit at each.

    Assumes y=1 is "default" (bad) and y=0 is "repaid" (good):
      TP: correctly reject a defaulter -> gain_tp
      FP: reject a good borrower -> loss_fp
      FN: approve a defaulter, loan goes bad -> loss_fn
      TN: correctly approve a good borrower -> gain_tn

    Check the sign/label convention against how loan_status is encoded in
    your data before trusting the dollar figures.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        profit = tp * gain_tp + fp * loss_fp + fn * loss_fn + tn * gain_tn
        rows.append({"threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "profit": profit})
    return pd.DataFrame(rows)


def best_threshold(sweep_df: pd.DataFrame) -> pd.Series:
    return sweep_df.loc[sweep_df["profit"].idxmax()]


def plot_profit_curve(sweep_df: pd.DataFrame, save_path=None):
    best = best_threshold(sweep_df)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep_df["threshold"], sweep_df["profit"], color="#2E86AB")
    ax.axvline(best["threshold"], color="red", linestyle="--",
               label=f"optimal thr={best['threshold']:.2f} (profit={best['profit']:.1f})")
    ax.axvline(0.5, color="gray", linestyle=":", label="default thr=0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected profit (units)")
    ax.set_title("Profit Curve vs. Decision Threshold")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def segment_profit_surface(
    df: pd.DataFrame,
    segment_col: str,
    y_true_col: str,
    y_prob_col: str,
    **cost_kwargs,
) -> pd.DataFrame:
    """Run the profit sweep separately within each segment (for example a
    loan_percent_income band) to check whether a single global threshold
    leaves profit on the table compared to segment-specific thresholds.
    """
    results = []
    for segment, g in df.groupby(segment_col):
        sweep = profit_threshold_sweep(g[y_true_col], g[y_prob_col], **cost_kwargs)
        best = best_threshold(sweep)
        results.append({
            "segment": segment, "n": len(g),
            "best_threshold": best["threshold"], "best_profit": best["profit"],
            "profit_per_applicant": best["profit"] / len(g),
        })
    return pd.DataFrame(results).set_index("segment")
