"""Exploratory data analysis utilities.

Plain functions instead of one-off notebook cells, so EDA can be reused,
tested, and rerun against a refreshed data pull without copy-pasting code.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")


def _savefig(fig, path: str | Path | None):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_target_balance(df: pd.DataFrame, target: str, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[target].value_counts().sort_index()
    pcts = (counts / counts.sum() * 100).round(1)
    bars = ax.bar(counts.index.astype(str), counts.values, color=["#2E86AB", "#E4572E"])
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct}%",
                ha="center", va="bottom")
    ax.set_title("Target Class Balance (0 = repaid, 1 = default)")
    ax.set_xlabel(target)
    ax.set_ylabel("Applicants")
    _savefig(fig, save_path)
    return fig


def plot_missingness(df: pd.DataFrame, save_path=None):
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(missing) == 0:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    else:
        (missing * 100).plot(kind="barh", ax=ax, color="#6A4C93")
        ax.set_xlabel("% missing")
        ax.set_title("Missingness by Column")
    _savefig(fig, save_path)
    return fig


def plot_numeric_distributions(df: pd.DataFrame, numeric_cols, target=None, save_path=None):
    n = len(numeric_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        vals = df[col].dropna()
        lo, hi = np.percentile(vals, [1, 99])
        clipped = df[col].clip(lo, hi)
        if target and target in df.columns:
            sns.histplot(x=clipped, hue=df[target], bins=30, ax=ax, element="step",
                         stat="density", common_norm=False)
        else:
            sns.histplot(x=clipped, bins=30, ax=ax)
        ax.set_title(col)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Numeric Feature Distributions (1st to 99th percentile, split by target)", y=1.02)
    _savefig(fig, save_path)
    return fig


def plot_categorical_default_rates(df: pd.DataFrame, categorical_cols, target, save_path=None):
    n = len(categorical_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    overall_rate = df[target].mean()
    for ax, col in zip(axes, categorical_cols):
        rates = df.groupby(col)[target].mean().sort_values(ascending=False)
        rates.plot(kind="bar", ax=ax, color="#2E86AB")
        ax.axhline(overall_rate, color="red", linestyle="--", linewidth=1,
                   label=f"overall={overall_rate:.2f}")
        ax.set_title(f"Default Rate by {col}")
        ax.set_ylabel("Default rate")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols, target, save_path=None):
    corr = df[numeric_cols + [target]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
               cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation Heatmap (Numeric Features + Target)")
    _savefig(fig, save_path)
    return fig


def summarize_eda_findings(df: pd.DataFrame, numeric_cols, categorical_cols, target) -> dict:
    """Compute the headline numbers used in the EDA writeup, so the notebook
    prose and README stay in sync with whatever data was actually loaded.
    """
    default_rate = df[target].mean()
    corr_with_target = (
        df[numeric_cols + [target]].corr()[target].drop(target).sort_values(key=abs, ascending=False)
    )
    cat_spread = {
        col: (df.groupby(col)[target].mean().max() - df.groupby(col)[target].mean().min())
        for col in categorical_cols
    }
    return {
        "n_rows": len(df),
        "n_features": len(numeric_cols) + len(categorical_cols),
        "default_rate": round(float(default_rate), 4),
        "top_numeric_correlations": corr_with_target.round(3).to_dict(),
        "categorical_default_rate_spread": {k: round(float(v), 3) for k, v in cat_spread.items()},
        "pct_missing_any": round(float(df.isna().any(axis=1).mean() * 100), 2),
    }
