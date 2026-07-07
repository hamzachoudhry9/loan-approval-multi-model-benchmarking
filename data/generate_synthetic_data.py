"""
Generates a synthetic loan-application dataset that mirrors the schema of the
Kaggle Playground Series S4E10 "Loan Approval Prediction" dataset.

The real competition data (~58k rows) isn't redistributed in this repo for
licensing reasons. This script creates a structurally realistic stand-in
(same columns, same dtypes, same rough relationships between features and
default risk) so contributors can clone the repo and run the full notebook
right away, and so CI has something deterministic to test against.

To reproduce the numbers reported in the README, download the real dataset
from Kaggle (see README, Reproducing Results) and drop train.csv/test.csv
into this data/ folder. Every downstream script reads from here and doesn't
care whether the data is real or synthetic.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_loan_data(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    person_age = rng.integers(20, 70, n_rows)
    person_income = rng.lognormal(mean=10.8, sigma=0.55, size=n_rows).round(0)
    person_income = np.clip(person_income, 8000, 500000)

    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN", "OTHER"], size=n_rows, p=[0.45, 0.40, 0.12, 0.03]
    )
    emp_length = np.clip(rng.exponential(scale=5, size=n_rows), 0, 40).round(1)

    loan_intent = rng.choice(
        ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
        size=n_rows,
    )
    loan_grade = rng.choice(
        ["A", "B", "C", "D", "E", "F", "G"], size=n_rows, p=[0.30, 0.27, 0.20, 0.12, 0.06, 0.03, 0.02]
    )
    grade_rate_base = {"A": 7.5, "B": 10.5, "C": 13.5, "D": 16.5, "E": 19.0, "F": 21.5, "G": 24.0}
    loan_int_rate = np.array([grade_rate_base[g] for g in loan_grade]) + rng.normal(0, 1.2, n_rows)
    loan_int_rate = np.clip(loan_int_rate, 5, 30).round(2)

    loan_amnt = np.clip(rng.lognormal(mean=9.2, sigma=0.5, size=n_rows), 500, 40000).round(0)
    loan_percent_income = np.clip(loan_amnt / person_income, 0.01, 1.5).round(4)

    default_on_file = rng.choice(["Y", "N"], size=n_rows, p=[0.18, 0.82])
    cred_hist_length = np.clip(
        (person_age - rng.integers(18, 25, n_rows)).astype(float) + rng.normal(0, 2, n_rows), 0, None
    ).round(0)

    # Latent risk score drives the target so relationships are learnable but
    # noisy, similar to the mix of signal and noise in real credit data.
    grade_penalty = {"A": -1.5, "B": -0.8, "C": 0.0, "D": 0.8, "E": 1.6, "F": 2.4, "G": 3.2}
    z = (
        -0.00002 * (person_income - 60000)
        + 2.2 * loan_percent_income
        + 0.10 * loan_int_rate
        + np.array([grade_penalty[g] for g in loan_grade])
        + np.where(default_on_file == "Y", 1.4, 0.0)
        - 0.03 * emp_length
        - 0.02 * cred_hist_length
        + rng.normal(0, 1.0, n_rows)
        - 2.3
    )
    p_default = _sigmoid(z)
    loan_status = rng.binomial(1, p_default)  # 1 = default/bad, 0 = repaid/good

    df = pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "person_age": person_age,
            "person_income": person_income.astype(int),
            "person_home_ownership": home_ownership,
            "person_emp_length": emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt.astype(int),
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": default_on_file,
            "cb_person_cred_hist_length": cred_hist_length.astype(int),
            "loan_status": loan_status,
        }
    )

    # small amount of realistic missingness
    for col, frac in [("person_emp_length", 0.02), ("loan_int_rate", 0.03)]:
        mask = rng.random(n_rows) < frac
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    full = generate_loan_data(n_rows=8000)
    train = full.sample(frac=0.8, random_state=42).sort_values("id")
    test = full.drop(train.index).drop(columns=["loan_status"]).sort_values("id")

    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    print(f"Wrote data/train.csv {train.shape} and data/test.csv {test.shape}")
