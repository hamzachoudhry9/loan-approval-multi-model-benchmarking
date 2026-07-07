# Loan Approval Prediction: Multi-Model Benchmarking & Profit-Optimized Decisioning

End-to-end credit risk pipeline that predicts loan default probability, benchmarks nine ML
classifiers, tunes and ensembles the top performers, and converts model scores into a
profit-maximizing approve/reject policy instead of stopping at accuracy or AUC.

**[Open the full notebook](notebooks/loan_approval_prediction.ipynb)**

---

## 1. Business Problem

A lender needs to automate the accept/reject decision on incoming loan applications. The two
possible errors are not equally costly: approving a loan that defaults loses principal, while
rejecting a borrower who would have repaid only gives up interest income. Because these costs
are asymmetric, a model chosen purely on accuracy or ROC-AUC does not necessarily maximize the
lender's profit. This project treats that as the actual objective:

- Modeling objective: rank applicants by default risk as well as possible (ROC-AUC, PR-AUC).
- Business objective: pick the decision threshold, and where useful a segment-specific
  threshold, that maximizes expected profit under an explicit cost structure.

## 2. Dataset

- Source: Kaggle Playground Series S4E10, Loan Approval Prediction.
- Size: ~58k labeled training applications, ~39k unlabeled test applications.
- Target: `loan_status` (1 = default, 0 = repaid).
- Features:
  - Numeric: age, annual income, employment length, loan amount, interest rate,
    loan-to-income ratio, credit history length.
  - Categorical: home ownership, loan intent, loan grade, prior default flag.

This repo does not redistribute the Kaggle data. `data/generate_synthetic_data.py` produces a
schema-matched synthetic sample so the pipeline runs right after cloning. See
[How to Run](#6-how-to-run) for swapping in the real dataset.

## 3. Data Science Lifecycle Covered

| Stage | What's in the notebook |
|---|---|
| Business understanding | Problem framing, cost asymmetry, success metrics (Section 1) |
| Data understanding & EDA | Class balance, missingness, distributions split by target, categorical default rates, correlation heatmap, and a data-driven findings summary (Section 2) |
| Data preparation | Imputation, scaling, one-hot encoding via a single leak-free ColumnTransformer; 3 engineered features with stated rationale (Section 3) |
| Modeling | 9-model benchmark under identical splits and preprocessing (Section 4) |
| Hyperparameter tuning | 5-fold CV grid search on Random Forest and XGBoost (Section 5) |
| Ensembling | Soft-voting and stacking ensembles, benchmarked against the single best tuned model (Section 6) |
| Evaluation | ROC-AUC, PR-AUC, precision, recall, F1, confusion matrices, ROC curve comparison, 5-fold CV variance (Section 7) |
| Business impact | Cost-sensitive threshold sweep, profit curve, segment-specific threshold analysis (Section 8) |
| Conclusions | Limitations and production next steps: calibration, explainability, drift monitoring (Section 9) |

## 4. Key Results

Numbers below are from a full run against the real ~58k-row Kaggle dataset. Running against the
synthetic sample shipped in this repo will produce different but comparable numbers; see the
notebook for the exact figures on whatever data it's run against.

- Best single model: hyperparameter-tuned XGBoost, ROC-AUC around 0.80 on held-out validation.
- 9 classifiers benchmarked under identical preprocessing: Logistic Regression, k-NN, SVM,
  Gaussian Naive Bayes, Decision Tree, Random Forest, AdaBoost, MLP, XGBoost.
- Soft-voting and stacking ensembles benchmarked against the tuned single models; see Section 6
  of the notebook for whether the added complexity is worth it on this dataset.
- Profit-based threshold selection quantifies the gap between the default 0.5 cutoff and the
  cost-aware optimum (Section 8), plus a segment-specific threshold analysis by
  loan_percent_income band.

## 5. Repository Structure

```
.
├── data/
│   ├── generate_synthetic_data.py   # schema-matched demo data generator
│   ├── train.csv                    # generated locally, gitignored
│   └── test.csv                     # generated locally, gitignored
├── notebooks/
│   └── loan_approval_prediction.ipynb
├── src/
│   ├── data_loading.py       # ingestion and data quality report
│   ├── eda.py                 # reusable EDA plotting functions
│   ├── preprocessing.py       # feature engineering and ColumnTransformer
│   ├── train_models.py        # model zoo, tuning, ensembles
│   ├── evaluate_models.py     # metrics, CV, comparison plots
│   └── profit_analysis.py     # cost-sensitive threshold optimization
├── figures/                   # plots saved by the notebook (gitignored)
├── reports/
│   └── Final_Project_Report.pdf
├── requirements.txt
└── README.md
```

## 6. How to Run

```bash
git clone https://github.com/hamzachoudhry9/loan-approval-multi-model-benchmarking.git
cd loan-approval-multi-model-benchmarking
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Option A: run immediately against a synthetic demo dataset
python data/generate_synthetic_data.py

# Option B: reproduce the numbers reported above against the real dataset
# 1. Download "Loan Approval Prediction" (Playground Series S4E10) from Kaggle
# 2. Save as data/train.csv and data/test.csv

jupyter notebook notebooks/loan_approval_prediction.ipynb
```

The notebook imports directly from `src/`, so every transformation and model definition used in
the analysis is version-controlled and testable, not copy-pasted inline.

## 7. Tech Stack

- Language: Python 3.11
- Core libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, jupyter

## 8. Limitations & Next Steps

- Profit and cost weights used for threshold optimization are illustrative, not calibrated
  against a real lender's recovery rates or cost of capital. This is called out directly in the
  notebook (Section 8) rather than presented as fact.
- No temporal validation: the dataset has no application-date field, so evaluation is against a
  random hold-out rather than a genuinely future population.
- Production next steps: probability calibration (CalibratedClassifierCV), SHAP-based
  explanations (needed for fair-lending compliance in real credit decisioning), and
  post-deployment drift monitoring.

---

Originally built as a course project; restructured into a modular, documented pipeline covering
the full data science lifecycle from business framing through profit-based decisioning.
