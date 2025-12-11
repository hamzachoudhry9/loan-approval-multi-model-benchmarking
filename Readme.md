# Loan Approval Prediction: Multi-Model Benchmarking and Ensembles

End-to-end machine learning pipeline for predicting bank loan approval decisions from customer demographics, financial attributes, and credit history. The project benchmarks multiple models and builds cost-aware ensembles that align predictions with business risk and profit.

---

## 1. Project Overview

This project uses a structured tabular dataset of historical loan applications to predict whether a new application should be approved or rejected. The pipeline covers:

- Data ingestion and cleaning
- Exploratory data analysis (EDA) with visualizations
- Feature engineering for numeric and categorical variables
- Model training and comparison across several classifiers
- Hyperparameter tuning for tree-based ensembles
- Cost-sensitive and segment-specific decision rules based on expected profit

The final system outputs approval probabilities and provides decision thresholds that can be tuned to match different risk preferences.

---

## 2. Dataset

- Source: Kaggle **“Loan Approval Prediction”** (Playground Series).  
- Size: ~58k labeled training applications and ~39k unlabeled test applications.  
- Target: `loan_status` (1 = approved, 0 = rejected).  
- Features:
  - **Numeric:** age, annual income, loan amount, interest rate, employment length, credit history length, loan-to-income ratio.
  - **Categorical:** loan intent, loan grade, home ownership, prior default flag.

> Download the dataset from Kaggle and place the CSV files in a local `data/` directory.

---

## 3. Key Features

- **Full preprocessing pipeline**
  - Median and mode imputation
  - Standardization of numeric features
  - One-hot encoding of categorical features using `ColumnTransformer`.

- **Feature engineering**
  - Loan-to-income ratio and other derived variables to capture repayment capacity.

- **Model benchmarking and comparison**
  - Logistic Regression
  - Gaussian Naive Bayes
  - k-Nearest Neighbors
  - Support Vector Machine
  - Decision Tree
  - Random Forest
  - AdaBoost
  - Multilayer Perceptron
  - XGBoost

- **Ensembles**
  - Soft-voting ensemble over tuned Random Forest, tuned XGBoost, and Logistic Regression.
  - Stacked ensemble with a meta-learner on top of multiple base models.
  - Segment-specific “expert” models for different loan-to-income bands.

- **Business-aware evaluation**
  - Profit-based threshold selection that penalizes approving bad loans more than rejecting good ones.
  - Profit curves and profit surfaces for different thresholds and borrower segments.

---

## 4. Tech Stack

- **Language:** Python 3.x  
- **Core libraries:** 
  - `pandas`, `numpy`
  - `scikit-learn`
  - `xgboost`
  - `matplotlib`, `seaborn` (for plots)
  - `jupyter` / `notebook` for interactive analysis

---

## 5. Repository Structure

```text
.
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── loan_approval_prediction.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── profit_analysis.py
├── reports/
│   └── Final_Project_Report.pdf
├── requirements.txt
└── README.md
