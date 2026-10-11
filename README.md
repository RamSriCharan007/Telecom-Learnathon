# 📊 Telco Churn: Leakage-Safe Baseline vs Tree Model
### *University Machine Learning Hackathon Submission — Production-Grade Predictive Pipeline*

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.53.0-green.svg)](https://shap.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Complete%20%26%20Leakage--Free-brightgreen.svg)]()

---

## 🎯 Executive Summary & Overview
Customer attrition (*churn*) is a multi-million-dollar vulnerability in the telecommunications sector. Replacing an existing subscriber incurs 5–7x higher customer acquisition costs (CAC) compared to maintaining an active account.

This repository delivers an **end-to-end, 100% data-leakage-free predictive pipeline** and an **executable Jupyter notebook** on the Telco Customer Churn dataset (7,043 customers). The solution is built with strict academic and production integrity across 5 core phases:

1. **Phase 1: Data Ingestion, Cleaning & Leakage Prevention (R1 & R2)**
2. **Phase 2: Transparent Baseline & Challenger Modeling (R3)**
3. **Phase 3: Threshold Tuning & Decision Mapping (R4)**
4. **Phase 4: Model Explainability & Feature Drivers (R4)**
5. **Phase 5: Controlled Extension & Failure Analysis (R5)**

---

## 📁 Repository Structure

```tree
telco-churn-hackathon/
│
├── telco_churn_pipeline.ipynb             # Master fully executed Jupyter Notebook with all outputs & plots
├── run_pipeline.py                       # CLI script to execute the end-to-end pipeline
├── create_notebook.py                    # Script used to compile and pre-execute the notebook
├── README.md                             # Comprehensive project documentation
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # 7,043 records Kaggle telco churn dataset
│
├── src/                                  # Modular Python source code
│   ├── __init__.py
│   ├── config.py                         # Global hyperparams, column mappings, and cost assumptions
│   ├── data_loader.py                    # Ingestion, cleaning, deterministic imputation, CV manifest
│   ├── pipeline_builder.py               # Leakage-safe ColumnTransformer and sklearn Pipelines
│   ├── modeling.py                       # Stratified 5-Fold CV runner with OOF probability logging
│   ├── evaluation.py                     # Threshold tuning, 3-tier risk mapping, calibration curves
│   ├── explainability.py                 # Logistic Odds Ratios, SHAP TreeExplainer, executive insights
│   └── failure_analysis.py               # Diagnostic audit of FP/FN, long-tenure high-charge breakdown
│
└── reports/
    ├── figures/                          # 10 High-resolution (300 DPI) publication figures
    │   ├── 01_eda_and_churn_distribution.png
    │   ├── 02_baseline_vs_challenger_roc_pr.png
    │   ├── 03_threshold_tuning_f1_curve.png
    │   ├── 04_operational_risk_tiers_distribution.png
    │   ├── 05_confusion_matrices_comparison.png
    │   ├── 06_logistic_regression_odds_ratios.png
    │   ├── 07_shap_summary_and_importance.png
    │   ├── 08_shap_dependence_tenure_charges_contract.png
    │   ├── 09_probability_calibration_curve.png
    │   └── 10_failure_analysis_fp_fn_breakdown.png
    └── metrics/                          # Quantitative CSV and JSON artifacts
        ├── data_dictionary.csv
        ├── model_comparison_metrics.csv
        ├── operational_tiers_summary.csv
        ├── logistic_regression_coefficients.csv
        ├── failure_analysis_error_summary.csv
        ├── long_tenure_false_positives_audit.json
        └── executive_business_insights.json
```

---

## ⚙️ Phase-by-Phase Technical Methodology

### Phase 1: Ingestion, Cleaning & Leakage Prevention (R1 & R2)
- **ID Dropping:** Dropped `customerID` immediately upon ingestion to eliminate surrogate key memorization.
- **Deterministic Imputation for `TotalCharges`:** 11 records contained whitespace strings (`" "`). Inspection confirmed all 11 customers had `tenure = 0` (brand-new accounts). Imputing via `TotalCharges = tenure * MonthlyCharges` assigned exactly `$0.00`, preserving strict physical billing reality without leaking sample statistics.
- **CV Split Manifest:** Built a 5-Fold `StratifiedKFold` manifest maintaining the exact **26.54%** churn balance across all folds.
- **Pipeline Encapsulation:** `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` are packaged in an `sklearn.compose.ColumnTransformer` inside an `sklearn.pipeline.Pipeline`. Preprocessing fit parameters are derived exclusively on training folds.

### Phase 2: Baseline vs. Challenger Modeling (R3)
Both models were evaluated strictly **Out-Of-Fold (OOF)** on held-out validation folds:

| Model Architecture | OOF ROC-AUC | OOF PR-AUC | OOF F1-Score ($t=0.50$) | OOF Recall ($t=0.50$) | OOF Precision ($t=0.50$) | OOF Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logistic Regression Balanced)** | 0.8449 (±0.0151) | 0.6532 (±0.0305) | 0.6251 (±0.0103) | **0.8010** (±0.0180) | 0.5125 (±0.0076) | 0.1660 |
| **Challenger (XGBoost Ensemble)** | **0.8452** (±0.0120) | **0.6606** (±0.0263) | **0.6313** (±0.0167) | 0.7854 (±0.0168) | **0.5277** (±0.0176) | **0.1598** |

### Phase 3: Threshold Tuning & Operational Tier Mapping (R4)
- **Threshold Optimization:** Swept $t \in [0.01, 0.99]$ on OOF probabilities:
  - **Max F1 Threshold:** $t^* = 0.54$ yields $\text{F1} = 0.6329$, $\text{Precision} = 0.5441$, $\text{Recall} = 0.7566$.
  - **Optimal Business Cost Threshold:** $t_{cost} = 0.54$ minimizes net churn loss and yields **$82,150.00** in net retained value over baseline.
- **Permitted Operational Tiers:**

| Tier | Probability Range | Volume (% Total) | Operational Action Playbook |
| :--- | :---: | :---: | :--- |
| 🟢 **`LOW_RISK`** | $[0.00, 0.30)$ | 3,260 (46.3%) | Standard service & automated invoicing; zero retention spending. |
| 🟡 **`WATCH`** | $[0.30, 0.54)$ | 1,184 (16.8%) | Non-invasive engagement: automated check-in email, CSAT survey, feature guides. |
| 🔴 **`OFFER_RETENTION`** | $[0.54, 1.00]$ | 2,599 (36.9%) | High-priority outreach: Retention Specialist outbound call + $50 renewal incentive. |

### Phase 4: Model Explainability & Feature Drivers (R4)
- **Logistic Regression Odds Ratios:**
  - `Contract_Month-to-month`: **+3.24x** higher odds of churn compared to 1/2-year contracts.
  - `tenure`: **0.38x** protective factor per standard deviation increase.
  - `PaymentMethod_Electronic check`: **+1.35x** increased churn odds due to manual payment friction.
- **SHAP Tree Analysis (XGBoost):**
  - **Tenure Churn Cliff:** Churn hazard is intensely concentrated in months 0–12. Beyond month 24, churn probability flattens due to ecosystem inertia.
  - **Monthly Charges Acceleration:** Churn risk accelerates above **$65–$70/month**, driven by standalone Fiber Optic lines lacking bundled security/cloud support.

### Phase 5: Controlled Extension & Failure Analysis (R5)
- **Probability Calibration:** XGBoost achieves a superior Brier score of **0.1598** and low Expected Calibration Error (ECE = 0.054), showing strong reliability.
- **Failure Case Audit:**
  - **True Positives (TP):** 1,414 (20.1%) — High risk, churned.
  - **True Negatives (TN):** 3,989 (56.6%) — Low risk, retained.
  - **False Positives (FP):** 1,185 (16.8%) — High risk, retained.
  - **False Negatives (FN):** 455 (6.5%) — Low risk, churned.
- **Long-Tenure High-Charge False Positive Archetype:**
  - Identified **226 customers** with $\text{tenure} \ge 36\text{ months}$ and $\text{MonthlyCharges} \ge \$70.00$ who stayed loyal despite high model risk.
  - Profile: 99.6% have Fiber Optic and 85.8% have Multiple Lines (high stickiness).
  - Offering unnecessary discounts to this group wastes **$11,300.00** in retention budget.
  - **Recommended Business Rule:** Downgrade long-tenure multi-line customers from `OFFER_RETENTION` to `WATCH` prior to campaign dispatch.

---

## 🚀 Quickstart & Execution

### 1. Set Up Environment
```bash
# Clone or navigate to the repository
cd d:/l

# Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\pip install pandas scikit-learn matplotlib seaborn shap xgboost ipykernel nbformat nbclient
```

### 2. Execute End-to-End Pipeline
```bash
.venv\Scripts\python run_pipeline.py
```

### 3. Open and Inspect Master Jupyter Notebook
```bash
# Open telco_churn_pipeline.ipynb in Jupyter or VS Code / IDE
```

---

## 🏆 Key Conclusions for Hackathon Judges
1. **Methodological Rigor:** Complete absence of data leakage via deterministic TotalCharges imputation and cross-validated ColumnTransformer pipelines.
2. **Business Alignment:** Threshold optimization directly ties ML probabilities to financial outcomes ($82.1k net savings).
3. **Honest Governance:** Deep failure analysis detects and solves the $11.3k budget waste problem on loyal multi-line subscribers.
