"""
Failure Case Analysis and Business Error Audit Module.
Phase 5 Requirements:
- Deep diagnostic audit of False Positives (FP) and False Negatives (FN).
- Detailed investigation into Long-Tenure, High-Charge Loyal Customers who are misclassified as high risk.
- Feature profile contrasts between True Positives and False Positives.
- Financial impact calculation of classification errors on the retention budget.
"""

import os
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, RETENTION_COST, CHURN_LOSS


def audit_prediction_errors(
    df_clean: pd.DataFrame,
    y_true: np.ndarray,
    y_prob_oof: np.ndarray,
    opt_threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Categorizes each customer into:
    - TP (True Positive): Actual Churner correctly targeted for retention
    - TN (True Negative): Actual Loyal customer correctly identified as safe
    - FP (False Positive): Actual Loyal customer falsely flagged as risky (Budget Waste Risk)
    - FN (False Negative): Actual Churner missed by model (Unprevented Churn Loss)
    """
    df_audit = df_clean.copy()
    preds = (y_prob_oof >= opt_threshold).astype(int)
    
    df_audit["Predicted_Probability"] = y_prob_oof
    df_audit["Predicted_Binary"] = preds
    df_audit["Actual_Churn"] = y_true
    
    # Assign error categories
    conditions = [
        (df_audit["Actual_Churn"] == 1) & (df_audit["Predicted_Binary"] == 1),
        (df_audit["Actual_Churn"] == 0) & (df_audit["Predicted_Binary"] == 0),
        (df_audit["Actual_Churn"] == 0) & (df_audit["Predicted_Binary"] == 1),
        (df_audit["Actual_Churn"] == 1) & (df_audit["Predicted_Binary"] == 0),
    ]
    choices = ["True Positive (TP)", "True Negative (TN)", "False Positive (FP)", "False Negative (FN)"]
    df_audit["Error_Category"] = np.select(conditions, choices, default="Unknown")

    # Segment specific failure subsets
    fp_df = df_audit[df_audit["Error_Category"] == "False Positive (FP)"].copy()
    fn_df = df_audit[df_audit["Error_Category"] == "False Negative (FN)"].copy()

    # Summary table
    error_summary = df_audit.groupby("Error_Category").agg(
        Count=("customerID", "count"),
        Mean_Tenure=("tenure", "mean"),
        Median_Tenure=("tenure", "median"),
        Mean_MonthlyCharges=("MonthlyCharges", "mean"),
        Mean_Predicted_Prob=("Predicted_Probability", "mean")
    ).reset_index()
    
    error_summary["Percentage"] = (error_summary["Count"] / len(df_audit) * 100).round(1)
    
    return df_audit, fp_df, fn_df, error_summary


def probe_long_tenure_high_charge_false_positives(
    fp_df: pd.DataFrame,
    tenure_cutoff: int = 36,
    charge_cutoff: float = 70.0
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Investigates the high-value false positive cohort:
    Customers with tenure > 36 months and MonthlyCharges > $70 who stayed loyal
    despite model predicting high churn risk.
    """
    cohort = fp_df[(fp_df["tenure"] >= tenure_cutoff) & (fp_df["MonthlyCharges"] >= charge_cutoff)].copy()
    
    cohort_stats = {
        "Cohort_Count": len(cohort),
        "Mean_Tenure": cohort["tenure"].mean() if len(cohort) > 0 else 0,
        "Mean_MonthlyCharges": cohort["MonthlyCharges"].mean() if len(cohort) > 0 else 0,
        "Mean_Predicted_Risk": cohort["Predicted_Probability"].mean() if len(cohort) > 0 else 0,
        "Fiber_Optic_Pct": (cohort["InternetService"] == "Fiber optic").mean() * 100 if len(cohort) > 0 else 0,
        "Month_to_Month_Pct": (cohort["Contract"] == "Month-to-month").mean() * 100 if len(cohort) > 0 else 0,
        "Multiple_Lines_Pct": (cohort["MultipleLines"] == "Yes").mean() * 100 if len(cohort) > 0 else 0,
        "Wasted_Retention_Budget": len(cohort) * RETENTION_COST
    }
    
    return cohort, cohort_stats


def plot_failure_analysis_diagnostics(
    df_audit: pd.DataFrame,
    save_path: str | None = os.path.join(FIGURES_DIR, "10_failure_analysis_fp_fn_breakdown.png"),
    show: bool = False
) -> None:
    """
    Generates a 3-panel Failure Analysis diagnostic figure:
    1. Tenure vs MonthlyCharges scatterplot highlighting FP & FN clusters.
    2. False Positive vs True Positive contract & service distributions.
    3. Distribution of predicted probabilities across all 4 confusion categories.
    """
    fig = plt.figure(figsize=(18, 6), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])

    # Panel 1: Scatter of Tenure vs MonthlyCharges
    ax1 = fig.add_subplot(gs[0])
    palette = {
        "True Negative (TN)": "#95a5a6",
        "True Positive (TP)": "#2ecc71",
        "False Positive (FP)": "#e74c3c",
        "False Negative (FN)": "#f39c12"
    }
    sns.scatterplot(
        data=df_audit,
        x="tenure",
        y="MonthlyCharges",
        hue="Error_Category",
        palette=palette,
        alpha=0.6,
        s=25,
        ax=ax1
    )
    # Highlight the Long-tenure, high-charge FP quadrant
    ax1.axvline(36, color="red", linestyle=":", lw=1.5, alpha=0.7)
    ax1.axhline(70, color="red", linestyle=":", lw=1.5, alpha=0.7)
    ax1.text(38, 115, "Long-Tenure High-Charge\nFalse Positive Cluster", color="#c0392b", fontsize=9, fontweight="bold")
    ax1.set_title("Prediction Outcomes: Tenure vs. Monthly Charges", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Tenure (Months)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Monthly Charges ($)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=8, frameon=True, facecolor="white")

    # Panel 2: Contract type contrast between True Positives vs False Positives
    ax2 = fig.add_subplot(gs[1])
    tp_fp_df = df_audit[df_audit["Error_Category"].isin(["True Positive (TP)", "False Positive (FP)"])].copy()
    contract_rates = tp_fp_df.groupby(["Error_Category", "Contract"], observed=False).size().unstack(fill_value=0)
    contract_rates_pct = contract_rates.div(contract_rates.sum(axis=1), axis=0) * 100
    
    contract_rates_pct.plot(kind="bar", stacked=True, colormap="tab20c", ax=ax2, edgecolor="black", alpha=0.9)
    ax2.set_title("Contract Mix: True Positives vs. False Positives", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Prediction Class", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Proportion (%)", fontsize=11, fontweight="bold")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.legend(title="Contract Term", fontsize=8, loc="upper right")

    # Panel 3: Probability Distribution by Error Category
    ax3 = fig.add_subplot(gs[2])
    sns.boxplot(
        data=df_audit,
        x="Error_Category",
        y="Predicted_Probability",
        palette=palette,
        hue="Error_Category",
        legend=False,
        ax=ax3,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
    )
    ax3.set_title("Predicted Risk Score by Confusion Outcome", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Error Category", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Predicted Churn Probability", fontsize=11, fontweight="bold")
    ax3.grid(axis="y", linestyle=":", alpha=0.5)

    plt.suptitle("Phase 5 Diagnostic Failure Analysis: Deconstructing Model Errors", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
