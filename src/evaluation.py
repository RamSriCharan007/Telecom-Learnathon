"""
Threshold Tuning, Operational Decision Mapping, and Probability Calibration Module.
Phase 3 & Phase 5 Requirements:
- Out-of-fold threshold optimization for F1-Score and Business Cost Matrix.
- Precise mapping of probabilities into permitted operational tiers:
  LOW_RISK, WATCH, OFFER_RETENTION.
- Probability Calibration curves, Brier Score, and Expected Calibration Error (ECE).
- High-quality matplotlib/seaborn publication-grade visualizations.
"""

import os
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.calibration import calibration_curve

from src.config import (
    FIGURES_DIR, DEFAULT_TIER_LOW, DEFAULT_TIER_HIGH,
    RETENTION_COST, CHURN_LOSS, RETENTION_SUCCESS_RATE
)


def tune_decision_threshold(
    y_true: np.ndarray,
    y_prob_oof: np.ndarray,
    thresholds: np.ndarray = np.linspace(0.01, 0.99, 197),
    retention_cost: float = RETENTION_COST,
    churn_loss: float = CHURN_LOSS,
    success_rate: float = RETENTION_SUCCESS_RATE
) -> dict[str, Any]:
    """
    Sweeps probability thresholds strictly over Out-Of-Fold (OOF) validation predictions
    to find the optimal decision boundary for:
    1. Maximum F1-Score
    2. Minimum Business Net Financial Loss / Maximum Retained Value.
    """
    records = []
    
    for t in thresholds:
        preds = (y_prob_oof >= t).astype(int)
        
        # Confusion matrix elements
        tn = np.sum((y_true == 0) & (preds == 0))
        fp = np.sum((y_true == 0) & (preds == 1))
        fn = np.sum((y_true == 1) & (preds == 0))
        tp = np.sum((y_true == 1) & (preds == 1))
        
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        
        # Business cost model:
        # Cost of retention offers sent: (TP + FP) * retention_cost
        # Churn losses prevented: TP * success_rate * churn_loss
        # Churn losses incurred (uncontacted FN + unprevented TP): FN * churn_loss + TP * (1 - success_rate) * churn_loss
        total_retention_spend = (tp + fp) * retention_cost
        gross_churn_loss = (fn + tp * (1 - success_rate)) * churn_loss
        total_business_cost = total_retention_spend + gross_churn_loss
        
        # Net savings compared to doing nothing (Do nothing cost = Total Actual Churners * churn_loss)
        do_nothing_cost = np.sum(y_true == 1) * churn_loss
        net_business_savings = do_nothing_cost - total_business_cost
        
        records.append({
            "threshold": t,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "total_cost": total_business_cost,
            "net_savings": net_business_savings
        })
        
    df_curve = pd.DataFrame(records)
    
    # Best F1 threshold
    best_f1_idx = df_curve["f1_score"].idxmax()
    best_f1_row = df_curve.iloc[best_f1_idx]
    
    # Best Cost/Savings threshold
    best_cost_idx = df_curve["net_savings"].idxmax()
    best_cost_row = df_curve.iloc[best_cost_idx]
    
    return {
        "curve_df": df_curve,
        "best_f1_threshold": float(best_f1_row["threshold"]),
        "best_f1_score": float(best_f1_row["f1_score"]),
        "best_f1_precision": float(best_f1_row["precision"]),
        "best_f1_recall": float(best_f1_row["recall"]),
        "best_cost_threshold": float(best_cost_row["threshold"]),
        "best_cost_net_savings": float(best_cost_row["net_savings"]),
        "best_cost_f1": float(best_cost_row["f1_score"])
    }


def map_operational_tiers(
    y_prob: np.ndarray,
    t_low: float = DEFAULT_TIER_LOW,
    t_high: float = DEFAULT_TIER_HIGH
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Maps continuous churn probabilities into exactly 3 operational tiers:
    - LOW_RISK: p < t_low
    - WATCH: t_low <= p < t_high
    - OFFER_RETENTION: p >= t_high
    """
    tiers = np.empty(len(y_prob), dtype=object)
    
    low_mask = y_prob < t_low
    watch_mask = (y_prob >= t_low) & (y_prob < t_high)
    retention_mask = y_prob >= t_high
    
    tiers[low_mask] = "LOW_RISK"
    tiers[watch_mask] = "WATCH"
    tiers[retention_mask] = "OFFER_RETENTION"
    
    tier_summary = pd.DataFrame({
        "Tier": ["LOW_RISK", "WATCH", "OFFER_RETENTION"],
        "Probability_Range": [f"[0.00, {t_low:.2f})", f"[{t_low:.2f}, {t_high:.2f})", f"[{t_high:.2f}, 1.00]"],
        "Customer_Count": [low_mask.sum(), watch_mask.sum(), retention_mask.sum()],
        "Percentage": [f"{low_mask.mean()*100:.1f}%", f"{watch_mask.mean()*100:.1f}%", f"{retention_mask.mean()*100:.1f}%"],
        "Operational_Action": [
            "Standard Service & Regular Invoicing; No invasive outreach.",
            "Proactive engagement, CSAT check, newsletter, feature adoption tips.",
            "High-priority Retention Specialist call; Contract renewal discounts / upgrade voucher."
        ]
    })
    
    return tiers, tier_summary


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculates Expected Calibration Error (ECE) across uniform probability bins.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        bin_size = in_bin.sum()
        
        if bin_size > 0:
            bin_acc = y_true[in_bin].mean()
            bin_conf = y_prob[in_bin].mean()
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)
            
    return ece


# -------------------------------------------------------------
# Visualization Functions (Saved as high-res PNGs)
# -------------------------------------------------------------

def plot_roc_pr_comparison(
    results_list: list[dict[str, Any]],
    y_true: np.ndarray,
    save_path: str | None = os.path.join(FIGURES_DIR, "02_baseline_vs_challenger_roc_pr.png"),
    show: bool = False
) -> None:
    """
    Generates side-by-side ROC and Precision-Recall curves comparing Baseline vs Challenger.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    # Subplot 1: ROC Curves
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC = 0.5000)")
    for idx, res in enumerate(results_list):
        fpr, tpr, _ = roc_curve(y_true, res["oof_probs"])
        auc_val = roc_auc_score(y_true, res["oof_probs"])
        ax1.plot(fpr, tpr, color=palette[idx % len(palette)], lw=2.2,
                 label=f"{res['model_name']} (AUC = {auc_val:.4f})")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    ax1.set_title("Out-Of-Fold ROC Curve Comparison", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Precision-Recall Curves
    ax2 = axes[1]
    baseline_pr = y_true.mean()
    ax2.axhline(y=baseline_pr, color="gray", linestyle="--", label=f"No-Skill Baseline ({baseline_pr:.3f})")
    for idx, res in enumerate(results_list):
        prec, rec, _ = precision_recall_curve(y_true, res["oof_probs"])
        pr_auc = average_precision_score(y_true, res["oof_probs"])
        ax2.plot(rec, prec, color=palette[idx % len(palette)], lw=2.2,
                 label=f"{res['model_name']} (PR-AUC = {pr_auc:.4f})")
    ax2.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax2.set_title("Out-Of-Fold Precision-Recall Curve Comparison", fontsize=13, fontweight="bold", pad=12)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_threshold_tuning_curves(
    tuning_dict: dict[str, Any],
    save_path: str | None = os.path.join(FIGURES_DIR, "03_threshold_tuning_f1_curve.png"),
    show: bool = False
) -> None:
    """
    Visualizes F1, Precision, Recall, and Net Business Savings across all thresholds.
    """
    df = tuning_dict["curve_df"]
    best_f1_t = tuning_dict["best_f1_threshold"]
    best_f1 = tuning_dict["best_f1_score"]
    best_cost_t = tuning_dict["best_cost_threshold"]

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # Left Axis: Metric Curves
    ax1.plot(df["threshold"], df["f1_score"], color="#1f77b4", lw=2.5, label="F1-Score")
    ax1.plot(df["threshold"], df["precision"], color="#2ca02c", lw=2.0, linestyle="-.", label="Precision")
    ax1.plot(df["threshold"], df["recall"], color="#ff7f0e", lw=2.0, linestyle="--", label="Recall")
    
    # Highlight optimal threshold
    ax1.axvline(x=best_f1_t, color="#d62728", linestyle=":", lw=2.0,
                label=f"Max F1 Threshold ($t^*={best_f1_t:.2f}$, F1={best_f1:.3f})")
    ax1.axvline(x=0.50, color="gray", linestyle="--", alpha=0.7, label="Default Threshold ($t=0.50$)")

    ax1.set_xlabel("Decision Threshold (Probability Cutoff)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax1.set_title("Out-Of-Fold Threshold Optimization & Metric Tradeoffs", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlim(0.05, 0.95)
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Right Axis: Net Business Savings ($)
    ax2 = ax1.twinx()
    ax2.plot(df["threshold"], df["net_savings"] / 1000.0, color="#9467bd", lw=2.0, linestyle=":", label="Net Business Savings ($k)")
    ax2.set_ylabel("Net Churn Savings ($ in Thousands)", fontsize=11, fontweight="bold", color="#9467bd")
    ax2.tick_params(axis="y", labelcolor="#9467bd")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=True, facecolor="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrices(
    y_true: np.ndarray,
    baseline_oof: np.ndarray,
    challenger_oof: np.ndarray,
    opt_threshold: float,
    save_path: str | None = os.path.join(FIGURES_DIR, "05_confusion_matrices_comparison.png"),
    show: bool = False
) -> None:
    """
    Plots a 2x2 grid of confusion matrices:
    - Baseline at t=0.50 vs Baseline at Tuned t*
    - Challenger at t=0.50 vs Challenger at Tuned t*
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    
    configs = [
        ("Baseline (Logistic Reg) [t=0.50]", (baseline_oof >= 0.50).astype(int), axes[0, 0]),
        (f"Baseline (Logistic Reg) [Tuned t*={opt_threshold:.2f}]", (baseline_oof >= opt_threshold).astype(int), axes[0, 1]),
        ("Challenger (XGBoost) [t=0.50]", (challenger_oof >= 0.50).astype(int), axes[1, 0]),
        (f"Challenger (XGBoost) [Tuned t*={opt_threshold:.2f}]", (challenger_oof >= opt_threshold).astype(int), axes[1, 1])
    ]

    for title, preds, ax in configs:
        cm = confusion_matrix(y_true, preds)
        rec = recall_score(y_true, preds)
        prec = precision_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"size": 13, "fontweight": "bold"})
        ax.set_title(f"{title}\nF1: {f1:.3f} | Recall: {rec:.3f} | Prec: {prec:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label (0 = Retained, 1 = Churned)", fontsize=10)
        ax.set_ylabel("True Label (0 = Retained, 1 = Churned)", fontsize=10)
        ax.set_xticklabels(["Retained", "Churned"])
        ax.set_yticklabels(["Retained", "Churned"])

    plt.suptitle("Out-Of-Fold Confusion Matrix Comparison Across Models & Thresholds", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_operational_tiers_distribution(
    df_clean: pd.DataFrame,
    tiers: np.ndarray,
    save_path: str | None = os.path.join(FIGURES_DIR, "04_operational_risk_tiers_distribution.png"),
    show: bool = False
) -> None:
    """
    Visualizes distribution of operational risk tiers along with actual churn rate within each tier.
    """
    df_plot = df_clean.copy()
    df_plot["Risk_Tier"] = pd.Categorical(tiers, categories=["LOW_RISK", "WATCH", "OFFER_RETENTION"], ordered=True)
    
    tier_stats = df_plot.groupby("Risk_Tier", observed=False).agg(
        Total_Count=("Churn", "count"),
        Actual_Churn_Rate=("Churn", "mean")
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    tier_colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    # Bar 1: Customer Volume per Tier
    ax1 = axes[0]
    bars1 = ax1.bar(tier_stats["Risk_Tier"], tier_stats["Total_Count"], color=tier_colors, edgecolor="black", alpha=0.85)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{int(yval):,} ({yval/len(df_plot)*100:.1f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_title("Customer Volume Allocation per Risk Tier", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Customers", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(tier_stats["Total_Count"]) * 1.15)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Bar 2: Actual Churn Rate per Tier
    ax2 = axes[1]
    bars2 = ax2.bar(tier_stats["Risk_Tier"], tier_stats["Actual_Churn_Rate"] * 100, color=tier_colors, edgecolor="black", alpha=0.85)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_title("Actual Empirical Churn Rate by Risk Tier", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Actual Churn Rate (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("Operational Tier Decision Strategy: Volume vs. Empirical Risk", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_calibration_curves(
    results_list: list[dict[str, Any]],
    y_true: np.ndarray,
    n_bins: int = 10,
    save_path: str | None = os.path.join(FIGURES_DIR, "09_probability_calibration_curve.png"),
    show: bool = False
) -> None:
    """
    Phase 5: Generates reliability / probability calibration curves and histograms.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), dpi=300, gridspec_kw={"height_ratios": [3, 1]})
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    # Perfect calibration reference line
    ax1.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")

    for idx, res in enumerate(results_list):
        prob_true, prob_pred = calibration_curve(y_true, res["oof_probs"], n_bins=n_bins, strategy="uniform")
        ece = compute_expected_calibration_error(y_true, res["oof_probs"], n_bins=n_bins)
        brier = np.mean((res["oof_probs"] - y_true) ** 2)
        
        ax1.plot(prob_pred, prob_true, "s-", color=palette[idx % len(palette)], lw=2.2,
                 label=f"{res['model_name']} (Brier={brier:.4f}, ECE={ece:.4f})")
        
        # Bottom histogram of predicted probabilities
        ax2.hist(res["oof_probs"], range=(0, 1), bins=n_bins, histtype="step", lw=2,
                 color=palette[idx % len(palette)], label=res["model_name"])

    ax1.set_ylabel("Fraction of Positives (Empirical Churn Rate)", fontsize=11, fontweight="bold")
    ax1.set_title("Reliability Calibration Curve (10 Bins)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, facecolor="white")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_ylim(-0.02, 1.02)

    ax2.set_xlabel("Mean Predicted Risk Probability", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
