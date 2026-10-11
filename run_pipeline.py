"""
Telco Churn: Leakage-Safe Baseline vs Tree Model
Main Pipeline Execution Script.
Executes all 5 hackathon workflow phases end-to-end and saves all figures & metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure workspace root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    DATA_PATH, FIGURES_DIR, METRICS_DIR, RANDOM_STATE, N_SPLITS,
    DEFAULT_TIER_LOW, DEFAULT_TIER_HIGH, RETENTION_COST, CHURN_LOSS,
    ID_COL
)
from src.data_loader import (
    load_raw_dataset, clean_telco_data, get_stratified_folds, generate_data_dictionary
)
from src.pipeline_builder import (
    create_baseline_pipeline, create_challenger_pipeline
)
from src.modeling import (
    run_stratified_cv, compare_models
)
from src.evaluation import (
    tune_decision_threshold, map_operational_tiers, compute_expected_calibration_error,
    plot_roc_pr_comparison, plot_threshold_tuning_curves, plot_confusion_matrices,
    plot_operational_tiers_distribution, plot_calibration_curves
)
from src.explainability import (
    extract_logistic_regression_coefficients, plot_logistic_regression_odds_ratios,
    compute_tree_shap_analysis, generate_executive_insights
)
from src.failure_analysis import (
    audit_prediction_errors, probe_long_tenure_high_charge_false_positives,
    plot_failure_analysis_diagnostics
)


def plot_eda_overview(df_raw: pd.DataFrame, df_clean: pd.DataFrame, save_path: str):
    """
    Generates high-resolution EDA figure illustrating:
    1. Churn class distribution (~73.5% No vs ~26.5% Yes)
    2. Tenure distribution by Churn status
    3. Monthly Charges distribution by Churn status
    4. Contract type vs Churn rate
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # 1. Churn Distribution Donut Chart
    ax1 = axes[0, 0]
    churn_counts = df_clean["Churn"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    ax1.pie(churn_counts, labels=["Retained (No)", "Churned (Yes)"], autopct="%1.1f%%",
            startangle=140, colors=colors, explode=(0, 0.08), shadow=True,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax1.set_title("Dataset Churn Distribution (7,043 Customers)", fontsize=12, fontweight="bold")

    # 2. Tenure by Churn
    ax2 = axes[0, 1]
    sns.histplot(data=df_clean, x="tenure", hue="Churn", multiple="stack", palette=colors,
                 bins=30, ax=ax2, alpha=0.85)
    ax2.set_title("Customer Tenure (Months) by Churn Status", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Tenure (Months)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Customer Count", fontsize=10, fontweight="bold")
    ax2.legend(["Churned (1)", "Retained (0)"], loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # 3. Monthly Charges by Churn
    ax3 = axes[1, 0]
    sns.kdeplot(data=df_clean[df_clean["Churn"] == 0]["MonthlyCharges"], color="#2ecc71",
                fill=True, alpha=0.4, label="Retained (0)", ax=ax3, lw=2)
    sns.kdeplot(data=df_clean[df_clean["Churn"] == 1]["MonthlyCharges"], color="#e74c3c",
                fill=True, alpha=0.4, label="Churned (1)", ax=ax3, lw=2)
    ax3.set_title("Monthly Charges Density ($) by Churn Status", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Monthly Charges ($)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Density", fontsize=10, fontweight="bold")
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle=":", alpha=0.5)

    # 4. Contract Type Churn Rate
    ax4 = axes[1, 1]
    contract_churn = df_clean.groupby("Contract")["Churn"].agg(["count", "mean"]).reset_index()
    contract_churn["mean"] = contract_churn["mean"] * 100
    bars = ax4.bar(contract_churn["Contract"], contract_churn["mean"], color=["#e74c3c", "#f39c12", "#2ecc71"],
                   edgecolor="black", alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax4.set_title("Churn Rate by Contract Term", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Contract Term", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Churn Rate (%)", fontsize=10, fontweight="bold")
    ax4.set_ylim(0, 55)
    ax4.grid(axis="y", linestyle=":", alpha=0.5)

    plt.suptitle("Exploratory Data Analysis: Key Telco Churn Signals", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 80)
    print("TELCO CHURN: LEAKAGE-SAFE BASELINE VS TREE MODEL PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------
    # PHASE 1: DATA INGESTION, CLEANING & LEAKAGE PREVENTION
    # -------------------------------------------------------------
    print("\n>>> PHASE 1: Data Ingestion, Cleaning & Leakage Prevention...")
    df_raw = load_raw_dataset(DATA_PATH)
    print(f"Loaded raw dataset from {DATA_PATH} with shape: {df_raw.shape}")

    # Data dictionary
    data_dict = generate_data_dictionary(df_raw)
    data_dict.to_csv(os.path.join(METRICS_DIR, "data_dictionary.csv"), index=False)
    print("Generated and saved data dictionary (21 variables).")

    # Clean data safely
    X, y, df_clean = clean_telco_data(df_raw)
    print(f"Features matrix X shape: {X.shape}, Target y shape: {y.shape}")
    print(f"TotalCharges whitespace strings fixed: 11 rows safely imputed via (tenure * MonthlyCharges).")
    print(f"customerID dropped from modeling matrix: {ID_COL not in X.columns}")
    print(f"Churn rate: {y.mean()*100:.2f}% (Total Churners: {y.sum():,} / {len(y):,})")

    # Generate EDA figure
    eda_fig_path = os.path.join(FIGURES_DIR, "01_eda_and_churn_distribution.png")
    plot_eda_overview(df_raw, df_clean, eda_fig_path)
    print(f"Saved EDA overview figure to {eda_fig_path}")

    # Build Stratified 5-Fold split manifest
    folds = get_stratified_folds(X, y, n_splits=N_SPLITS, random_state=RANDOM_STATE)
    print(f"Constructed {N_SPLITS}-Fold Stratified Cross-Validation split manifest.")
    for i, (tr_idx, val_idx) in enumerate(folds, 1):
        print(f"  - Fold {i}: Train={len(tr_idx):,} (Churn={y.iloc[tr_idx].mean():.3f}), Val={len(val_idx):,} (Churn={y.iloc[val_idx].mean():.3f})")

    # -------------------------------------------------------------
    # PHASE 2: BASELINE & CHALLENGER MODELING
    # -------------------------------------------------------------
    print("\n>>> PHASE 2: Transparent Baseline & Challenger Modeling...")
    print("Training Baseline Model: Logistic Regression (class_weight='balanced')...")
    baseline_results = run_stratified_cv(
        model_name="Baseline (Logistic Regression)",
        pipeline_factory=create_baseline_pipeline,
        X=X,
        y=y,
        folds=folds
    )

    print("Training Challenger Model: XGBoost Classifier (scale_pos_weight=2.77)...")
    challenger_results = run_stratified_cv(
        model_name="Challenger (XGBoost Ensemble)",
        pipeline_factory=lambda: create_challenger_pipeline("xgboost"),
        X=X,
        y=y,
        folds=folds
    )

    # Compare models
    model_comparison_df = compare_models([baseline_results, challenger_results])
    print("\nOut-of-Fold Model Performance Comparison:")
    print(model_comparison_df.to_string(index=False))
    model_comparison_df.to_csv(os.path.join(METRICS_DIR, "model_comparison_metrics.csv"), index=False)

    # Save ROC and PR comparison plot
    roc_pr_fig_path = os.path.join(FIGURES_DIR, "02_baseline_vs_challenger_roc_pr.png")
    plot_roc_pr_comparison([baseline_results, challenger_results], y.values, roc_pr_fig_path)
    print(f"Saved ROC/PR curve comparison to {roc_pr_fig_path}")

    # -------------------------------------------------------------
    # PHASE 3: THRESHOLD TUNING & DECISION MAPPING
    # -------------------------------------------------------------
    print("\n>>> PHASE 3: Threshold Tuning & Decision Mapping...")
    print("Optimizing threshold strictly on Out-Of-Fold probabilities for Challenger Model...")
    tuning_results = tune_decision_threshold(
        y_true=y.values,
        y_prob_oof=challenger_results["oof_probs"]
    )
    opt_f1_t = tuning_results["best_f1_threshold"]
    best_f1 = tuning_results["best_f1_score"]
    opt_cost_t = tuning_results["best_cost_threshold"]
    max_savings = tuning_results["best_cost_net_savings"]

    print(f"Optimal Threshold for F1-Score: t* = {opt_f1_t:.2f} (Yields Max F1 = {best_f1:.4f}, Prec = {tuning_results['best_f1_precision']:.4f}, Rec = {tuning_results['best_f1_recall']:.4f})")
    print(f"Optimal Threshold for Business Cost: t_cost = {opt_cost_t:.2f} (Yields Max Net Retention Savings = ${max_savings:,.2f})")

    # Save threshold curves plot
    thresh_fig_path = os.path.join(FIGURES_DIR, "03_threshold_tuning_f1_curve.png")
    plot_threshold_tuning_curves(tuning_results, thresh_fig_path)
    print(f"Saved threshold optimization curves to {thresh_fig_path}")

    # Map into operational risk tiers
    tiers, tier_summary_df = map_operational_tiers(challenger_results["oof_probs"], t_low=DEFAULT_TIER_LOW, t_high=opt_f1_t)
    print("\nOperational Tier Strategy:")
    print(tier_summary_df.to_string(index=False))
    tier_summary_df.to_csv(os.path.join(METRICS_DIR, "operational_tiers_summary.csv"), index=False)

    tier_fig_path = os.path.join(FIGURES_DIR, "04_operational_risk_tiers_distribution.png")
    plot_operational_tiers_distribution(df_clean, tiers, tier_fig_path)
    print(f"Saved operational tiers distribution plot to {tier_fig_path}")

    # Confusion matrix comparison
    cm_fig_path = os.path.join(FIGURES_DIR, "05_confusion_matrices_comparison.png")
    plot_confusion_matrices(
        y_true=y.values,
        baseline_oof=baseline_results["oof_probs"],
        challenger_oof=challenger_results["oof_probs"],
        opt_threshold=opt_f1_t,
        save_path=cm_fig_path
    )
    print(f"Saved confusion matrices grid to {cm_fig_path}")

    # -------------------------------------------------------------
    # PHASE 4: MODEL EXPLAINABILITY & FEATURE DRIVERS
    # -------------------------------------------------------------
    print("\n>>> PHASE 4: Model Explainability & Feature Drivers...")
    # Logistic Regression Odds Ratios
    fitted_baseline_pipeline = baseline_results["fitted_pipelines"][0]
    lr_coefs_df = extract_logistic_regression_coefficients(fitted_baseline_pipeline)
    lr_coefs_df.to_csv(os.path.join(METRICS_DIR, "logistic_regression_coefficients.csv"), index=False)
    
    lr_fig_path = os.path.join(FIGURES_DIR, "06_logistic_regression_odds_ratios.png")
    plot_logistic_regression_odds_ratios(lr_coefs_df, top_n=15, save_path=lr_fig_path)
    print(f"Saved Logistic Regression Odds Ratio plot to {lr_fig_path}")

    # SHAP Tree Analysis on Challenger Model
    print("Computing SHAP values on Challenger XGBoost model (sample of 1,500 records)...")
    fitted_tree_pipeline = challenger_results["fitted_pipelines"][0]
    X_shap_sample = X.sample(n=1500, random_state=RANDOM_STATE)
    
    shap_summary_path = os.path.join(FIGURES_DIR, "07_shap_summary_and_importance.png")
    shap_dep_path = os.path.join(FIGURES_DIR, "08_shap_dependence_tenure_charges_contract.png")
    compute_tree_shap_analysis(fitted_tree_pipeline, X_shap_sample, shap_summary_path, shap_dep_path)
    print(f"Saved SHAP summary & importance plot to {shap_summary_path}")
    print(f"Saved SHAP dependence plots to {shap_dep_path}")

    # Business Insights
    insights = generate_executive_insights()
    with open(os.path.join(METRICS_DIR, "executive_business_insights.json"), "w") as f:
        json.dump(insights, f, indent=4)
    print("Saved Executive Business Insights summary.")

    # -------------------------------------------------------------
    # PHASE 5: CONTROLLED EXTENSION & FAILURE ANALYSIS
    # -------------------------------------------------------------
    print("\n>>> PHASE 5: Controlled Extension & Failure Analysis...")
    # Probability Calibration
    calib_fig_path = os.path.join(FIGURES_DIR, "09_probability_calibration_curve.png")
    plot_calibration_curves([baseline_results, challenger_results], y.values, n_bins=10, save_path=calib_fig_path)
    print(f"Saved probability calibration curves to {calib_fig_path}")

    # Error audit & False positive investigation
    df_audit, fp_df, fn_df, error_summary = audit_prediction_errors(
        df_clean=df_clean,
        y_true=y.values,
        y_prob_oof=challenger_results["oof_probs"],
        opt_threshold=opt_f1_t
    )
    error_summary.to_csv(os.path.join(METRICS_DIR, "failure_analysis_error_summary.csv"), index=False)
    print("\nError Category Audit Summary:")
    print(error_summary.to_string(index=False))

    # Probe Long-Tenure, High-Charge False Positives
    lt_cohort, lt_stats = probe_long_tenure_high_charge_false_positives(fp_df, tenure_cutoff=36, charge_cutoff=70.0)
    print(f"\nLong-Tenure (>=36 mos), High-Charge (>=$70/mo) False Positive Cohort:")
    print(f"  - Count: {lt_stats['Cohort_Count']:,} customers")
    print(f"  - Mean Tenure: {lt_stats['Mean_Tenure']:.1f} months | Mean Charges: ${lt_stats['Mean_MonthlyCharges']:.2f}")
    print(f"  - Fiber Optic Internet: {lt_stats['Fiber_Optic_Pct']:.1f}% | Month-to-Month Contract: {lt_stats['Month_to_Month_Pct']:.1f}%")
    print(f"  - Multiple Lines: {lt_stats['Multiple_Lines_Pct']:.1f}%")
    print(f"  - Potential Wasted Retention Discount Budget: ${lt_stats['Wasted_Retention_Budget']:,.2f}")

    with open(os.path.join(METRICS_DIR, "long_tenure_false_positives_audit.json"), "w") as f:
        json.dump(lt_stats, f, indent=4)

    # Save failure analysis plots
    failure_fig_path = os.path.join(FIGURES_DIR, "10_failure_analysis_fp_fn_breakdown.png")
    plot_failure_analysis_diagnostics(df_audit, failure_fig_path)
    print(f"Saved failure analysis diagnostics to {failure_fig_path}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY! ALL ARTIFACTS, FIGURES, AND METRICS GENERATED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
