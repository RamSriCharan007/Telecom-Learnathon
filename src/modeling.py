"""
Cross-Validation Runner and Model Training Module.
Phase 2 Requirements:
- Train Baseline (Logistic Regression balanced) and Challenger (Tree Ensemble) across exact same Stratified 5-Fold splits.
- Evaluate both models strictly out-of-fold (OOF) on ROC-AUC, F1-Score, Recall, Precision, and PR-AUC.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from typing import Callable, Any


def run_stratified_cv(
    model_name: str,
    pipeline_factory: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    default_threshold: float = 0.50
) -> dict[str, Any]:
    """
    Executes a leakage-safe Stratified K-Fold cross-validation loop.
    
    Args:
        model_name: Human-readable model identifier.
        pipeline_factory: Zero-argument callable returning a fresh sklearn Pipeline instance.
        X: Feature matrix.
        y: Target series.
        folds: List of (train_idx, val_idx) tuples.
        default_threshold: Probability decision threshold for binary classification metrics.

    Returns:
        Dictionary containing:
        - "model_name": Name of the model
        - "oof_probs": 1D array of out-of-fold predicted probabilities for all samples
        - "oof_preds": 1D array of binary predictions at default threshold
        - "fold_metrics": pd.DataFrame of metrics per fold
        - "summary_metrics": pd.Series of mean and std across folds
        - "fitted_pipelines": list of fitted Pipeline instances (one per fold)
    """
    n_samples = len(X)
    oof_probs = np.zeros(n_samples, dtype=float)
    fitted_pipelines = []
    fold_records = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Instantiate a fresh pipeline to ensure complete isolation
        pipeline = pipeline_factory()
        
        # Fit strictly on train fold
        pipeline.fit(X_train, y_train)
        fitted_pipelines.append(pipeline)

        # Predict probability for positive class (Churn = 1) on validation fold
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs
        val_preds = (val_probs >= default_threshold).astype(int)

        # Compute fold-level metrics
        fold_roc_auc = roc_auc_score(y_val, val_probs)
        fold_pr_auc = average_precision_score(y_val, val_probs)
        fold_f1 = f1_score(y_val, val_preds, zero_division=0)
        fold_recall = recall_score(y_val, val_preds, zero_division=0)
        fold_precision = precision_score(y_val, val_preds, zero_division=0)
        fold_brier = brier_score_loss(y_val, val_probs)

        fold_records.append({
            "Fold": f"Fold {fold_idx}",
            "ROC_AUC": fold_roc_auc,
            "PR_AUC": fold_pr_auc,
            "F1_Score": fold_f1,
            "Recall": fold_recall,
            "Precision": fold_precision,
            "Brier_Score": fold_brier
        })

    fold_metrics_df = pd.DataFrame(fold_records)

    # Compute overall Out-Of-Fold metrics across full dataset
    oof_preds = (oof_probs >= default_threshold).astype(int)
    overall_roc_auc = roc_auc_score(y, oof_probs)
    overall_pr_auc = average_precision_score(y, oof_probs)
    overall_f1 = f1_score(y, oof_preds, zero_division=0)
    overall_recall = recall_score(y, oof_preds, zero_division=0)
    overall_precision = precision_score(y, oof_preds, zero_division=0)
    overall_brier = brier_score_loss(y, oof_probs)

    summary_metrics = {
        "ROC_AUC_Mean": fold_metrics_df["ROC_AUC"].mean(),
        "ROC_AUC_Std": fold_metrics_df["ROC_AUC"].std(),
        "PR_AUC_Mean": fold_metrics_df["PR_AUC"].mean(),
        "PR_AUC_Std": fold_metrics_df["PR_AUC"].std(),
        "F1_Score_Mean": fold_metrics_df["F1_Score"].mean(),
        "F1_Score_Std": fold_metrics_df["F1_Score"].std(),
        "Recall_Mean": fold_metrics_df["Recall"].mean(),
        "Recall_Std": fold_metrics_df["Recall"].std(),
        "Precision_Mean": fold_metrics_df["Precision"].mean(),
        "Precision_Std": fold_metrics_df["Precision"].std(),
        "Brier_Score_Mean": fold_metrics_df["Brier_Score"].mean(),
        "Overall_OOF_ROC_AUC": overall_roc_auc,
        "Overall_OOF_PR_AUC": overall_pr_auc,
        "Overall_OOF_F1": overall_f1,
        "Overall_OOF_Recall": overall_recall,
        "Overall_OOF_Precision": overall_precision,
        "Overall_OOF_Brier": overall_brier
    }

    return {
        "model_name": model_name,
        "oof_probs": oof_probs,
        "oof_preds": oof_preds,
        "fold_metrics": fold_metrics_df,
        "summary_metrics": pd.Series(summary_metrics),
        "fitted_pipelines": fitted_pipelines
    }


def compare_models(results_list: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Compiles a side-by-side comparison table of all evaluated models.
    """
    comparison_rows = []
    for res in results_list:
        summary = res["summary_metrics"]
        comparison_rows.append({
            "Model": res["model_name"],
            "OOF ROC-AUC": f"{summary['Overall_OOF_ROC_AUC']:.4f} (±{summary['ROC_AUC_Std']:.4f})",
            "OOF PR-AUC": f"{summary['Overall_OOF_PR_AUC']:.4f} (±{summary['PR_AUC_Std']:.4f})",
            "OOF F1-Score (t=0.5)": f"{summary['Overall_OOF_F1']:.4f} (±{summary['F1_Score_Std']:.4f})",
            "OOF Recall (t=0.5)": f"{summary['Overall_OOF_Recall']:.4f} (±{summary['Recall_Std']:.4f})",
            "OOF Precision (t=0.5)": f"{summary['Overall_OOF_Precision']:.4f} (±{summary['Precision_Std']:.4f})",
            "OOF Brier Score": f"{summary['Overall_OOF_Brier']:.4f}"
        })
    return pd.DataFrame(comparison_rows)
