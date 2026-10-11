"""
Data Ingestion, Cleaning, and Leakage-Safe CV Split Manifest Module.
Phase 1 Requirements:
- Load Telco Customer Churn dataset (7,043 rows).
- Drop customerID immediately to prevent identity leakage.
- Safely clean TotalCharges: coerce whitespace strings to numeric and impute via tenure * MonthlyCharges.
- Construct Stratified 5-Fold cross-validation manifest preserving ~26.5% churn distribution.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.config import DATA_PATH, TARGET_COL, ID_COL, RANDOM_STATE, N_SPLITS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def load_raw_dataset(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load raw Telco dataset from CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df


def clean_telco_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Cleans raw dataframe:
    1. Immediately drops customerID to avoid identification leakage.
    2. Converts TotalCharges to numeric, coercing blank spaces (' ') to NaN.
    3. Imputes missing TotalCharges using deterministic formula: tenure * MonthlyCharges.
       (For tenure=0, TotalCharges is correctly set to 0.0).
    4. Encodes Churn target ('Yes' -> 1, 'No' -> 0).

    Returns:
        X (pd.DataFrame): Cleaned feature matrix without customerID and Churn.
        y (pd.Series): Binary target series.
        df_clean (pd.DataFrame): Combined cleaned dataframe for auditing and failure analysis.
    """
    df = df_raw.copy()

    # Step 1: Drop customerID immediately
    if ID_COL in df.columns:
        customer_ids = df[ID_COL]
        df = df.drop(columns=[ID_COL])
    else:
        customer_ids = pd.Series(range(len(df)), name=ID_COL)

    # Step 2: Handle TotalCharges whitespace strings
    total_charges_raw = df["TotalCharges"]
    whitespace_mask = total_charges_raw.astype(str).str.strip() == ""
    n_whitespace = whitespace_mask.sum()

    # Convert to numeric (coercing whitespace to NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Step 3: Impute missing TotalCharges using domain logic: tenure * MonthlyCharges
    missing_mask = df["TotalCharges"].isna()
    imputed_values = df.loc[missing_mask, "tenure"] * df.loc[missing_mask, "MonthlyCharges"]
    df.loc[missing_mask, "TotalCharges"] = imputed_values

    # Step 4: Encode binary target
    if TARGET_COL in df.columns:
        if set(df[TARGET_COL].dropna().unique()).issubset({"Yes", "No"}):
            y = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
        else:
            y = pd.to_numeric(df[TARGET_COL]).astype(int)
        X = df.drop(columns=[TARGET_COL])
    else:
        y = None
        X = df

    # Retain full clean dataframe with original metadata for failure analysis
    df_clean = df.copy()
    if y is not None:
        df_clean[TARGET_COL] = y
    df_clean[ID_COL] = customer_ids

    return X, y, df_clean


def get_stratified_folds(X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generates a reproducible Stratified K-Fold cross-validation split manifest.
    Ensures class balance (~26.5% churn) is preserved across all training and validation folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(X, y))
    return folds


def generate_data_dictionary(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Generates an audit-ready Data Dictionary detailing column descriptions,
    raw data types, clean data types, cardinality, missingness, and business role.
    """
    dict_records = [
        {"Column": "customerID", "Type": "Categorical (ID)", "Description": "Unique customer identifier", "Role": "Dropped (Leakage Risk)"},
        {"Column": "gender", "Type": "Binary Categorical", "Description": "Customer gender (Male, Female)", "Role": "Feature"},
        {"Column": "SeniorCitizen", "Type": "Binary (0/1)", "Description": "Whether customer is a senior citizen", "Role": "Feature"},
        {"Column": "Partner", "Type": "Binary Categorical", "Description": "Whether customer has a partner (Yes, No)", "Role": "Feature"},
        {"Column": "Dependents", "Type": "Binary Categorical", "Description": "Whether customer has dependents (Yes, No)", "Role": "Feature"},
        {"Column": "tenure", "Type": "Integer Numeric", "Description": "Number of months customer has stayed with company", "Role": "Key Feature"},
        {"Column": "PhoneService", "Type": "Binary Categorical", "Description": "Whether customer has phone service (Yes, No)", "Role": "Feature"},
        {"Column": "MultipleLines", "Type": "Categorical (3 levels)", "Description": "Whether customer has multiple lines (No, Yes, No phone service)", "Role": "Feature"},
        {"Column": "InternetService", "Type": "Categorical (3 levels)", "Description": "Internet provider type (DSL, Fiber optic, No)", "Role": "Key Feature"},
        {"Column": "OnlineSecurity", "Type": "Categorical (3 levels)", "Description": "Online security add-on (Yes, No, No internet)", "Role": "Feature"},
        {"Column": "OnlineBackup", "Type": "Categorical (3 levels)", "Description": "Online backup add-on (Yes, No, No internet)", "Role": "Feature"},
        {"Column": "DeviceProtection", "Type": "Categorical (3 levels)", "Description": "Device protection plan (Yes, No, No internet)", "Role": "Feature"},
        {"Column": "TechSupport", "Type": "Categorical (3 levels)", "Description": "Premium tech support add-on (Yes, No, No internet)", "Role": "Feature"},
        {"Column": "StreamingTV", "Type": "Categorical (3 levels)", "Description": "Streaming TV service (Yes, No, No internet)", "Role": "Feature"},
        {"Column": "StreamingMovies", "Type": "Categorical (3 levels)", "Description": "Streaming movies service (Yes, No, No internet)", "Role": "Feature"},
        {"Column": "Contract", "Type": "Categorical (3 levels)", "Description": "Contract term (Month-to-month, One year, Two year)", "Role": "Key Feature"},
        {"Column": "PaperlessBilling", "Type": "Binary Categorical", "Description": "Paperless billing enrollment (Yes, No)", "Role": "Feature"},
        {"Column": "PaymentMethod", "Type": "Categorical (4 levels)", "Description": "Payment method (Electronic check, Mailed check, Bank transfer, Credit card)", "Role": "Key Feature"},
        {"Column": "MonthlyCharges", "Type": "Continuous Numeric", "Description": "Amount charged to customer monthly ($)", "Role": "Key Feature"},
        {"Column": "TotalCharges", "Type": "Continuous Numeric", "Description": "Total amount charged to customer ($), imputed if blank", "Role": "Key Feature"},
        {"Column": "Churn", "Type": "Binary Target", "Description": "Whether customer churned within last month (Yes=1, No=0)", "Role": "Target Variable"}
    ]
    return pd.DataFrame(dict_records)
