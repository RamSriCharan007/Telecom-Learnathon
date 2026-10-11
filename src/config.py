"""
Configuration and constants for the Telco Churn Machine Learning Pipeline.
Leakage-safe Baseline vs. Tree Model Hackathon Submission.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "reports", "metrics")

# Create output directories if not present
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Random Seed & CV Folds
RANDOM_STATE = 42
N_SPLITS = 5

# Target column
TARGET_COL = "Churn"
ID_COL = "customerID"

# Feature definitions
NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

# Operational Tier Thresholds
# Tiers: LOW_RISK (< TIER_LOW), WATCH (TIER_LOW to TIER_HIGH), OFFER_RETENTION (>= TIER_HIGH)
DEFAULT_TIER_LOW = 0.30
DEFAULT_TIER_HIGH = 0.50

# Business Cost Matrix Assumptions (for Cost-Tuning)
# Retention campaign cost vs churn loss
RETENTION_COST = 50.0       # Cost of offering retention incentives (discounts/vouchers)
CHURN_LOSS = 300.0          # Customer lifetime value lost if customer churns unaddressed
RETENTION_SUCCESS_RATE = 0.50  # Probability that an offer saves a customer who intended to churn
