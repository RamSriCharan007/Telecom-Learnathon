"""
Leakage-Safe Pipeline Construction Module.
Phase 1 & Phase 2 Requirements:
- Build scikit-learn ColumnTransformer for numerical and categorical preprocessing.
- Encapsulate all transformations (imputation, scaling, one-hot encoding) strictly INSIDE the Pipeline.
- Create Baseline (Logistic Regression balanced) and Challenger (Tree-based Ensemble) models.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, RANDOM_STATE


def build_preprocessor(drop_first: bool = False) -> ColumnTransformer:
    """
    Constructs a leakage-safe ColumnTransformer.
    
    Args:
        drop_first (bool): If True, drops first category in OneHotEncoder to prevent collinearity
                           for linear models. If False, keeps all categories for tree models.
    """
    # Numerical preprocessing: Median Imputation (safety) + Standard Scaling
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing: Most Frequent Imputation + One-Hot Encoding
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first" if drop_first else None,
                                 handle_unknown="ignore" if not drop_first else "infrequent_if_exist",
                                 sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def create_baseline_pipeline(c_param: float = 1.0) -> Pipeline:
    """
    Phase 2: Baseline Model.
    Logistic Regression with balanced class weights to address the 73.5% / 26.5% class imbalance.
    """
    preprocessor = build_preprocessor(drop_first=True)
    model = LogisticRegression(
        C=c_param,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    return pipeline


def create_challenger_pipeline(model_type: str = "xgboost") -> Pipeline:
    """
    Phase 2: Challenger Model.
    High-capacity, regularized tree ensemble model (XGBoost or Random Forest).
    Configured with class weighting and hyperparameter regularization to avoid overfitting.
    """
    preprocessor = build_preprocessor(drop_first=False)

    if model_type.lower() == "xgboost":
        # scale_pos_weight is calculated as approx (neg_count / pos_count) = 73.5 / 26.5 ~ 2.77
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2.77,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        )
    elif model_type.lower() == "random_forest":
        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown challenger model type: {model_type}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    return pipeline


def get_feature_names_from_preprocessor(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """
    Extracts human-readable transformed feature names from a fitted ColumnTransformer.
    """
    feature_names = []
    for name, trans, cols in fitted_preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "named_steps"):
            last_step = trans.named_steps[list(trans.named_steps.keys())[-1]]
            if hasattr(last_step, "get_feature_names_out"):
                names = last_step.get_feature_names_out(cols)
                feature_names.extend(names)
            else:
                feature_names.extend(cols)
        elif hasattr(trans, "get_feature_names_out"):
            feature_names.extend(trans.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)
    return feature_names
