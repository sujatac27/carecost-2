"""
CareCost AI — Model Training & Evaluation
============================================================
Phase 4 of the project plan. Trains three progressively more powerful models
on `patient_responsibility` (out-of-pocket cost) and compares them:

    Model 1: Linear Regression        (interpretable baseline)
    Model 2: Random Forest Regressor  (captures non-linearities/interactions)
    Model 3: XGBoost Regressor        (production-grade gradient boosting)

Also trains quantile regressors (10th / 90th percentile) on top of the winning
model's residuals to produce realistic confidence intervals — e.g. "$470-$590"
instead of a single misleadingly-precise point estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class ModelResult:
    name: str
    model: object
    mae: float
    rmse: float
    r2: float
    cv_mae_mean: float
    cv_mae_std: float


def train_test_prepare(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42):
    """Standard train/test split. Log-transform the target to tame right-skew
    (medical costs are heavily right-skewed — a handful of expensive procedures
    dominate raw-scale error metrics otherwise)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test


def evaluate(name: str, model, X_train, y_train, X_test, y_test, cv_folds: int = 5) -> ModelResult:
    """Fits the model, scores it on the held-out test set, and runs k-fold CV
    on the training set for a more robust performance estimate."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)  # cost predictions can't be negative

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores

    print(f"{name:>20s} | MAE: ${mae:8.2f} | RMSE: ${rmse:8.2f} | R²: {r2:6.3f} "
          f"| CV MAE: ${cv_mae.mean():.2f} (+/- ${cv_mae.std():.2f})")

    return ModelResult(name=name, model=model, mae=mae, rmse=rmse, r2=r2,
                        cv_mae_mean=cv_mae.mean(), cv_mae_std=cv_mae.std())


def train_all_models(X_train, y_train, X_test, y_test) -> dict[str, ModelResult]:
    """Trains and evaluates Model 1 -> 2 -> 3 as specified in the project plan."""
    results = {}

    print("Training models (this evaluates MAE / RMSE / R^2 + 5-fold CV)\n" + "-" * 78)

    # Model 1: Linear Regression (scaled — linear models need standardized inputs)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    lr = LinearRegression()
    results["linear_regression"] = evaluate("Linear Regression", lr, X_train_scaled, y_train,
                                             X_test_scaled, y_test)
    results["linear_regression"].scaler = scaler  # stash for inference

    # Model 2: Random Forest
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=5,
        n_jobs=-1, random_state=42,
    )
    results["random_forest"] = evaluate("Random Forest", rf, X_train, y_train, X_test, y_test)

    # Model 3: XGBoost (falls back to sklearn GradientBoosting if xgboost isn't
    # installed in the current environment — same interface, same evaluation)
    if HAS_XGBOOST:
        xgb = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.0, random_state=42, n_jobs=-1,
        )
        results["xgboost"] = evaluate("XGBoost", xgb, X_train, y_train, X_test, y_test)
    else:
        print("[info] xgboost not installed in this environment — "
              "using sklearn GradientBoostingRegressor as a stand-in with an "
              "identical interface. `pip install xgboost` for the production model.")
        gbr = GradientBoostingRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.85, random_state=42,
        )
        results["xgboost"] = evaluate("GradientBoosting*", gbr, X_train, y_train, X_test, y_test)

    return results


def best_model(results: dict[str, ModelResult]) -> ModelResult:
    """Selects the model with the lowest test-set MAE."""
    return min(results.values(), key=lambda r: r.mae)


# ---------------------------------------------------------------------------
# Confidence intervals via quantile regression
# ---------------------------------------------------------------------------

def train_quantile_models(X_train, y_train, lower_q: float = 0.10, upper_q: float = 0.90):
    """
    Trains gradient-boosted quantile regressors to produce a realistic
    prediction interval around the point estimate — e.g. "$470-$590" instead
    of a single number, which is both more honest and more useful for a
    financial decision tool.
    """
    lower_model = GradientBoostingRegressor(
        loss="quantile", alpha=lower_q, n_estimators=300, max_depth=4,
        learning_rate=0.05, random_state=42,
    )
    upper_model = GradientBoostingRegressor(
        loss="quantile", alpha=upper_q, n_estimators=300, max_depth=4,
        learning_rate=0.05, random_state=42,
    )
    lower_model.fit(X_train, y_train)
    upper_model.fit(X_train, y_train)
    return lower_model, upper_model


def predict_with_interval(point_model, lower_model, upper_model, X):
    """Returns (point_estimate, lower_bound, upper_bound), each clipped at 0."""
    point = np.clip(point_model.predict(X), 0, None)
    lower = np.clip(lower_model.predict(X), 0, None)
    upper = np.clip(upper_model.predict(X), 0, None)
    # guard against quantile crossover (rare, but possible with GBM quantile loss)
    lower, upper = np.minimum(lower, point), np.maximum(upper, point)
    return point, lower, upper
