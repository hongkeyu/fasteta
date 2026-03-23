"""
model.py
Shared XGBoost train/evaluate. Config never changes across stages.
"""
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Fixed config — identical across all stages
# Changing these makes cross-stage MAE comparisons invalid
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100


def train_and_evaluate(features, targets):
    """
    Train XGBoost and return MAE on held-out test set.

    Why XGBoost over Random Forest:
    - Delivery time has complex feature interactions
      (high traffic + rain + low driver rating compound nonlinearly)
    - XGBoost builds trees sequentially, each correcting previous errors

    Why MAE not RMSE:
    - MAE is in minutes — directly interpretable
    - RMSE penalizes outliers more — less meaningful here
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE  # fixed — never change
    )
    model = XGBRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        verbosity=0  # suppress XGBoost output
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return mean_absolute_error(y_test, predictions)
