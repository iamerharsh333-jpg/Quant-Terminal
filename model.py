"""Model training and evaluation module (backward compatible)."""

from __future__ import annotations

from core.model_engine import (
    TimeSeriesSplitResult,
    evaluate_classifier,
    get_logistic_regression_coefficients,
    get_random_forest_feature_importance,
    is_xgboost_available,
    predict,
    predict_proba,
    predict_signals,
    time_series_train_test_split,
    train_gradient_boosting_model,
    train_logistic_regression_model,
    train_model,
    train_xgboost_model,
)

__all__ = [
    "TimeSeriesSplitResult",
    "time_series_train_test_split",
    "train_model",
    "train_logistic_regression_model",
    "train_xgboost_model",
    "train_gradient_boosting_model",
    "predict",
    "predict_proba",
    "predict_signals",
    "evaluate_classifier",
    "get_random_forest_feature_importance",
    "get_logistic_regression_coefficients",
    "is_xgboost_available",
]
