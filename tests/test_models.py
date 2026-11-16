"""Tests for ML Engine, time series split, training, and evaluation."""

from __future__ import annotations

import pandas as pd
import pytest
from core.data_provider import SyntheticDataProvider
from core.feature_engineering import make_features
from core.model_engine import (
    evaluate_classifier,
    get_logistic_regression_coefficients,
    get_random_forest_feature_importance,
    predict,
    predict_proba,
    predict_signals,
    time_series_train_test_split,
    train_gradient_boosting_model,
    train_logistic_regression_model,
    train_model,
    train_xgboost_model,
)


@pytest.fixture
def feature_dataset():
    provider = SyntheticDataProvider()
    df = provider.fetch_ohlcv("RELIANCE.NS", start="2021-01-01", end="2023-01-01")
    return make_features(df, horizon=3)


def test_time_series_split_chronology(feature_dataset):
    split = time_series_train_test_split(feature_dataset.X, feature_dataset.y, test_size=0.20, embargo_bars=3)

    # Strictly chronological
    assert split.X_train.index.max() < split.X_test.index.min()
    assert len(split.X_train) + len(split.X_test) <= len(feature_dataset.X)
    assert len(split.X_test) > 50


def test_model_training_and_predictions(feature_dataset):
    split = time_series_train_test_split(feature_dataset.X, feature_dataset.y, test_size=0.20)

    # 1. Random Forest
    rf = train_model(split.X_train, split.y_train, n_estimators=50, max_depth=4)
    rf_preds = predict(rf, split.X_test)
    rf_probs = predict_proba(rf, split.X_test)
    assert len(rf_preds) == len(split.X_test)
    assert len(rf_probs) == len(split.X_test)
    assert (rf_probs >= 0.0).all() and (rf_probs <= 1.0).all()

    # 2. Logistic Regression
    lr = train_logistic_regression_model(split.X_train, split.y_train)
    lr_preds = predict(lr, split.X_test)
    assert len(lr_preds) == len(split.X_test)

    # 3. Gradient Boosting
    gb = train_gradient_boosting_model(split.X_train, split.y_train, max_iter=50)
    gb_preds = predict(gb, split.X_test)
    assert len(gb_preds) == len(split.X_test)


def test_signal_generation_thresholds(feature_dataset):
    split = time_series_train_test_split(feature_dataset.X, feature_dataset.y, test_size=0.20)
    rf = train_model(split.X_train, split.y_train, n_estimators=30, max_depth=3)

    signals = predict_signals(rf, split.X_test, buy_threshold=0.60, sell_threshold=0.40)
    assert set(signals.unique()).issubset({"BUY", "SELL", "HOLD"})
    assert len(signals) == len(split.X_test)


def test_classification_metrics_evaluation():
    y_true = pd.Series([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = pd.Series([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    y_prob = pd.Series([0.8, 0.2, 0.7, 0.4, 0.3, 0.9, 0.6, 0.1, 0.85, 0.25])

    metrics = evaluate_classifier(y_true, y_pred, y_prob=y_prob)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0

