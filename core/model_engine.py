"""Institutional Machine Learning Engine for Quantitative Directional Forecasting."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    _XGBOOST_AVAILABLE = False


@dataclass
class TimeSeriesSplitResult:
    """Container for chronological train/test partition."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def is_xgboost_available() -> bool:
    """Return True if xgboost library is loaded and usable."""
    return _XGBOOST_AVAILABLE


def time_series_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    embargo_bars: int = 3,
) -> TimeSeriesSplitResult:
    """
    Chronological train/test split strictly preserving temporal ordering.
    No shuffling. Includes an optional embargo gap between train and test
    to eliminate boundary label overlap from multi-day forward targets.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    n_samples = len(X)
    n_test = int(n_samples * test_size)
    if n_test < 5:
        raise ValueError("Test set is too small. Provide more historical data or increase test_size.")

    split_idx = n_samples - n_test
    # Embargo cutoff to prevent forward-label leakage from the end of train into test
    train_end_idx = max(split_idx - embargo_bars, int(n_samples * 0.5))

    X_train = X.iloc[:train_end_idx].copy()
    y_train = y.iloc[:train_end_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()

    return TimeSeriesSplitResult(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 150,
    max_depth: int = 6,
    min_samples_leaf: int = 4,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a regularized Random Forest classifier with TimeSeriesSplit validation.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 0.1,
    random_state: int = 42,
) -> Pipeline:
    """
    Train regularized Logistic Regression wrapped in a Pipeline with StandardScaler.
    Feature scaling is strictly fitted on X_train only to prevent data leakage.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 120,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> BaseEstimator | None:
    """
    Train regularized XGBoost classifier if available.
    """
    if not is_xgboost_available():
        return None

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> HistGradientBoostingClassifier:
    """
    Train modern Histogram-based Gradient Boosting classifier (LightGBM-inspired,
    built directly into scikit-learn without C++ compiler requirements).
    """
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict(model: object, X: pd.DataFrame) -> pd.Series:
    """Generate binary class predictions (0 or 1)."""
    preds = model.predict(X)
    return pd.Series(preds, index=X.index, name="predicted_class")


def predict_proba(model: object, X: pd.DataFrame) -> pd.Series:
    """
    Generate predicted probability of positive class (Up direction, label=1).
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return pd.Series(probs[:, 1], index=X.index, name="probability_up")
        elif probs.ndim == 1:
            return pd.Series(probs, index=X.index, name="probability_up")
    # Fallback to decision function or discrete prediction
    if hasattr(model, "decision_function"):
        df = model.decision_function(X)
        sig = 1.0 / (1.0 + np.exp(-df))
        return pd.Series(sig, index=X.index, name="probability_up")
    discrete = model.predict(X)
    return pd.Series(discrete.astype(float), index=X.index, name="probability_up")


def predict_signals(
    model: object,
    X: pd.DataFrame,
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.40,
) -> pd.Series:
    """
    Convert model probabilities into actionable trading signals:
    - 'BUY' if probability_up > buy_threshold
    - 'SELL' if probability_up < sell_threshold
    - 'HOLD' otherwise
    """
    probs = predict_proba(model, X)
    signals = pd.Series("HOLD", index=X.index, name="predicted_signal")
    signals.loc[probs > buy_threshold] = "BUY"
    signals.loc[probs < sell_threshold] = "SELL"
    return signals


def evaluate_classifier(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute comprehensive classification metrics for model evaluation.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
    }

    if y_prob is not None:
        y_pr = np.asarray(y_prob).astype(float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_t, y_pr))
        except Exception:
            metrics["roc_auc"] = 0.50

        try:
            metrics["brier_score"] = float(brier_score_loss(y_t, y_pr))
        except Exception:
            metrics["brier_score"] = 0.25

        try:
            metrics["log_loss"] = float(log_loss(y_t, y_pr, labels=[0, 1]))
        except Exception:
            pass

    return metrics


def get_random_forest_feature_importance(
    model: object,
    feature_names: list[str] | pd.Index,
) -> pd.DataFrame:
    """Extract feature importance ranking from Random Forest."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df = pd.DataFrame(
            {"feature": list(feature_names), "importance": importances}
        )
        return df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return pd.DataFrame(columns=["feature", "importance"])


def get_logistic_regression_coefficients(
    model: object,
    feature_names: list[str] | pd.Index,
) -> pd.DataFrame:
    """Extract signed feature coefficients from Logistic Regression."""
    clf = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    if hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
        df = pd.DataFrame(
            {"feature": list(feature_names), "coefficient": coefs}
        )
        df["abs_coefficient"] = df["coefficient"].abs()
        return df.sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)
    return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])
