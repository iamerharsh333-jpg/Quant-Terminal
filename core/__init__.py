"""Core quantitative engine modules."""

from core.backtesting_engine import (
    BacktestResult,
    calculate_performance_metrics,
    plot_equity_curve,
    run_backtest,
)
from core.data_loader import (
    RELIANCE_END,
    RELIANCE_START,
    RELIANCE_SYMBOL,
    load_market_data,
    load_price_data,
    load_reliance_data,
)
from core.data_provider import (
    BaseDataProvider,
    DataProviderFactory,
    LocalCSVProvider,
    SyntheticDataProvider,
    YahooFinanceProvider,
)
from core.feature_engineering import (
    DEFAULT_HORIZON,
    FEATURE_COLUMNS,
    FeatureSet,
    build_feature_frame,
    calculate_technical_indicators,
    make_features,
)
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
from core.portfolio_engine import PortfolioAnalyticsResult, build_portfolio_analytics
from core.risk_engine import RiskAnalyticsResult, compute_risk_analytics

__all__ = [
    "RELIANCE_SYMBOL",
    "RELIANCE_START",
    "RELIANCE_END",
    "BaseDataProvider",
    "SyntheticDataProvider",
    "YahooFinanceProvider",
    "LocalCSVProvider",
    "DataProviderFactory",
    "load_market_data",
    "load_price_data",
    "load_reliance_data",
    "DEFAULT_HORIZON",
    "FEATURE_COLUMNS",
    "FeatureSet",
    "build_feature_frame",
    "make_features",
    "calculate_technical_indicators",
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
    "BacktestResult",
    "calculate_performance_metrics",
    "run_backtest",
    "plot_equity_curve",
    "PortfolioAnalyticsResult",
    "build_portfolio_analytics",
    "RiskAnalyticsResult",
    "compute_risk_analytics",
]

