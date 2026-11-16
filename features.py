"""Feature engineering module for stock prediction (backward compatible)."""

from __future__ import annotations

from core.feature_engineering import (
    DEFAULT_HORIZON,
    FEATURE_COLUMNS,
    FeatureSet,
    build_feature_frame,
    calculate_atr,
    calculate_macd,
    calculate_rsi,
    calculate_technical_indicators,
    make_features,
)

__all__ = [
    "DEFAULT_HORIZON",
    "FEATURE_COLUMNS",
    "FeatureSet",
    "build_feature_frame",
    "make_features",
    "calculate_rsi",
    "calculate_macd",
    "calculate_atr",
    "calculate_technical_indicators",
]
