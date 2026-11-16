"""Tests for technical feature engineering and target calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from core.data_provider import SyntheticDataProvider
from core.feature_engineering import (
    FEATURE_COLUMNS,
    build_feature_frame,
    calculate_macd,
    calculate_rsi,
    calculate_technical_indicators,
    make_features,
)


@pytest.fixture
def sample_ohlcv():
    provider = SyntheticDataProvider()
    return provider.fetch_ohlcv("TEST", start="2022-01-01", end="2023-06-01")


def test_calculate_rsi_bounds(sample_ohlcv):
    rsi = calculate_rsi(sample_ohlcv["Close"], period=14)
    assert (rsi >= 0.0).all()
    assert (rsi <= 100.0).all()
    assert len(rsi) == len(sample_ohlcv)


def test_calculate_macd_components(sample_ohlcv):
    macd_line, sig_line, hist = calculate_macd(sample_ohlcv["Close"])
    assert len(macd_line) == len(sample_ohlcv)
    # Histogram is line - signal
    np.testing.assert_allclose(hist.values, (macd_line - sig_line).values, rtol=1e-5)


def test_technical_indicators(sample_ohlcv):
    df = calculate_technical_indicators(sample_ohlcv)
    expected_indicators = ["SMA_20", "SMA_50", "SMA_200", "BB_Upper", "BB_Lower", "RSI_14", "MACD", "ATR_14"]
    for ind in expected_indicators:
        assert ind in df.columns
    # Bollinger Upper >= Lower
    valid_bb = df.dropna(subset=["BB_Upper", "BB_Lower"])
    assert (valid_bb["BB_Upper"] >= valid_bb["BB_Lower"]).all()


def test_make_features_structure(sample_ohlcv):
    horizon = 3
    feature_set = make_features(sample_ohlcv, horizon=horizon)

    assert isinstance(feature_set.X, pd.DataFrame)
    assert isinstance(feature_set.y, pd.Series)
    assert len(feature_set.X) == len(feature_set.y)
    assert set(FEATURE_COLUMNS).issubset(set(feature_set.X.columns))

    # Binary target
    assert set(feature_set.y.unique()).issubset({0, 1})

    # No NaNs in X or y
    assert not feature_set.X.isna().any().any()
    assert not feature_set.y.isna().any()

