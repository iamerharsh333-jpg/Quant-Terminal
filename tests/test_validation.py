"""Tests for input validation and OHLCV sanitization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from utils.validation import clean_ticker_symbol, validate_ohlcv_dataframe


def test_clean_ticker_symbol():
    assert clean_ticker_symbol(" reliance.ns  ") == "RELIANCE.NS"
    assert clean_ticker_symbol("AAPL; DROP TABLE;") == "AAPLDROPTABLE"
    assert clean_ticker_symbol("^NSEI") == "^NSEI"
    assert clean_ticker_symbol("") == ""


def test_validate_ohlcv_valid():
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    df = pd.DataFrame(
        {
            "Open": np.linspace(100, 150, 50),
            "High": np.linspace(105, 155, 50),
            "Low": np.linspace(95, 145, 50),
            "Close": np.linspace(102, 152, 50),
            "Volume": np.full(50, 1000000),
        },
        index=dates,
    )
    validated = validate_ohlcv_dataframe(df)
    assert len(validated) == 50
    assert list(validated.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_validate_ohlcv_missing_columns():
    df = pd.DataFrame({"Close": [10, 20, 30]})
    with pytest.raises(ValueError, match="Missing required OHLCV columns"):
        validate_ohlcv_dataframe(df)


def test_validate_ohlcv_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_ohlcv_dataframe(pd.DataFrame())


def test_validate_ohlcv_insufficient_rows():
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    df = pd.DataFrame(
        {
            "Open": range(10),
            "High": range(10),
            "Low": range(10),
            "Close": range(10),
            "Volume": range(10),
        },
        index=dates,
    )
    with pytest.raises(ValueError, match="Insufficient historical data"):
        validate_ohlcv_dataframe(df, min_rows=30)

