"""Tests for Data Provider abstraction and Synthetic Demo generator."""

from __future__ import annotations

import pandas as pd
import pytest
from core.data_provider import (
    BaseDataProvider,
    DataProviderFactory,
    FutureAPIAdapterPlaceholder,
    LocalCSVProvider,
    SyntheticDataProvider,
    YahooFinanceProvider,
)


def test_synthetic_provider_offline_generation():
    provider = SyntheticDataProvider()
    assert provider.is_available() is True
    assert provider.is_synthetic is True

    df = provider.fetch_ohlcv("RELIANCE.NS", start="2022-01-01", end="2023-01-01")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 200

    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in expected_cols:
        assert col in df.columns

    # High >= Low, High >= Close, High >= Open
    assert (df["High"] >= df["Low"]).all()
    assert (df["High"] >= df["Close"]).all()
    assert (df["High"] >= df["Open"]).all()
    assert (df["Low"] <= df["Close"]).all()
    assert (df["Low"] <= df["Open"]).all()
    assert (df["Volume"] > 0).all()


def test_data_provider_factory():
    demo_p = DataProviderFactory.get_provider("demo")
    assert isinstance(demo_p, SyntheticDataProvider)

    hist_p = DataProviderFactory.get_provider("historical")
    assert isinstance(hist_p, YahooFinanceProvider)

    local_p = DataProviderFactory.get_provider("local")
    assert isinstance(local_p, LocalCSVProvider)

    api_p = DataProviderFactory.get_provider("api")
    assert isinstance(api_p, FutureAPIAdapterPlaceholder)


def test_future_api_adapter_placeholder_without_keys():
    adapter = FutureAPIAdapterPlaceholder(provider_id="mock_vendor")
    with pytest.raises(NotImplementedError):
        adapter.fetch_ohlcv("AAPL", "2023-01-01", "2023-12-31")

