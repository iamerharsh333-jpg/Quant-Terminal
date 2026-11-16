"""Tests for Portfolio Analytics, covariance, and risk contributions."""

from __future__ import annotations

import pandas as pd
import pytest
from core.data_provider import SyntheticDataProvider
from core.portfolio_engine import build_portfolio_analytics


def test_portfolio_analytics_basic():
    provider = SyntheticDataProvider()
    p1 = provider.fetch_ohlcv("STOCK_A", start="2022-01-01", end="2023-01-01")["Close"]
    p2 = provider.fetch_ohlcv("STOCK_B", start="2022-01-01", end="2023-01-01")["Close"]

    res = build_portfolio_analytics(
        asset_prices={"STOCK_A": p1, "STOCK_B": p2},
        weights={"STOCK_A": 0.6, "STOCK_B": 0.4},
        initial_capital=100_000.0,
    )

    assert res.portfolio_equity.iloc[0] == pytest.approx(100_000.0, rel=0.05)
    assert len(res.daily_returns) > 200
    assert "portfolio_total_return" in res.metrics
    assert "portfolio_volatility" in res.metrics
    assert res.metrics["portfolio_volatility"] > 0.0

    # Risk contribution sums to 1.0 (100% of risk)
    pct_contrib = res.risk_contributions["Pct_Risk_Contribution"].sum()
    assert abs(pct_contrib - 1.0) < 1e-3

    # Correlation matrix diagonal must be 1.0
    assert res.correlation_matrix.loc["STOCK_A", "STOCK_A"] == pytest.approx(1.0, 1e-4)
    assert res.correlation_matrix.loc["STOCK_B", "STOCK_B"] == pytest.approx(1.0, 1e-4)


def test_portfolio_empty_raises():
    with pytest.raises(ValueError):
        build_portfolio_analytics({})

