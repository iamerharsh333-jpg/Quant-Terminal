"""Tests for Risk Analytics Engine: VaR, CVaR, Beta, Drawdowns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from core.risk_engine import compute_risk_analytics


def test_compute_risk_analytics_gaussian():
    # Gaussian synthetic return series
    rng = np.random.default_rng(42)
    daily_returns = pd.Series(rng.normal(0.0005, 0.015, size=500))
    benchmark_returns = pd.Series(rng.normal(0.0004, 0.012, size=500))

    risk = compute_risk_analytics(
        stock_returns=daily_returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=0.05,
    )

    # VaR 99% > VaR 95%
    assert risk.historical_var_99 > risk.historical_var_95
    # CVaR >= VaR
    assert risk.cvar_95 >= risk.historical_var_95
    assert risk.cvar_99 >= risk.historical_var_99

    assert risk.annualized_volatility > 0.0
    assert risk.max_drawdown >= 0.0
    assert len(risk.rolling_volatility_20d) == 500


def test_insufficient_risk_data_raises():
    short_returns = pd.Series([0.01, -0.02, 0.03])
    with pytest.raises(ValueError, match="At least 20 historical return periods"):
        compute_risk_analytics(short_returns)

