"""Tests for Backtesting Engine, transaction cost accounting, and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from core.backtesting_engine import calculate_performance_metrics, run_backtest


def test_calculate_performance_metrics_exact_math():
    # Independently calculated verification case
    # Daily returns: 5 days with known gains and losses
    strat_ret = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01])
    mkt_ret = pd.Series([0.005, 0.002, 0.01, -0.005, 0.005])
    pos = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    turnover = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0])

    metrics = calculate_performance_metrics(
        strategy_returns=strat_ret,
        market_returns=mkt_ret,
        positions=pos,
        turnover=turnover,
        costs_paid=100.0,
        initial_capital=100_000.0,
        risk_free_rate=0.0,
        annualization_factor=252,
    )

    # Expected cumulative return: (1.01)*(0.995)*(1.02)*(1.015)*(0.99) - 1.0
    expected_cum = (1.01 * 0.995 * 1.02 * 1.015 * 0.99) - 1.0
    assert abs(metrics["strategy_total_return"] - expected_cum) < 1e-4
    assert metrics["strategy_max_drawdown"] >= 0.0
    assert metrics["win_rate"] == 0.60  # 3 positive days out of 5


def test_run_backtest_with_execution_lag():
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    # Alternating signals
    signals = pd.Series(["HOLD", "BUY", "BUY", "BUY", "SELL", "SELL", "HOLD", "BUY", "BUY", "HOLD"], index=dates)
    mkt_returns = pd.Series([0.01] * 10, index=dates)

    res = run_backtest(
        market_returns=mkt_returns,
        predictions=signals,
        transaction_cost=0.001,
        slippage=0.0005,
        execution_lag=1,
    )

    df = res.frame
    # Execution lag: position at t=1 is 0.0 even though signal is BUY
    assert df["position"].iloc[1] == 0.0
    # Position at t=2 becomes 1.0 (executed 1 bar after BUY signal at t=1)
    assert df["position"].iloc[2] == 1.0

    # Turnover is non-zero ONLY when position flips
    assert df["turnover"].iloc[2] == 1.0
    assert df["turnover"].iloc[3] == 0.0  # static hold -> zero turnover and zero cost!


def test_backtest_empty_input():
    res = run_backtest(pd.Series(dtype=float), pd.Series(dtype=str))
    assert res.frame.empty
    assert res.metrics == {}

