"""Backtesting engine module (backward compatible)."""

from __future__ import annotations

from core.backtesting_engine import (
    BacktestResult,
    calculate_performance_metrics,
    plot_equity_curve,
    run_backtest,
)

__all__ = [
    "BacktestResult",
    "run_backtest",
    "calculate_performance_metrics",
    "plot_equity_curve",
]
