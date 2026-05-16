"""Backward-compatible backtesting API.

The production implementation now lives in ``backtesting.engine``. This wrapper
keeps existing imports such as ``from backtest import run_backtest`` working.
"""

from backtesting.engine import *  # noqa: F401,F403

