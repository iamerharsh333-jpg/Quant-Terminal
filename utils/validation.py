"""Validation utilities for data and parameters."""

from __future__ import annotations

import re
import pandas as pd

REQUIRED_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def validate_ohlcv_dataframe(df: pd.DataFrame, min_rows: int = 30) -> pd.DataFrame:
    """
    Validate that input DataFrame contains clean, sorted OHLCV columns.

    Raises ValueError with actionable explanations if invalid.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Price data must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Price DataFrame is empty.")

    missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required OHLCV columns: {missing}. Available: {list(df.columns)}"
        )

    # Check index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception as exc:
            raise ValueError(f"DataFrame index could not be converted to DatetimeIndex: {exc}") from exc

    df = df.sort_index()

    # Deduplicate index if any
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]

    # Check row count
    if len(df) < min_rows:
        raise ValueError(
            f"Insufficient historical data: received {len(df)} rows, minimum required is {min_rows}."
        )

    # Clean non-positive prices
    for col in ["Open", "High", "Low", "Close"]:
        if (df[col] <= 0).any():
            # Replace invalid non-positive prices with forward fill
            df[col] = df[col].mask(df[col] <= 0).ffill().bfill()

    # Ensure Low <= High and Open/Close are bounded
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)

    return df[REQUIRED_OHLCV_COLUMNS].copy()


def clean_ticker_symbol(symbol: str) -> str:
    """Normalize user-entered ticker symbol."""
    if not symbol or not isinstance(symbol, str):
        return ""
    cleaned = symbol.strip().upper()
    # Strip dangerous characters
    cleaned = re.sub(r"[^A-Z0-9\.\_\-\^]", "", cleaned)
    return cleaned

