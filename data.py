"""Data loading utilities for stock prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RELIANCE_SYMBOL = "RELIANCE.NS"
RELIANCE_START = "2018-01-01"
RELIANCE_END = "2024-12-31"


def _normalize_ohlcv_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Normalize vendor output into single-level OHLCV columns.

    yfinance may return MultiIndex columns (field, ticker) even for one symbol.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer selecting by ticker level when available.
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1, drop_level=True)
        else:
            # Fallback for unexpected multi-index shapes.
            df.columns = df.columns.get_level_values(0)

    # In case of duplicate single-level columns, keep the first occurrence.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    return df


def _read_cached_data(cache_path: Path) -> pd.DataFrame:
    """Read cached OHLCV data from disk."""
    df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_price_data(
    symbol: str,
    start: str,
    end: str,
    cache_dir: str = "data_cache",
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Download daily OHLCV prices for a single stock symbol.

    Data is cached to CSV to make repeat runs faster and deterministic.
    """
    safe_symbol = symbol.replace("/", "_")
    cache_path = Path(cache_dir) / f"{safe_symbol}_{start}_{end}.csv"

    if cache_path.exists() and not refresh:
        return _read_cached_data(cache_path)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "yfinance is required. Install it with: pip install yfinance"
        ) from exc

    df = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        interval="1d",
    )

    if df.empty:
        raise ValueError(
            f"No price data returned for symbol={symbol}, start={start}, end={end}."
        )

    df = _normalize_ohlcv_columns(df, symbol=symbol)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns from data source: {missing}")

    out = df[required_cols].copy().dropna().sort_index()
    out.index = pd.to_datetime(out.index)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index_label="Date")
    return out


def load_reliance_data(
    cache_dir: str = "data_cache",
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Download RELIANCE.NS OHLCV data from 2018 through 2024.
    """
    return load_price_data(
        symbol=RELIANCE_SYMBOL,
        start=RELIANCE_START,
        end=RELIANCE_END,
        cache_dir=cache_dir,
        refresh=refresh,
    )
