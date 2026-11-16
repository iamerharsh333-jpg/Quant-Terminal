"""High-level market data loading pipeline with multi-mode fallback."""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

from config.settings import CACHE_DIR
from core.data_provider import (
    BaseDataProvider,
    DataProviderFactory,
    SyntheticDataProvider,
    YahooFinanceProvider,
)
from utils.validation import clean_ticker_symbol

logger = logging.getLogger(__name__)

RELIANCE_SYMBOL = "RELIANCE.NS"
RELIANCE_START = "2018-01-01"
RELIANCE_END = "2024-12-31"


def load_market_data(
    symbol: str,
    start: str = RELIANCE_START,
    end: str = RELIANCE_END,
    mode: str = "demo",
    refresh: bool = False,
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, BaseDataProvider]:
    """
    Load OHLCV dataframe using requested mode, falling back to synthetic
    data gracefully if external networks or dependencies fail.

    Returns:
        tuple of (pd.DataFrame, provider_used)
    """
    provider = DataProviderFactory.get_provider(mode=mode)
    clean_sym = clean_ticker_symbol(symbol) or RELIANCE_SYMBOL

    try:
        df = provider.fetch_ohlcv(symbol=clean_sym, start=start, end=end, refresh=refresh)
        return df, provider
    except Exception as exc:
        logger.warning(
            f"Provider '{provider.provider_name}' failed for {clean_sym} ({exc}). "
            "Falling back to Synthetic Demo Provider."
        )
        fallback = SyntheticDataProvider()
        df = fallback.fetch_ohlcv(symbol=clean_sym, start=start, end=end, refresh=refresh)
        return df, fallback


def load_price_data(
    symbol: str,
    start: str = RELIANCE_START,
    end: str = RELIANCE_END,
    cache_dir: str = "data_cache",
    refresh: bool = False,
    mode: str = "historical",
) -> pd.DataFrame:
    """
    Backward-compatible entry point for loading price data.
    Attempts historical Yahoo Finance; if offline or fails, falls back gracefully.
    """
    df, _ = load_market_data(
        symbol=symbol,
        start=start,
        end=end,
        mode=mode,
        refresh=refresh,
        cache_dir=cache_dir,
    )
    return df


def load_reliance_data(
    cache_dir: str = "data_cache",
    refresh: bool = False,
    mode: str = "historical",
) -> pd.DataFrame:
    """Backward-compatible loader for standard Reliance benchmark."""
    return load_price_data(
        symbol=RELIANCE_SYMBOL,
        start=RELIANCE_START,
        end=RELIANCE_END,
        cache_dir=cache_dir,
        refresh=refresh,
        mode=mode,
    )

