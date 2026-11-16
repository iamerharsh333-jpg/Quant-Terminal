"""Caching utilities for price datasets and computations."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from config.settings import CACHE_DIR


def get_cache_path(symbol: str, start: str, end: str, cache_dir: Path | str | None = None) -> Path:
    """Generate safe filename for caching historical OHLCV data."""
    base_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "_").replace("^", "IDX_").replace(".", "_")
    return base_dir / f"{safe_symbol}_{start}_{end}.csv"


def read_cached_dataframe(cache_path: Path) -> pd.DataFrame | None:
    """Read cached OHLCV CSV from disk if present and non-empty."""
    if not cache_path.exists():
        return None
    try:
        df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception:
        return None


def write_cached_dataframe(df: pd.DataFrame, cache_path: Path) -> None:
    """Write validated OHLCV DataFrame to CSV cache."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index_label="Date")
    except Exception:
        pass

