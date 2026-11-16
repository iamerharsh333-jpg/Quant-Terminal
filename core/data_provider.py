"""Modular data providers supporting Demo, Historical, and API modes."""

from __future__ import annotations

import abc
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from config.settings import API_CONFIG, CACHE_DIR
from utils.caching import get_cache_path, read_cached_dataframe, write_cached_dataframe
from utils.validation import clean_ticker_symbol, validate_ohlcv_dataframe


class BaseDataProvider(abc.ABC):
    """Abstract base class for quantitative market data providers."""

    @abc.abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch historical daily OHLCV prices for symbol between start and end dates."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if provider dependencies, network, or credentials are valid."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider description."""
        raise NotImplementedError

    @property
    def is_synthetic(self) -> bool:
        """Whether returned data is synthetic."""
        return False


class SyntheticDataProvider(BaseDataProvider):
    """
    Realistic offline synthetic market data generator.

    Uses Geometric Brownian Motion with stochastic volatility clustering
    and realistic volume/spread dynamics. Operates completely offline with
    zero external dependencies or API keys.
    """

    @property
    def provider_name(self) -> str:
        return "Synthetic Demo Engine (Offline)"

    @property
    def is_synthetic(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str = "2018-01-01",
        end: str = "2024-12-31",
        refresh: bool = False,
    ) -> pd.DataFrame:
        clean_sym = clean_ticker_symbol(symbol) or "DEMO"

        # Deterministic seed based on symbol name to make demo consistent
        seed = int(hashlib.md5(clean_sym.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)
        rng = np.random.default_rng(seed)

        # Generate business days calendar
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        date_range = pd.date_range(start=start_dt, end=end_dt, freq="B")
        n_bars = len(date_range)
        if n_bars < 30:
            # Fallback if range is too narrow
            date_range = pd.date_range(end=datetime.now(), periods=250, freq="B")
            n_bars = len(date_range)

        # Baseline price calibration depending on asset class
        base_price = 2400.0 if ".NS" in clean_sym else 180.0
        annual_drift = 0.12  # 12% long-term drift
        annual_vol = 0.22    # 22% volatility
        dt = 1.0 / 252.0

        # GARCH-style stochastic volatility modulation
        returns = np.zeros(n_bars)
        vol_process = np.ones(n_bars) * (annual_vol * np.sqrt(dt))

        for t in range(1, n_bars):
            # Volatility clustering
            vol_process[t] = np.sqrt(
                0.00002 + 0.15 * (returns[t - 1] ** 2) + 0.82 * (vol_process[t - 1] ** 2)
            )
            # Bound daily volatility between 0.4% and 4.5%
            vol_process[t] = np.clip(vol_process[t], 0.004, 0.045)
            # Random shock
            shock = rng.standard_normal()
            returns[t] = (annual_drift * dt) + (vol_process[t] * shock)

        # Close price series
        close_prices = base_price * np.exp(np.cumsum(returns))

        # Intraday High, Low, Open synthesis
        # Open price with overnight gap
        overnight_gap = rng.normal(0.0, 0.003, size=n_bars)
        open_prices = np.roll(close_prices, 1) * (1.0 + overnight_gap)
        open_prices[0] = close_prices[0] * 0.998

        # Intraday range proportional to volatility
        intraday_vol = np.maximum(np.abs(returns), vol_process)
        high_spread = rng.uniform(0.3, 1.2, size=n_bars) * intraday_vol
        low_spread = rng.uniform(0.3, 1.2, size=n_bars) * intraday_vol

        high_prices = np.maximum(open_prices, close_prices) * (1.0 + high_spread)
        low_prices = np.minimum(open_prices, close_prices) * (1.0 - low_spread)

        # Volume process with lognormal distribution and correlation to absolute returns
        base_volume = 4_500_000 if ".NS" in clean_sym else 25_000_000
        vol_shock = rng.lognormal(mean=0.0, sigma=0.45, size=n_bars)
        activity_multiplier = 1.0 + 3.0 * np.abs(returns)
        volumes = (base_volume * vol_shock * activity_multiplier).astype(int)

        df = pd.DataFrame(
            {
                "Open": np.round(open_prices, 2),
                "High": np.round(high_prices, 2),
                "Low": np.round(low_prices, 2),
                "Close": np.round(close_prices, 2),
                "Volume": volumes,
            },
            index=date_range,
        )
        df.index.name = "Date"
        return validate_ohlcv_dataframe(df)


class YahooFinanceProvider(BaseDataProvider):
    """Yahoo Finance provider using yfinance with local caching and retries."""

    @property
    def provider_name(self) -> str:
        return "Yahoo Finance Historical API"

    def is_available(self) -> bool:
        try:
            import yfinance  # noqa: F401
            return True
        except ImportError:
            return False

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str = "2018-01-01",
        end: str = "2024-12-31",
        refresh: bool = False,
    ) -> pd.DataFrame:
        clean_sym = clean_ticker_symbol(symbol)
        if not clean_sym:
            raise ValueError("Invalid empty ticker symbol.")

        cache_path = get_cache_path(clean_sym, start, end)

        if not refresh:
            cached_df = read_cached_dataframe(cache_path)
            if cached_df is not None and not cached_df.empty:
                return validate_ohlcv_dataframe(cached_df)

        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "yfinance is not installed. Install via: pip install yfinance"
            ) from exc

        # Download with timeout and error handling
        try:
            df = yf.download(
                tickers=clean_sym,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                interval="1d",
                timeout=10,
            )
        except Exception as exc:
            # If network fails but cache exists, use cache
            cached_df = read_cached_dataframe(cache_path)
            if cached_df is not None and not cached_df.empty:
                return validate_ohlcv_dataframe(cached_df)
            raise RuntimeError(
                f"Failed to fetch market data for '{clean_sym}' from Yahoo Finance: {exc}"
            ) from exc

        if df is None or df.empty:
            cached_df = read_cached_dataframe(cache_path)
            if cached_df is not None and not cached_df.empty:
                return validate_ohlcv_dataframe(cached_df)
            raise ValueError(
                f"No market data returned for '{clean_sym}' between {start} and {end}."
            )

        # Normalize MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            if clean_sym in df.columns.get_level_values(-1):
                df = df.xs(clean_sym, axis=1, level=-1, drop_level=True)
            else:
                df.columns = df.columns.get_level_values(0)

        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        df = validate_ohlcv_dataframe(df)
        write_cached_dataframe(df, cache_path)
        return df


class LocalCSVProvider(BaseDataProvider):
    """Local CSV provider for offline datasets and backtesting."""

    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = Path(directory) if directory else CACHE_DIR

    @property
    def provider_name(self) -> str:
        return f"Local CSV Store ({self.directory})"

    def is_available(self) -> bool:
        return self.directory.exists()

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str = "2018-01-01",
        end: str = "2024-12-31",
        refresh: bool = False,
    ) -> pd.DataFrame:
        clean_sym = clean_ticker_symbol(symbol)
        cache_path = get_cache_path(clean_sym, start, end, cache_dir=self.directory)
        df = read_cached_dataframe(cache_path)
        if df is not None and not df.empty:
            return validate_ohlcv_dataframe(df)

        # Try generic match in directory
        for file in self.directory.glob(f"*{clean_sym}*.csv"):
            df = read_cached_dataframe(file)
            if df is not None and not df.empty:
                return validate_ohlcv_dataframe(df)

        raise FileNotFoundError(
            f"No local CSV found for '{clean_sym}' in {self.directory}."
        )


class FutureAPIAdapterPlaceholder(BaseDataProvider):
    """
    Placeholder adapter for future commercial API connections
    (e.g., AlphaVantage, Tiingo, Polygon, Zerodha Kite).
    """

    def __init__(self, provider_id: str = "commercial_api") -> None:
        self.provider_id = provider_id

    @property
    def provider_name(self) -> str:
        return f"External API Adapter ({self.provider_id}) [Configuration Pending]"

    def is_available(self) -> bool:
        return API_CONFIG.has_any_credentials

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        if not self.is_available():
            raise NotImplementedError(
                f"External API mode is inactive. No API keys configured for '{self.provider_id}'. "
                "Set ALPHA_VANTAGE_API_KEY, POLYGON_API_KEY, or TIINGO_API_KEY in .env to activate."
            )
        raise NotImplementedError("Direct commercial API adapter placeholder.")


class DataProviderFactory:
    """Factory to produce and configure market data providers."""

    @staticmethod
    def get_provider(mode: str = "demo") -> BaseDataProvider:
        mode_lower = (mode or "").lower().strip()
        if mode_lower == "demo":
            return SyntheticDataProvider()
        elif mode_lower in ("historical", "yahoo"):
            return YahooFinanceProvider()
        elif mode_lower == "local":
            return LocalCSVProvider()
        elif mode_lower == "api":
            return FutureAPIAdapterPlaceholder()
        return SyntheticDataProvider()

