"""Central configuration and settings for Quant Terminal."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data_cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class AppConfig:
    """Application level metadata."""

    title: str = "QUANT TERMINAL"
    subtitle: str = "Institutional Quantitative Research & Stock Analytics Platform"
    version: str = "2.0.0"
    author: str = "Senior Quantitative Research Team"
    dark_theme: bool = True


@dataclass(frozen=True)
class TradingConfig:
    """Quantitative backtesting and risk defaults."""

    initial_capital: float = 100_000.0
    default_transaction_cost: float = 0.0010  # 10 bps
    default_slippage: float = 0.0005          # 5 bps
    default_execution_lag: int = 1            # 1 bar lag (next day execution)
    default_position_size: float = 1.0        # 100% equity allocation on BUY
    annualization_factor: int = 252           # 252 trading days per year
    default_risk_free_rate: float = 0.065     # 6.5% benchmark risk-free rate


@dataclass(frozen=True)
class MLConfig:
    """Machine learning hyperparameters and defaults."""

    default_horizon: int = 3
    test_size: float = 0.20
    cv_splits: int = 5
    default_buy_threshold: float = 0.60
    default_sell_threshold: float = 0.40
    random_state: int = 42


@dataclass
class APIConfig:
    """External API credentials loaded safely from environment (optional)."""

    alpha_vantage_api_key: str = field(
        default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY", "")
    )
    polygon_api_key: str = field(
        default_factory=lambda: os.getenv("POLYGON_API_KEY", "")
    )
    tiingo_api_key: str = field(
        default_factory=lambda: os.getenv("TIINGO_API_KEY", "")
    )
    zerodha_api_key: str = field(
        default_factory=lambda: os.getenv("ZERODHA_API_KEY", "")
    )

    @property
    def has_any_credentials(self) -> bool:
        return bool(
            self.alpha_vantage_api_key
            or self.polygon_api_key
            or self.tiingo_api_key
            or self.zerodha_api_key
        )


# Global instances
APP_CONFIG = AppConfig()
TRADING_CONFIG = TradingConfig()
ML_CONFIG = MLConfig()
API_CONFIG = APIConfig()

# Standard asset universes
INDIAN_EQUITIES = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Larsen & Toubro": "LT.NS",
    "ITC Limited": "ITC.NS",
    "Tata Motors": "TATAMOTORS.NS",
}

US_EQUITIES = {
    "Apple Inc.": "AAPL",
    "Microsoft Corp.": "MSFT",
    "NVIDIA Corp.": "NVDA",
    "Amazon.com Inc.": "AMZN",
    "Alphabet Inc. (Google)": "GOOGL",
    "Meta Platforms": "META",
    "Tesla Inc.": "TSLA",
    "JPMorgan Chase & Co.": "JPM",
}

BENCHMARK_INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
}

