"""Feature engineering for stock prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_HORIZON = 3
FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "log_ret_1d",
    "close_vs_ma_5",
    "close_vs_ma_10",
    "close_vs_ma_20",
    "ma_5_vs_ma_20",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_band_width",
    "bb_percent_b",
    "rsi_14",
    "rsi_7",
    "rsi_14_scaled",
    "atr_14",
    "atr_14_normalized",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_60d",
    "volatility_ratio_5d_20d",
    "annualized_volatility_20d",
    "annualized_volatility_60d",
    "high_low_spread",
    "close_open_spread",
    "volume_change",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "volume_trend_5d",
    "dollar_volume_ratio_20d",
    "obv_change_5d",
]


@dataclass
class FeatureSet:
    """Container for model-ready features and target."""

    X: pd.DataFrame
    y: pd.Series


def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing with pandas.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range using Wilder smoothing."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _validate_and_sort_price_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    """Validate the OHLCV input schema and return a date-sorted copy."""
    df = price_df.copy().sort_index()
    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"price_df is missing required columns: {', '.join(missing_columns)}"
        )
    return df


def build_feature_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix without requiring target availability."""
    df = _validate_and_sort_price_frame(price_df)

    # Return features
    df["ret_1d"] = df["Close"].pct_change(1)
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_10d"] = df["Close"].pct_change(10)
    df["log_ret_1d"] = np.log(df["Close"] / df["Close"].shift(1))

    # Moving-average features
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["close_vs_ma_5"] = (df["Close"] / df["ma_5"]) - 1.0
    df["close_vs_ma_10"] = (df["Close"] / df["ma_10"]) - 1.0
    df["close_vs_ma_20"] = (df["Close"] / df["ma_20"]) - 1.0
    df["ma_5_vs_ma_20"] = (df["ma_5"] / df["ma_20"]) - 1.0

    # MACD trend and momentum features
    df["macd_line"], df["macd_signal"], df["macd_hist"] = _calculate_macd(df["Close"])

    # Bollinger Band features
    df["bb_mid_20"] = df["Close"].rolling(20).mean()
    df["bb_std_20"] = df["Close"].rolling(20).std()
    df["bb_upper_20"] = df["bb_mid_20"] + (2.0 * df["bb_std_20"])
    df["bb_lower_20"] = df["bb_mid_20"] - (2.0 * df["bb_std_20"])
    df["bb_band_width"] = (df["bb_upper_20"] - df["bb_lower_20"]) / df["bb_mid_20"]
    df["bb_percent_b"] = (df["Close"] - df["bb_lower_20"]) / (
        df["bb_upper_20"] - df["bb_lower_20"]
    )

    # RSI momentum features
    df["rsi_14"] = _calculate_rsi(df["Close"], period=14)
    df["rsi_7"] = _calculate_rsi(df["Close"], period=7)
    df["rsi_14_scaled"] = df["rsi_14"] / 100.0

    # Volatility and intraday structure
    df["atr_14"] = _calculate_atr(df["High"], df["Low"], df["Close"], period=14)
    df["atr_14_normalized"] = df["atr_14"] / df["Close"]
    df["volatility_5d"] = df["ret_1d"].rolling(5).std()
    df["volatility_10d"] = df["ret_1d"].rolling(10).std()
    df["volatility_20d"] = df["ret_1d"].rolling(20).std()
    df["volatility_60d"] = df["ret_1d"].rolling(60).std()
    df["volatility_ratio_5d_20d"] = df["volatility_5d"] / df["volatility_20d"]
    df["annualized_volatility_20d"] = df["volatility_20d"] * np.sqrt(252)
    df["annualized_volatility_60d"] = df["volatility_60d"] * np.sqrt(252)
    df["high_low_spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["close_open_spread"] = (df["Close"] - df["Open"]) / df["Open"]

    # Volume participation features
    df["volume_change"] = df["Volume"].pct_change(1)
    df["volume_ma_5"] = df["Volume"].rolling(5).mean()
    df["volume_ma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio_5d"] = df["Volume"] / df["volume_ma_5"]
    df["volume_ratio_20d"] = df["Volume"] / df["volume_ma_20"]
    df["volume_trend_5d"] = (df["volume_ma_5"] / df["volume_ma_20"]) - 1.0
    df["dollar_volume"] = df["Close"] * df["Volume"]
    df["dollar_volume_ma_20"] = df["dollar_volume"].rolling(20).mean()
    df["dollar_volume_ratio_20d"] = df["dollar_volume"] / df["dollar_volume_ma_20"]
    signed_volume = np.sign(df["Close"].diff()).fillna(0.0) * df["Volume"]
    df["obv"] = signed_volume.cumsum()
    df["obv_change_5d"] = df["obv"].pct_change(5)

    return df[FEATURE_COLUMNS].dropna().copy()


def make_features(price_df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> FeatureSet:
    """
    Build tabular features from OHLCV data.

    Target is binary forward-return direction over `horizon` day(s):
    1 if the forward return is positive, else 0.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    df = _validate_and_sort_price_frame(price_df)
    feature_frame = build_feature_frame(df)

    # Targets for forward return magnitude and direction.
    future_close = df["Close"].shift(-horizon)
    df["target_return"] = (future_close / df["Close"]) - 1.0
    df["target"] = (df["target_return"] > 0.0).astype(int)
    target_frame = df[["target", "target_return"]]
    model_df = feature_frame.join(target_frame, how="left").dropna().copy()
    X = model_df[FEATURE_COLUMNS]
    y = model_df["target"].astype(int)
    return FeatureSet(X=X, y=y)
