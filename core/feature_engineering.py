"""Feature engineering for quantitative stock modeling and technical analysis."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from utils.validation import validate_ohlcv_dataframe

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


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Prevent division by zero
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def calculate_macd(
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


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range using Wilder's smoothing."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def calculate_technical_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive technical indicators for charting and analytics.
    Includes SMA 20, 50, 200, Bollinger Bands, RSI, MACD, ATR, Volatility.
    """
    df = validate_ohlcv_dataframe(price_df)

    # Moving averages
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # Bollinger Bands (20, 2)
    df["BB_Mid"] = df["SMA_20"]
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + (2.0 * bb_std)
    df["BB_Lower"] = df["BB_Mid"] - (2.0 * bb_std)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
    df["BB_PctB"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]).replace(0.0, np.nan)

    # RSI
    df["RSI_14"] = calculate_rsi(df["Close"], period=14)
    df["RSI_7"] = calculate_rsi(df["Close"], period=7)

    # MACD
    macd_line, macd_sig, macd_hist = calculate_macd(df["Close"])
    df["MACD"] = macd_line
    df["MACD_Signal"] = macd_sig
    df["MACD_Hist"] = macd_hist

    # ATR
    df["ATR_14"] = calculate_atr(df["High"], df["Low"], df["Close"], period=14)

    # Volatility
    daily_ret = df["Close"].pct_change()
    df["Daily_Return"] = daily_ret
    df["Rolling_Vol_20d"] = daily_ret.rolling(20).std() * np.sqrt(252)

    return df


def build_feature_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix for ML inference or training.
    Strictly past/contemporaneous features; no forward information.
    """
    df = validate_ohlcv_dataframe(price_df)

    # Returns
    df["ret_1d"] = df["Close"].pct_change(1)
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_10d"] = df["Close"].pct_change(10)
    df["log_ret_1d"] = np.log(df["Close"] / df["Close"].shift(1))

    # Moving-average ratios
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["close_vs_ma_5"] = (df["Close"] / df["ma_5"]) - 1.0
    df["close_vs_ma_10"] = (df["Close"] / df["ma_10"]) - 1.0
    df["close_vs_ma_20"] = (df["Close"] / df["ma_20"]) - 1.0
    df["ma_5_vs_ma_20"] = (df["ma_5"] / df["ma_20"]) - 1.0

    # MACD
    df["macd_line"], df["macd_signal"], df["macd_hist"] = calculate_macd(df["Close"])

    # Bollinger Bands
    df["bb_mid_20"] = df["Close"].rolling(20).mean()
    df["bb_std_20"] = df["Close"].rolling(20).std()
    df["bb_upper_20"] = df["bb_mid_20"] + (2.0 * df["bb_std_20"])
    df["bb_lower_20"] = df["bb_mid_20"] - (2.0 * df["bb_std_20"])
    df["bb_band_width"] = (df["bb_upper_20"] - df["bb_lower_20"]) / df["bb_mid_20"]
    bb_range = (df["bb_upper_20"] - df["bb_lower_20"]).replace(0.0, np.nan)
    df["bb_percent_b"] = (df["Close"] - df["bb_lower_20"]) / bb_range

    # RSI
    df["rsi_14"] = calculate_rsi(df["Close"], period=14)
    df["rsi_7"] = calculate_rsi(df["Close"], period=7)
    df["rsi_14_scaled"] = df["rsi_14"] / 100.0

    # ATR & Volatility
    df["atr_14"] = calculate_atr(df["High"], df["Low"], df["Close"], period=14)
    df["atr_14_normalized"] = df["atr_14"] / df["Close"]
    df["volatility_5d"] = df["ret_1d"].rolling(5).std()
    df["volatility_10d"] = df["ret_1d"].rolling(10).std()
    df["volatility_20d"] = df["ret_1d"].rolling(20).std()
    df["volatility_60d"] = df["ret_1d"].rolling(60).std()
    df["volatility_ratio_5d_20d"] = df["volatility_5d"] / df["volatility_20d"].replace(0.0, np.nan)
    df["annualized_volatility_20d"] = df["volatility_20d"] * np.sqrt(252)
    df["annualized_volatility_60d"] = df["volatility_60d"] * np.sqrt(252)

    # Intraday spreads
    df["high_low_spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["close_open_spread"] = (df["Close"] - df["Open"]) / df["Open"]

    # Volume dynamics
    df["volume_change"] = df["Volume"].pct_change(1)
    df["volume_ma_5"] = df["Volume"].rolling(5).mean()
    df["volume_ma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio_5d"] = df["Volume"] / df["volume_ma_5"].replace(0.0, np.nan)
    df["volume_ratio_20d"] = df["Volume"] / df["volume_ma_20"].replace(0.0, np.nan)
    df["volume_trend_5d"] = (df["volume_ma_5"] / df["volume_ma_20"].replace(0.0, np.nan)) - 1.0

    df["dollar_volume"] = df["Close"] * df["Volume"]
    df["dollar_volume_ma_20"] = df["dollar_volume"].rolling(20).mean()
    df["dollar_volume_ratio_20d"] = df["dollar_volume"] / df["dollar_volume_ma_20"].replace(0.0, np.nan)

    signed_volume = np.sign(df["Close"].diff()).fillna(0.0) * df["Volume"]
    df["obv"] = signed_volume.cumsum()
    df["obv_change_5d"] = df["obv"].pct_change(5)

    return df[FEATURE_COLUMNS].dropna().copy()


def make_features(price_df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> FeatureSet:
    """
    Build tabular features from OHLCV data.
    Target is forward-return direction over `horizon` trading days:
    1 if forward return is positive, else 0.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    df = validate_ohlcv_dataframe(price_df)
    feature_frame = build_feature_frame(df)

    # Forward returns
    future_close = df["Close"].shift(-horizon)
    df["target_return"] = (future_close / df["Close"]) - 1.0
    df["target"] = (df["target_return"] > 0.0).astype(int)

    target_frame = df[["target", "target_return"]]
    model_df = feature_frame.join(target_frame, how="inner").dropna().copy()

    X = model_df[FEATURE_COLUMNS]
    y = model_df["target"].astype(int)
    return FeatureSet(X=X, y=y)

