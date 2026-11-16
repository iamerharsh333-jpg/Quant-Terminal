"""End-to-end integration tests for Quant Terminal."""

from __future__ import annotations

import pandas as pd
import pytest
from core.data_loader import load_market_data
from core.feature_engineering import make_features
from core.model_engine import (
    predict_signals,
    time_series_train_test_split,
    train_model,
)
from core.backtesting_engine import run_backtest
from utils.recommendations import build_recommendation


def test_full_offline_quant_pipeline():
    # 1. Load data via Demo Mode (guaranteed offline)
    prices_df, provider = load_market_data("RELIANCE.NS", mode="demo")
    assert not prices_df.empty
    assert provider.is_synthetic is True

    # 2. Engineer features
    f_set = make_features(prices_df, horizon=3)
    assert len(f_set.X) > 100

    # 3. Train/test split
    split = time_series_train_test_split(f_set.X, f_set.y, test_size=0.20)
    assert len(split.X_train) > 0
    assert len(split.X_test) > 0

    # 4. Train Model
    rf = train_model(split.X_train, split.y_train, n_estimators=40, max_depth=4)

    # 5. Generate signals
    signals = predict_signals(rf, split.X_test, buy_threshold=0.60, sell_threshold=0.40)
    assert len(signals) == len(split.X_test)

    # 6. Backtest
    mkt_returns = prices_df["Close"].pct_change().shift(-1).reindex(split.X_test.index).fillna(0.0)
    bk = run_backtest(
        market_returns=mkt_returns,
        predictions=signals,
        transaction_cost=0.001,
        slippage=0.0005,
        execution_lag=1,
    )
    assert not bk.frame.empty
    assert "strategy_total_return" in bk.metrics

    # 7. Recommendation
    latest_feat = f_set.X.tail(1)
    rec = build_recommendation(
        latest_feat,
        signal=str(signals.iloc[-1]),
        probability_up=0.65,
        confidence=0.75,
    )
    assert hasattr(rec, "action")
    assert hasattr(rec, "summary")
    assert len(rec.reasons) > 0

