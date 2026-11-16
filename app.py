"""QUANT TERMINAL - Modern Institutional Quantitative Trading & Analytics Platform."""

from __future__ import annotations

import logging
from datetime import datetime
import pandas as pd
import streamlit as st

# Custom Components & Engines
from components.metric_cards import render_dashboard_top_kpis, render_kpi_card
from components.sidebar import render_sidebar
from components.topbar import render_topbar
from config.settings import APP_CONFIG, BENCHMARK_INDICES, TRADING_CONFIG
from core.backtesting_engine import BacktestResult, run_backtest
from core.data_loader import load_market_data
from core.feature_engineering import build_feature_frame, make_features
from core.model_engine import (
    evaluate_classifier,
    get_logistic_regression_coefficients,
    get_random_forest_feature_importance,
    is_xgboost_available,
    predict,
    predict_proba,
    predict_signals,
    time_series_train_test_split,
    train_gradient_boosting_model,
    train_logistic_regression_model,
    train_model,
    train_xgboost_model,
)
from pages_custom import (
    render_backtesting_view,
    render_dashboard_view,
    render_market_overview_view,
    render_model_comparison_view,
    render_portfolio_view,
    render_predictions_view,
    render_risk_view,
    render_settings_view,
    render_stock_analysis_view,
    render_watchlist_view,
)
from utils.recommendations import build_recommendation

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="QUANT TERMINAL",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_fintech_theme() -> None:
    """Inject institutional dark theme matching Bloomberg, TradingView, and Linear."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg-base: #0B0E14;
            --bg-sidebar: #10141D;
            --bg-card: #151B26;
            --border: #252D3A;
            --text-main: #E6EAF0;
            --text-secondary: #8792A5;
            --positive: #16C784;
            --negative: #EA3943;
            --accent: #3861FB;
            --cyan: #22D3EE;
            --amber: #F59E0B;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            font-size: 13.5px;
        }

        .stApp {
            background-color: var(--bg-base);
            color: var(--text-main);
        }

        .block-container {
            padding-top: 0.75rem;
            padding-bottom: 2.5rem;
            max-width: 1540px;
        }

        /* Top Header */
        .terminal-topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.45rem 0.9rem;
            margin-bottom: 0.85rem;
        }

        .topbar-left, .topbar-right {
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }

        .terminal-logo {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 800;
            font-size: 0.95rem;
            letter-spacing: 0.06em;
            color: var(--text-main);
        }

        .terminal-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-secondary);
            background: rgba(135, 146, 165, 0.12);
            padding: 1px 5px;
            border-radius: 3px;
        }

        .ticker-pill {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: var(--accent);
            background: rgba(56, 97, 251, 0.12);
            border: 1px solid rgba(56, 97, 251, 0.28);
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 0.8rem;
        }

        .market-pill {
            font-size: 0.75rem;
            color: var(--text-secondary);
            background: rgba(135, 146, 165, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
        }

        .time-pill {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: var(--text-secondary);
        }

        .terminal-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 3px;
        }

        .badge-live {
            background: rgba(22, 199, 132, 0.12);
            color: var(--positive);
            border: 1px solid rgba(22, 199, 132, 0.3);
        }

        .badge-demo {
            background: rgba(245, 158, 11, 0.12);
            color: var(--amber);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        /* Sidebar Customization */
        [data-testid="stSidebar"] {
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border);
        }

        .sidebar-brand-box {
            padding: 0.3rem 0 0.6rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 0.7rem;
        }

        .brand-title {
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 800;
            font-size: 1.05rem;
            letter-spacing: 0.05em;
            color: var(--text-main);
        }

        .brand-sub {
            display: block;
            font-size: 0.7rem;
            color: var(--text-secondary);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .sidebar-divider {
            height: 1px;
            background-color: var(--border);
            margin: 0.8rem 0;
        }

        .sidebar-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            font-weight: 700;
            color: var(--text-secondary);
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
        }

        /* Generic Terminal Cards */
        .terminal-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.75rem 0.9rem;
        }

        .kpi-card {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .kpi-header {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            font-weight: 700;
            color: var(--text-secondary);
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .kpi-row {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            margin-top: 0.25rem;
        }

        .kpi-value {
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.01em;
            font-family: 'JetBrains Mono', monospace;
        }

        .kpi-delta {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .positive { color: var(--positive) !important; }
        .negative { color: var(--negative) !important; }
        .neutral  { color: var(--text-secondary) !important; }

        .view-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.01em;
            margin-bottom: 0.6rem;
            font-family: 'JetBrains Mono', monospace;
        }

        .card-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: var(--text-secondary);
            margin-bottom: 0.4rem;
        }

        .spacer-sm {
            height: 0.75rem;
        }

        .terminal-disclaimer {
            background: rgba(245, 158, 11, 0.06);
            border: 1px solid rgba(245, 158, 11, 0.2);
            border-radius: 4px;
            padding: 0.4rem 0.75rem;
            font-size: 0.74rem;
            color: var(--amber);
        }

        /* Streamlit Element Overrides */
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 4px;
        }

        .stButton > button {
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            border: 1px solid var(--border);
        }

        button[data-baseweb="tab"] {
            font-size: 0.82rem;
            font-family: 'JetBrains Mono', monospace;
            padding: 6px 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def execute_quant_pipeline(
    symbol: str,
    horizon: int,
    buy_threshold: float,
    sell_threshold: float,
    transaction_cost: float,
    slippage: float,
    execution_lag: int,
    short_on_sell: bool,
    initial_capital: float,
    data_mode: str,
) -> dict[str, object]:
    """Execute complete ML modeling, backtesting, and diagnostic pipeline."""
    prices_df, provider = load_market_data(symbol, mode=data_mode)
    feature_set = make_features(prices_df, horizon=horizon)
    latest_feature_frame = build_feature_frame(prices_df)

    if feature_set.X.empty or len(feature_set.X) < 40:
        raise ValueError(f"Insufficient historical data for {symbol} to run ML pipeline.")

    split = time_series_train_test_split(feature_set.X, feature_set.y, test_size=0.20)

    # Train 4 models
    rf_model = train_model(split.X_train, split.y_train)
    lr_model = train_logistic_regression_model(split.X_train, split.y_train)
    xgb_model = train_xgboost_model(split.X_train, split.y_train) if is_xgboost_available() else None
    gb_model = train_gradient_boosting_model(split.X_train, split.y_train)

    models_dict = {
        "Random Forest": rf_model,
        "Logistic Regression": lr_model,
        "Gradient Boosting": gb_model,
    }
    if xgb_model is not None:
        models_dict["XGBoost"] = xgb_model

    # Market returns on test set
    market_returns = prices_df["Close"].pct_change(1).shift(-1).reindex(split.X_test.index).fillna(0.0)

    model_evaluations = {}
    backtests_dict = {}
    comparison_rows = []

    for name, model_obj in models_dict.items():
        preds = predict(model_obj, split.X_test)
        probs = predict_proba(model_obj, split.X_test)
        signals = predict_signals(
            model_obj,
            split.X_test,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )

        class_metrics = evaluate_classifier(split.y_test, preds, y_prob=probs)
        bk_res = run_backtest(
            market_returns=market_returns,
            predictions=signals,
            transaction_cost=transaction_cost,
            slippage=slippage,
            position_size=1.0,
            execution_lag=execution_lag,
            short_on_sell=short_on_sell,
            initial_capital=initial_capital,
        )

        model_evaluations[name] = {
            "metrics": class_metrics,
            "predictions": preds,
            "probabilities": probs,
            "signals": signals,
        }
        backtests_dict[name] = bk_res

        comparison_rows.append({
            "model": name,
            "accuracy": class_metrics["accuracy"],
            "precision": class_metrics["precision"],
            "recall": class_metrics["recall"],
            "f1": class_metrics["f1"],
            "strategy_total_return": bk_res.metrics.get("strategy_total_return", 0.0),
            "strategy_sharpe_ratio": bk_res.metrics.get("strategy_sharpe_ratio", 0.0),
            "strategy_max_drawdown": bk_res.metrics.get("strategy_max_drawdown", 0.0),
            "strategy_annualized_return": bk_res.metrics.get("strategy_annualized_return", 0.0),
            "buy_hold_total_return": bk_res.metrics.get("buy_and_hold_total_return", 0.0),
            "total_trades": bk_res.metrics.get("total_trades", 0.0),
            "win_rate": bk_res.metrics.get("win_rate", 0.0),
            "percent_time_in_market": bk_res.metrics.get("percent_time_in_market", 0.0),
        })

    comparison_frame = pd.DataFrame(comparison_rows)
    rf_importance = get_random_forest_feature_importance(rf_model, split.X_train.columns)
    lr_coefficients = get_logistic_regression_coefficients(lr_model, split.X_train.columns)

    return {
        "prices_df": prices_df,
        "provider": provider,
        "split": split,
        "models": models_dict,
        "evaluations": model_evaluations,
        "backtests": backtests_dict,
        "comparison_frame": comparison_frame,
        "rf_importance": rf_importance,
        "lr_coefficients": lr_coefficients,
        "latest_features": latest_feature_frame.tail(1),
    }


def main() -> None:
    inject_fintech_theme()

    if "data_provider_mode" not in st.session_state:
        st.session_state["data_provider_mode"] = "demo"

    controls = render_sidebar()

    active_view = controls["active_view"]
    ticker = controls["ticker_symbol"]
    market_code = controls["market_code"]
    selected_model = controls["selected_model"]
    horizon = controls["horizon"]
    buy_thresh = controls["buy_threshold"]
    sell_thresh = controls["sell_threshold"]
    t_cost = controls["transaction_cost"]
    slippage = controls["slippage"]
    exec_lag = controls["execution_lag"]
    short_sell = controls["short_on_sell"]
    capital = controls["initial_capital"]
    data_mode = st.session_state["data_provider_mode"]

    cache_key = f"{ticker}_{horizon}_{buy_thresh}_{sell_thresh}_{data_mode}"

    if "pipeline_results" not in st.session_state or controls["run_strategy"] or st.session_state.get("last_cache_key") != cache_key:
        with st.spinner("Processing Strategy..."):
            try:
                results = execute_quant_pipeline(
                    symbol=ticker,
                    horizon=horizon,
                    buy_threshold=buy_thresh,
                    sell_threshold=sell_thresh,
                    transaction_cost=t_cost,
                    slippage=slippage,
                    execution_lag=exec_lag,
                    short_on_sell=short_sell,
                    initial_capital=capital,
                    data_mode=data_mode,
                )
                st.session_state["pipeline_results"] = results
                st.session_state["last_cache_key"] = cache_key
            except Exception as exc:
                st.error(f"Quant pipeline execution error: {exc}")
                logger.exception("Pipeline failed")
                return

    pipeline = st.session_state.get("pipeline_results")
    if not pipeline:
        st.warning("Execute strategy to view terminal analytics.")
        return

    prices_df = pipeline["prices_df"]
    provider = pipeline["provider"]
    latest_features = pipeline["latest_features"]
    models = pipeline["models"]
    backtests = pipeline["backtests"]
    comparison_frame = pipeline["comparison_frame"]
    rf_importance = pipeline["rf_importance"]

    active_model_obj = models.get(selected_model, models["Random Forest"])
    active_backtest = backtests.get(selected_model, backtests["Random Forest"])
    active_eval = pipeline["evaluations"].get(selected_model, pipeline["evaluations"]["Random Forest"])

    latest_prob = float(predict_proba(active_model_obj, latest_features).iloc[0])
    latest_sig = str(
        predict_signals(
            active_model_obj,
            latest_features,
            buy_threshold=buy_thresh,
            sell_threshold=sell_thresh,
        ).iloc[0]
    )

    if latest_sig == "BUY":
        confidence = (latest_prob - buy_thresh) / (1.0 - buy_thresh + 1e-9)
    elif latest_sig == "SELL":
        confidence = (sell_thresh - latest_prob) / (sell_thresh + 1e-9)
    else:
        confidence = 0.50
    confidence = float(max(0.0, min(1.0, confidence)))

    recommendation = build_recommendation(
        latest_features,
        signal=latest_sig,
        probability_up=latest_prob,
        confidence=confidence,
        buy_threshold=buy_thresh,
        sell_threshold=sell_thresh,
    )

    preds_table = pd.concat(
        [
            pipeline["split"].y_test.rename("target"),
            active_eval["predictions"].rename("pred"),
            active_eval["probabilities"].rename("prob"),
            active_eval["signals"].rename("signal"),
        ],
        axis=1,
    )

    # 1. Render Topbar
    render_topbar(
        selected_ticker=ticker,
        market_name=market_code,
        data_provider_name=provider.provider_name,
        is_synthetic=provider.is_synthetic,
    )

    # 2. View Routing
    if "Overview" in active_view:
        render_dashboard_view(
            selected_ticker=ticker,
            market_code=market_code,
            prices_df=prices_df,
            features_df=latest_features,
            latest_signal=latest_sig,
            latest_prob=latest_prob,
            confidence=confidence,
            recommendation=recommendation,
            backtest_result=active_backtest,
            comparison_frame=comparison_frame,
            data_mode=data_mode,
        )
    elif "Markets" in active_view:
        render_market_overview_view(data_mode=data_mode)
    elif "Stock Analysis" in active_view:
        render_stock_analysis_view(selected_ticker=ticker, prices_df=prices_df)
    elif "Predictions" in active_view:
        render_predictions_view(
            selected_ticker=ticker,
            selected_model=selected_model,
            horizon=horizon,
            buy_threshold=buy_thresh,
            sell_threshold=sell_thresh,
            latest_signal=latest_sig,
            latest_prob=latest_prob,
            confidence=confidence,
            classification_metrics=active_eval["metrics"],
            feature_importance_df=rf_importance,
            predictions_table=preds_table,
        )
    elif "Model Comparison" in active_view:
        render_model_comparison_view(
            selected_ticker=ticker,
            comparison_frame=comparison_frame,
            all_backtests=backtests,
        )
    elif "Backtesting" in active_view:
        render_backtesting_view(
            selected_ticker=ticker,
            selected_model=selected_model,
            backtest_result=active_backtest,
            initial_capital=capital,
        )
    elif "Portfolio" in active_view:
        render_portfolio_view(data_mode=data_mode)
    elif "Risk Analytics" in active_view:
        render_risk_view(selected_ticker=ticker, prices_df=prices_df, data_mode=data_mode)
    elif "Watchlist" in active_view:
        render_watchlist_view(data_mode=data_mode)
    elif "Settings" in active_view:
        render_settings_view()


if __name__ == "__main__":
    main()
