"""QUANT TERMINAL - Institutional Quantitative Research & Stock Analytics Platform."""

from __future__ import annotations

import logging
from datetime import datetime
import pandas as pd
import streamlit as st

# Custom modules
from components.metric_cards import render_dashboard_kpis, render_kpi_card
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
    page_title="QUANT TERMINAL | Institutional Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_terminal_css() -> None:
    """Inject institutional dark terminal styling inspired by Bloomberg and TradingView."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-base: #060911;
            --bg-card: rgba(13, 20, 36, 0.88);
            --border-subtle: rgba(148, 163, 184, 0.14);
            --border-accent: rgba(34, 211, 238, 0.35);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-dim: #64748b;
            --green: #10b981;
            --red: #ef4444;
            --cyan: #22d3ee;
            --amber: #f59e0b;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
        }

        .stApp {
            background: linear-gradient(180deg, #060911 0%, #0a0f1d 100%);
            color: var(--text-main);
        }

        .block-container {
            padding-top: 1.0rem;
            padding-bottom: 3.0rem;
            max-width: 1560px;
        }

        /* Topbar styling */
        .terminal-topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(15, 23, 42, 0.94);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 0.65rem 1.1rem;
            margin-bottom: 1.1rem;
            backdrop-filter: blur(12px);
        }

        .topbar-left, .topbar-right {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .terminal-logo {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 800;
            font-size: 1.05rem;
            letter-spacing: 0.08em;
            color: #38bdf8;
        }

        .topbar-divider {
            color: var(--text-dim);
        }

        .ticker-chip {
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: #38bdf8;
            padding: 0.15rem 0.55rem;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 0.85rem;
        }

        .market-chip {
            background: rgba(148, 163, 184, 0.1);
            color: var(--text-muted);
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .terminal-badge {
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
            letter-spacing: 0.04em;
            font-family: 'JetBrains Mono', monospace;
        }

        .badge-live {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }

        .badge-demo {
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }

        .provider-info, .timestamp-info {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-dim);
        }

        /* KPI Cards */
        .terminal-kpi-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
            transition: border-color 0.2s ease;
        }

        .terminal-kpi-card:hover {
            border-color: rgba(56, 189, 248, 0.35);
        }

        .kpi-header {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            font-weight: 700;
            color: var(--text-dim);
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }

        .kpi-main {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 0.4rem;
        }

        .kpi-value {
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--text-main);
            letter-spacing: -0.02em;
        }

        .kpi-delta {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            font-weight: 700;
        }

        .kpi-subtext {
            font-size: 0.72rem;
            color: var(--text-dim);
            margin-top: 0.25rem;
        }

        .positive { color: var(--green) !important; }
        .negative { color: var(--red) !important; }
        .neutral  { color: var(--text-muted) !important; }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: #090d16;
            border-right: 1px solid var(--border-subtle);
        }

        .sidebar-header {
            padding: 0.4rem 0 0.8rem 0;
        }

        .sidebar-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.15rem;
            font-weight: 800;
            color: #f8fafc;
            letter-spacing: 0.05em;
            margin-top: 0.3rem;
        }

        .sidebar-subtitle {
            font-size: 0.76rem;
            color: var(--text-dim);
        }

        .sidebar-section-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--cyan);
            letter-spacing: 0.08em;
            margin: 0.8rem 0 0.4rem 0;
        }

        .section-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: #e2e8f0;
            margin-bottom: 0.5rem;
        }

        .view-header {
            font-size: 1.45rem;
            font-weight: 800;
            color: var(--text-main);
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
        }

        .terminal-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 1.1rem;
        }

        .signal-banner {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
            font-weight: 800;
            padding: 0.5rem 0.8rem;
            border-radius: 6px;
            margin: 0.6rem 0;
            border: 1px solid currentColor;
            display: inline-block;
        }

        .terminal-list {
            padding-left: 1.1rem;
            margin-top: 0.4rem;
            font-size: 0.82rem;
            color: var(--text-muted);
            line-height: 1.5;
        }

        .terminal-disclaimer {
            background: rgba(245, 158, 11, 0.08);
            border: 1px solid rgba(245, 158, 11, 0.25);
            border-radius: 6px;
            padding: 0.65rem 0.95rem;
            font-size: 0.78rem;
            color: #fde68a;
            line-height: 1.4;
        }

        .card-copy {
            font-size: 0.85rem;
            color: var(--text-muted);
            line-height: 1.5;
            margin: 0.4rem 0;
        }

        /* Streamlit components refinement */
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
        }

        .stButton > button {
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            letter-spacing: 0.04em;
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

    # Train all 4 model families
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

        comparison_rows.append(
            {
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
            }
        )

    comparison_frame = pd.DataFrame(comparison_rows)

    # Feature Importance for Random Forest
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
    inject_terminal_css()

    # Data mode state
    if "data_provider_mode" not in st.session_state:
        st.session_state["data_provider_mode"] = "demo"

    # Sidebar inputs
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

    # Auto-run pipeline or run on click
    if "pipeline_results" not in st.session_state or controls["run_strategy"] or st.session_state.get("last_cache_key") != cache_key:
        with st.spinner(f"Quant Engine processing {ticker} ({selected_model})..."):
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
        st.warning("Please execute strategy to view terminal analytics.")
        return

    # Extract pipeline context
    prices_df = pipeline["prices_df"]
    provider = pipeline["provider"]
    latest_features = pipeline["latest_features"]
    models = pipeline["models"]
    backtests = pipeline["backtests"]
    comparison_frame = pipeline["comparison_frame"]
    rf_importance = pipeline["rf_importance"]

    # Active model state
    active_model_obj = models.get(selected_model, models["Random Forest"])
    active_backtest = backtests.get(selected_model, backtests["Random Forest"])
    active_eval = pipeline["evaluations"].get(selected_model, pipeline["evaluations"]["Random Forest"])

    # Latest prediction for active model
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

    # Predictions ledger table
    preds_table = pd.concat(
        [
            pipeline["split"].y_test.rename("actual_target"),
            active_eval["predictions"],
            active_eval["probabilities"],
            active_eval["signals"],
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

    # 2. View Routing Across the 10 Sections
    if "1. Dashboard" in active_view:
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
    elif "2. Market Overview" in active_view:
        render_market_overview_view(data_mode=data_mode)
    elif "3. Stock Analysis" in active_view:
        render_stock_analysis_view(selected_ticker=ticker, prices_df=prices_df)
    elif "4. ML Predictions" in active_view:
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
    elif "5. Model Comparison" in active_view:
        render_model_comparison_view(
            selected_ticker=ticker,
            comparison_frame=comparison_frame,
            all_backtests=backtests,
        )
    elif "6. Strategy Backtesting" in active_view:
        render_backtesting_view(
            selected_ticker=ticker,
            selected_model=selected_model,
            backtest_result=active_backtest,
            initial_capital=capital,
        )
    elif "7. Portfolio Analytics" in active_view:
        render_portfolio_view(data_mode=data_mode)
    elif "8. Risk Analytics" in active_view:
        render_risk_view(selected_ticker=ticker, prices_df=prices_df, data_mode=data_mode)
    elif "9. Watchlist" in active_view:
        render_watchlist_view(data_mode=data_mode)
    elif "10. About / Settings" in active_view:
        render_settings_view()


if __name__ == "__main__":
    main()
