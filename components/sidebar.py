"""Persistent institutional sidebar navigation and strategy controls."""

from __future__ import annotations

import streamlit as st
from config.settings import INDIAN_EQUITIES, US_EQUITIES
from core.model_engine import is_xgboost_available

VIEW_OPTIONS = [
    "1. Dashboard",
    "2. Market Overview",
    "3. Stock Analysis",
    "4. ML Predictions",
    "5. Model Comparison",
    "6. Strategy Backtesting",
    "7. Portfolio Analytics",
    "8. Risk Analytics",
    "9. Watchlist",
    "10. About / Settings",
]


def render_sidebar() -> dict[str, object]:
    """
    Render persistent sidebar navigation, universe controls, and strategy hyperparams.
    Returns dictionary of selected configuration state.
    """
    with st.sidebar:
        # Branding
        st.markdown(
            """
            <div class="sidebar-header">
                <div class="terminal-badge badge-live">● TERMINAL v2.0</div>
                <div class="sidebar-title">QUANT TERMINAL</div>
                <div class="sidebar-subtitle">Hedge Fund Quantitative Research Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">NAVIGATION</div>', unsafe_allow_html=True)
        active_view = st.radio(
            "Navigation Views",
            options=VIEW_OPTIONS,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown('<div class="sidebar-section-title">MARKET & ASSET SELECTION</div>', unsafe_allow_html=True)

        market_choice = st.selectbox(
            "Market Universe",
            options=["Indian Equities (NSE)", "US Equities"],
            index=0,
        )

        if market_choice == "Indian Equities (NSE)":
            ticker_map = INDIAN_EQUITIES
            market_code = "NSE"
            default_ticker = "RELIANCE.NS"
        else:
            ticker_map = US_EQUITIES
            market_code = "US"
            default_ticker = "AAPL"

        ticker_options = list(ticker_map.keys()) + ["Custom Ticker..."]
        selected_option = st.selectbox("Instrument Selector", options=ticker_options, index=0)

        if selected_option == "Custom Ticker...":
            ticker_symbol = st.text_input("Enter Ticker Symbol", value=default_ticker).strip().upper()
        else:
            ticker_symbol = ticker_map[selected_option]

        st.caption(f"Active Symbol: `{ticker_symbol}` | Market: `{market_code}`")

        st.markdown("---")
        st.markdown('<div class="sidebar-section-title">QUANT STRATEGY CONTROLS</div>', unsafe_allow_html=True)

        # Model options
        available_models = ["Random Forest", "Logistic Regression", "Gradient Boosting"]
        if is_xgboost_available():
            available_models.insert(1, "XGBoost")

        selected_model = st.selectbox("Primary Model", options=available_models, index=0)

        horizon = st.selectbox(
            "Forecast Horizon",
            options=[1, 3, 5, 10],
            index=1,
            format_func=lambda d: f"{d} Trading Day{'s' if d != 1 else ''}",
        )

        threshold = st.slider(
            "Signal Probability Threshold",
            min_value=0.51,
            max_value=0.85,
            value=0.60,
            step=0.01,
            help="Generate BUY when probability > threshold; SELL when probability < (1 - threshold).",
        )
        buy_threshold = float(threshold)
        sell_threshold = float(1.0 - threshold)

        with st.expander("Execution & Cost Parameters", expanded=False):
            transaction_cost_bps = st.number_input("Transaction Cost (bps)", min_value=0, max_value=100, value=10)
            slippage_bps = st.number_input("Slippage (bps)", min_value=0, max_value=50, value=5)
            execution_lag = st.selectbox("Execution Lag", options=[1, 2], index=0, help="1-bar lag executes at next bar open/close.")
            short_on_sell = st.checkbox("Allow Shorting on SELL", value=False)
            initial_capital = st.number_input("Initial Capital ($ / ₹)", min_value=10_000, value=100_000, step=10_000)

        run_strategy_btn = st.button("RUN QUANT STRATEGY", type="primary", use_container_width=True)

        st.markdown("---")
        st.caption("Quant Terminal is for quantitative research and educational analysis. No automated brokerage execution.")

    return {
        "active_view": active_view,
        "market_choice": market_choice,
        "market_code": market_code,
        "ticker_symbol": ticker_symbol,
        "selected_model": selected_model,
        "horizon": int(horizon),
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "transaction_cost": float(transaction_cost_bps) / 10000.0,
        "slippage": float(slippage_bps) / 10000.0,
        "execution_lag": int(execution_lag),
        "short_on_sell": short_on_sell,
        "initial_capital": float(initial_capital),
        "run_strategy": run_strategy_btn,
    }

