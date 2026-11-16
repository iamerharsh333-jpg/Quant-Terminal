"""Clean collapsible sidebar navigation and strategy controls."""

from __future__ import annotations

import streamlit as st
from config.settings import INDIAN_EQUITIES, US_EQUITIES
from core.model_engine import is_xgboost_available

VIEW_OPTIONS = [
    "📊 Overview",
    "🌐 Markets",
    "📈 Stock Analysis",
    "🔮 Predictions",
    "⚖️ Model Comparison",
    "🧪 Backtesting",
    "💼 Portfolio",
    "🛡️ Risk Analytics",
    "👁️ Watchlist",
    "⚙️ Settings",
]


def render_sidebar() -> dict[str, object]:
    """Render clean, compact sidebar navigation and parameter controls."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand-box">
                <span class="brand-title">QUANT TERMINAL</span>
                <span class="brand-sub">RESEARCH TERMINAL</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        active_view = st.radio(
            "Navigation",
            options=VIEW_OPTIONS,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">INSTRUMENT SELECTION</div>', unsafe_allow_html=True)

        market_choice = st.selectbox(
            "Market",
            options=["NSE (India)", "US Equities"],
            index=0,
            label_visibility="collapsed",
        )

        if "NSE" in market_choice:
            ticker_map = INDIAN_EQUITIES
            market_code = "NSE"
            default_ticker = "RELIANCE.NS"
        else:
            ticker_map = US_EQUITIES
            market_code = "US"
            default_ticker = "AAPL"

        ticker_options = list(ticker_map.keys()) + ["Custom..."]
        selected_option = st.selectbox(
            "Ticker",
            options=ticker_options,
            index=0,
            label_visibility="collapsed",
        )

        if selected_option == "Custom...":
            ticker_symbol = st.text_input("Symbol", value=default_ticker).strip().upper()
        else:
            ticker_symbol = ticker_map[selected_option]

        st.caption(f"Active: `{ticker_symbol}`")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">MODEL PARAMETERS</div>', unsafe_allow_html=True)

        available_models = ["Random Forest", "Logistic Regression", "Gradient Boosting"]
        if is_xgboost_available():
            available_models.insert(1, "XGBoost")

        selected_model = st.selectbox("Model", options=available_models, index=0)

        c_h1, c_h2 = st.columns(2)
        with c_h1:
            horizon = st.selectbox("Horizon", options=[1, 3, 5, 10], index=1, format_func=lambda d: f"{d}d")
        with c_h2:
            threshold = st.number_input("Threshold", min_value=0.51, max_value=0.85, value=0.60, step=0.01)

        buy_threshold = float(threshold)
        sell_threshold = float(1.0 - threshold)

        with st.expander("Execution Settings", expanded=False):
            t_cost_bps = st.number_input("Cost (bps)", min_value=0, max_value=100, value=10)
            slip_bps = st.number_input("Slippage (bps)", min_value=0, max_value=50, value=5)
            exec_lag = st.selectbox("Lag", options=[1, 2], index=0)
            short_on_sell = st.checkbox("Allow Shorting", value=False)
            initial_cap = st.number_input("Capital ($/₹)", min_value=10_000, value=100_000, step=10_000)

        run_btn = st.button("RUN STRATEGY", type="primary", use_container_width=True)

    return {
        "active_view": active_view,
        "market_choice": market_choice,
        "market_code": market_code,
        "ticker_symbol": ticker_symbol,
        "selected_model": selected_model,
        "horizon": int(horizon),
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "transaction_cost": float(t_cost_bps) / 10000.0,
        "slippage": float(slip_bps) / 10000.0,
        "execution_lag": int(exec_lag),
        "short_on_sell": short_on_sell,
        "initial_capital": float(initial_cap),
        "run_strategy": run_btn,
    }
