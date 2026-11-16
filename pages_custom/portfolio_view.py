"""Modern Portfolio View for Quant Terminal."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import plot_correlation_heatmap, style_figure
from components.metric_cards import render_kpi_card
from core.data_loader import load_market_data
from core.portfolio_engine import build_portfolio_analytics
from utils.formatting import format_currency, format_number, format_pct, tone_class

DEFAULT_PORTFOLIO_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]


def render_portfolio_view(data_mode: str = "demo") -> None:
    """Clean multi-asset portfolio analytics with allocation donut and risk budgeting."""
    st.markdown('<div class="view-title">PORTFOLIO ANALYTICS</div>', unsafe_allow_html=True)

    with st.expander("Portfolio Composition & Weights", expanded=False):
        c1, c2 = st.columns([2, 1])
        with c1:
            tickers_input = st.multiselect(
                "Assets",
                options=[
                    "RELIANCE.NS",
                    "TCS.NS",
                    "INFY.NS",
                    "HDFCBANK.NS",
                    "ICICIBANK.NS",
                    "TATAMOTORS.NS",
                    "AAPL",
                    "MSFT",
                    "NVDA",
                    "GOOGL",
                ],
                default=DEFAULT_PORTFOLIO_TICKERS,
            )
        with c2:
            port_capital = st.number_input("Capital", min_value=10_000, value=500_000, step=50_000)

        if not tickers_input or len(tickers_input) < 2:
            st.warning("Select at least 2 assets to evaluate a portfolio.")
            return

        equal_weight = 100.0 / len(tickers_input)
        weight_cols = st.columns(len(tickers_input))
        custom_weights = {}
        for col, t in zip(weight_cols, tickers_input):
            with col:
                w_val = st.number_input(f"{t} %", min_value=1.0, max_value=100.0, value=round(equal_weight, 1), step=1.0)
                custom_weights[t] = w_val / 100.0

    # Process Portfolio Data
    price_dict = {}
    for t in (tickers_input if 'tickers_input' in locals() and tickers_input else DEFAULT_PORTFOLIO_TICKERS):
        df, _ = load_market_data(t, mode=data_mode)
        if not df.empty:
            price_dict[t] = df["Close"]

    if len(price_dict) < 2:
        st.error("Insufficient asset history available.")
        return

    weights = custom_weights if 'custom_weights' in locals() and custom_weights else None
    capital = float(port_capital) if 'port_capital' in locals() else 500_000.0

    port_res = build_portfolio_analytics(
        asset_prices=price_dict,
        weights=weights,
        initial_capital=capital,
    )

    m = port_res.metrics
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        p_ret = m.get("portfolio_total_return", 0.0)
        render_kpi_card("PORTFOLIO RETURN", format_pct(p_ret), delta_tone=tone_class(p_ret))
    with k2:
        render_kpi_card("CAGR", format_pct(m.get("portfolio_cagr", 0.0)), delta_tone=tone_class(m.get("portfolio_cagr", 0.0)))
    with k3:
        render_kpi_card("VOLATILITY", f"{m.get('portfolio_volatility', 0.0)*100.0:.1f}%")
    with k4:
        render_kpi_card("SHARPE", format_number(m.get("portfolio_sharpe_ratio", 0.0)))

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    col_chart, col_alloc = st.columns([1.5, 1.0])

    with col_chart:
        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=port_res.portfolio_equity.index,
                y=port_res.portfolio_equity,
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#16C784", width=2.0),
            )
        )
        fig_eq = style_figure(fig_eq, title="Portfolio Growth", height=380)
        st.plotly_chart(fig_eq, use_container_width=True)

    with col_alloc:
        donut_fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(port_res.weights.keys()),
                    values=list(port_res.weights.values()),
                    hole=0.6,
                    textinfo="label+percent",
                    marker=dict(colors=["#22D3EE", "#16C784", "#F59E0B", "#8B5CF6", "#EC4899"]),
                )
            ]
        )
        donut_fig = style_figure(donut_fig, title="Asset Allocation", height=380)
        st.plotly_chart(donut_fig, use_container_width=True)

    # Risk Contribution Bar Chart & Correlation
    col_rc, col_corr = st.columns([1.2, 1.0])

    with col_rc:
        st.markdown('<div class="card-title">HOLDINGS & RISK ATTRIBUTION</div>', unsafe_allow_html=True)
        st.dataframe(
            port_res.risk_contributions[["Ticker", "Weight", "Annualized_Vol", "Pct_Risk_Contribution"]].style.format({
                "Weight": "{:.1%}",
                "Annualized_Vol": "{:.1%}",
                "Pct_Risk_Contribution": "{:.1%}",
            }),
            use_container_width=True,
            hide_index=True,
            height=200,
        )

    with col_corr:
        corr_fig = plot_correlation_heatmap(port_res.correlation_matrix)
        st.plotly_chart(corr_fig, use_container_width=True)
