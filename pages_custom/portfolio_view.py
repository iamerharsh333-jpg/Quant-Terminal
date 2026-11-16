"""Portfolio Analytics View for Quant Terminal."""

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
    """Render multi-asset portfolio constructor, risk attribution, and correlation matrix."""
    st.markdown('<div class="view-header">PORTFOLIO ANALYTICS & RISK DECOMPOSITION</div>', unsafe_allow_html=True)
    st.caption("Quantitative multi-asset portfolio optimization, risk budgeting, and asset contribution to risk.")

    # Portfolio Setup Controls
    with st.expander("PORTFOLIO ASSET COMPOSITION & WEIGHTS", expanded=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            tickers_input = st.multiselect(
                "Select Portfolio Assets",
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
            port_capital = st.number_input(
                "Portfolio Capital ($ / ₹)",
                min_value=10_000,
                value=500_000,
                step=50_000,
            )

        if not tickers_input or len(tickers_input) < 2:
            st.warning("Please select at least 2 assets to evaluate a portfolio.")
            return

        st.markdown("<b>Specify Target Allocations (%)</b>", unsafe_allow_html=True)
        equal_weight = 100.0 / len(tickers_input)
        weight_cols = st.columns(len(tickers_input))
        custom_weights = {}

        for col, t in zip(weight_cols, tickers_input):
            with col:
                w_val = st.number_input(f"{t} (%)", min_value=1.0, max_value=100.0, value=round(equal_weight, 1), step=1.0)
                custom_weights[t] = w_val / 100.0

    # Fetch Price Series for Portfolio
    with st.spinner("Fetching historical prices and computing portfolio dynamics..."):
        price_dict = {}
        for t in tickers_input:
            df, _ = load_market_data(t, mode=data_mode)
            if not df.empty:
                price_dict[t] = df["Close"]

        if len(price_dict) < 2:
            st.error("Could not fetch sufficient asset series for portfolio analytics.")
            return

        port_res = build_portfolio_analytics(
            asset_prices=price_dict,
            weights=custom_weights,
            initial_capital=float(port_capital),
        )

    # 1. Top Portfolio KPIs
    st.markdown('<div class="section-title">PORTFOLIO SUMMARY METRICS</div>', unsafe_allow_html=True)
    m = port_res.metrics
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        p_ret = m.get("portfolio_total_return", 0.0)
        render_kpi_card("PORTFOLIO RETURN", format_pct(p_ret), delta_tone=tone_class(p_ret), subtext="Cumulative Growth")
    with k2:
        cagr = m.get("portfolio_cagr", 0.0)
        render_kpi_card("ANNUALIZED (CAGR)", format_pct(cagr), delta_tone=tone_class(cagr), subtext="Compounded Annual")
    with k3:
        p_vol = m.get("portfolio_volatility", 0.0)
        render_kpi_card("ANNUALIZED VOL", f"{p_vol*100.0:.2f}%", subtext="Portfolio Std Dev")
    with k4:
        p_sh = m.get("portfolio_sharpe_ratio", 0.0)
        render_kpi_card("PORTFOLIO SHARPE", format_number(p_sh), delta_tone=tone_class(p_sh), subtext="Rf = 6.5%")
    with k5:
        p_mdd = m.get("portfolio_max_drawdown", 0.0)
        render_kpi_card("MAX DRAWDOWN", f"-{p_mdd*100.0:.2f}%", delta_tone="negative", subtext="Peak to Trough")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Charts: Equity Curve Left, Asset Allocation Donut Right
    col_chart, col_alloc = st.columns([1.6, 1.0])

    with col_chart:
        st.markdown('<div class="section-title">PORTFOLIO CUMULATIVE TRAJECTORY</div>', unsafe_allow_html=True)
        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=port_res.portfolio_equity.index,
                y=port_res.portfolio_equity,
                mode="lines",
                name="Total Portfolio Value",
                line=dict(color="#10b981", width=2.4),
            )
        )
        fig_eq = style_figure(fig_eq, title="Portfolio Growth Over Time ($ / ₹)", height=420)
        st.plotly_chart(fig_eq, use_container_width=True)

    with col_alloc:
        st.markdown('<div class="section-title">TARGET ALLOCATION WEIGHTS</div>', unsafe_allow_html=True)
        donut_fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(port_res.weights.keys()),
                    values=list(port_res.weights.values()),
                    hole=0.55,
                    textinfo="label+percent",
                    marker=dict(colors=["#38bdf8", "#10b981", "#f59e0b", "#ec4899", "#a855f7"]),
                )
            ]
        )
        donut_fig = style_figure(donut_fig, title="Asset Weights", height=420)
        st.plotly_chart(donut_fig, use_container_width=True)

    # 3. Risk Attribution: Component Contribution to Risk
    st.markdown('<div class="section-title">RISK BUDGETING & COMPONENT RISK CONTRIBUTION</div>', unsafe_allow_html=True)
    col_rc_table, col_rc_chart = st.columns([1.2, 1.0])

    with col_rc_table:
        rc_df = port_res.risk_contributions
        st.dataframe(
            rc_df.style.format(
                {
                    "Weight": "{:.1%}",
                    "Annualized_Vol": "{:.2%}",
                    "Marginal_Risk_Contrib": "{:.4f}",
                    "Component_Risk_Contrib": "{:.4f}",
                    "Pct_Risk_Contribution": "{:.1%}",
                }
            ),
            use_container_width=True,
        )

    with col_rc_chart:
        fig_rc = go.Figure(
            go.Bar(
                x=rc_df["Pct_Risk_Contribution"] * 100.0,
                y=rc_df["Ticker"],
                orientation="h",
                marker_color="#f59e0b",
            )
        )
        fig_rc = style_figure(fig_rc, title="Share of Total Portfolio Risk (%)", height=320)
        st.plotly_chart(fig_rc, use_container_width=True)

    # 4. Correlation Matrix Heatmap
    st.markdown('<div class="section-title">PORTFOLIO ASSET CORRELATION MATRIX</div>', unsafe_allow_html=True)
    corr_fig = plot_correlation_heatmap(port_res.correlation_matrix)
    st.plotly_chart(corr_fig, use_container_width=True)

    # 5. Model Assumptions Callout
    with st.expander("PORTFOLIO MATHEMATICAL ASSUMPTIONS", expanded=False):
        for assumption in port_res.assumptions:
            st.markdown(f"- {assumption}")

