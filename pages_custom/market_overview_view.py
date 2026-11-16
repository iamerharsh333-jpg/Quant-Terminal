"""Market Overview View for Quant Terminal."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import style_figure
from components.metric_cards import render_kpi_card
from config.settings import BENCHMARK_INDICES
from core.data_loader import load_market_data
from utils.formatting import format_pct, tone_class


def render_market_overview_view(data_mode: str = "demo") -> None:
    """Render broad market indices, heatmaps, and macro regime monitors."""
    st.markdown('<div class="view-header">MARKET OVERVIEW & MACRO REGIMES</div>', unsafe_allow_html=True)
    st.caption("Cross-market index tracking for Indian Equities (NSE/BSE) and US Indices.")

    index_data = {}
    cols = st.columns(len(BENCHMARK_INDICES))

    for col, (name, symbol) in zip(cols, BENCHMARK_INDICES.items()):
        df, _ = load_market_data(symbol, mode=data_mode)
        index_data[name] = df
        with col:
            if not df.empty:
                last_val = df["Close"].iloc[-1]
                daily_chg = df["Close"].pct_change().iloc[-1]
                vol_20d = df["Close"].pct_change().rolling(20).std().iloc[-1] * (252**0.5)
                sym_c = "₹" if "NSE" in symbol or "BSE" in symbol else "$"
                render_kpi_card(
                    label=name,
                    value=f"{sym_c}{last_val:,.0f}",
                    delta=format_pct(daily_chg),
                    delta_tone=tone_class(daily_chg),
                    subtext=f"20D Vol: {vol_20d*100.0:.1f}%",
                )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">NORMALIZED MULTI-INDEX CUMULATIVE PERFORMANCE (BASE 100)</div>', unsafe_allow_html=True)

    # Normalized comparison chart
    norm_fig = go.Figure()
    colors = ["#38bdf8", "#10b981", "#f59e0b", "#ec4899", "#a855f7"]

    for (name, df), color in zip(index_data.items(), colors):
        if not df.empty and len(df) > 30:
            norm_series = (df["Close"] / df["Close"].iloc[0]) * 100.0
            norm_fig.add_trace(
                go.Scatter(
                    x=norm_series.index,
                    y=norm_series,
                    mode="lines",
                    name=name,
                    line=dict(width=1.8, color=color),
                )
            )

    norm_fig = style_figure(norm_fig, title="Multi-Index Relative Strength", height=450)
    st.plotly_chart(norm_fig, use_container_width=True)

    # Volatility & Correlation Overview
    col_corr, col_stats = st.columns([1.2, 1.0])

    with col_corr:
        st.markdown('<div class="section-title">BENCHMARK CORRELATION MATRIX</div>', unsafe_allow_html=True)
        ret_dict = {
            name: df["Close"].pct_change()
            for name, df in index_data.items()
            if not df.empty
        }
        ret_df = pd.DataFrame(ret_dict).dropna()
        if not ret_df.empty:
            corr = ret_df.corr()
            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    colorscale="Viridis",
                    text=corr.values.round(2),
                    texttemplate="%{text}",
                )
            )
            fig_corr = style_figure(fig_corr, title="Cross-Market Correlations", height=380)
            st.plotly_chart(fig_corr, use_container_width=True)

    with col_stats:
        st.markdown('<div class="section-title">MARKET REGIME & VOLATILITY STATUS</div>', unsafe_allow_html=True)
        stat_rows = []
        for name, df in index_data.items():
            if not df.empty and len(df) > 60:
                ret = df["Close"].pct_change().dropna()
                ann_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0]) ** (252.0 / len(df)) - 1.0
                ann_vol = ret.std() * (252**0.5)
                sharpe = (ret.mean() * 252 - 0.06) / (ann_vol + 1e-9)
                cum_ret = (1.0 + ret).cumprod()
                max_dd = abs(((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min())

                stat_rows.append(
                    {
                        "Index": name,
                        "Annual Return": ann_ret,
                        "Annual Volatility": ann_vol,
                        "Sharpe Ratio": sharpe,
                        "Max Drawdown": max_dd,
                    }
                )

        if stat_rows:
            stats_table = pd.DataFrame(stat_rows)
            st.dataframe(
                stats_table.style.format(
                    {
                        "Annual Return": "{:+.2%}",
                        "Annual Volatility": "{:.2%}",
                        "Sharpe Ratio": "{:.2f}",
                        "Max Drawdown": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

