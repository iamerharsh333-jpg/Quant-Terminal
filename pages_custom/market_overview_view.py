"""Market Overview View for Quant Terminal."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import plot_correlation_heatmap, style_figure
from components.metric_cards import render_kpi_card
from config.settings import BENCHMARK_INDICES
from core.data_loader import load_market_data
from utils.formatting import format_pct, tone_class


def render_market_overview_view(data_mode: str = "demo") -> None:
    """Clean, information-dense market overview without text clutter."""
    st.markdown('<div class="view-title">MARKET OVERVIEW</div>', unsafe_allow_html=True)

    index_data = {}
    cols = st.columns(len(BENCHMARK_INDICES))

    for col, (name, symbol) in zip(cols, BENCHMARK_INDICES.items()):
        df, _ = load_market_data(symbol, mode=data_mode)
        index_data[name] = df
        with col:
            if not df.empty:
                last_val = df["Close"].iloc[-1]
                daily_chg = df["Close"].pct_change().iloc[-1]
                sym_c = "₹" if "NSE" in symbol or "BSE" in symbol else "$"
                render_kpi_card(
                    label=name,
                    value=f"{sym_c}{last_val:,.0f}",
                    delta=format_pct(daily_chg),
                    delta_tone=tone_class(daily_chg),
                    tooltip=f"Benchmark symbol: {symbol}",
                )

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    # Relative strength chart
    st.markdown('<div class="card-title">NORMALIZED BENCHMARK TRAJECTORY (BASE 100)</div>', unsafe_allow_html=True)
    norm_fig = go.Figure()
    colors = ["#22D3EE", "#16C784", "#F59E0B", "#8B5CF6", "#EC4899"]

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

    norm_fig = style_figure(norm_fig, title="", height=380)
    st.plotly_chart(norm_fig, use_container_width=True)

    col_corr, col_stats = st.columns([1.1, 1.0])

    with col_corr:
        ret_dict = {
            name: df["Close"].pct_change()
            for name, df in index_data.items()
            if not df.empty
        }
        ret_df = pd.DataFrame(ret_dict).dropna()
        if not ret_df.empty:
            corr_fig = plot_correlation_heatmap(ret_df.corr())
            st.plotly_chart(corr_fig, use_container_width=True)

    with col_stats:
        st.markdown('<div class="card-title">BENCHMARK STATISTICS</div>', unsafe_allow_html=True)
        stat_rows = []
        for name, df in index_data.items():
            if not df.empty and len(df) > 60:
                ret = df["Close"].pct_change().dropna()
                ann_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0]) ** (252.0 / len(df)) - 1.0
                ann_vol = ret.std() * (252**0.5)
                sharpe = (ret.mean() * 252 - 0.06) / (ann_vol + 1e-9)
                cum_ret = (1.0 + ret).cumprod()
                max_dd = abs(((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min())

                stat_rows.append({
                    "Index": name,
                    "CAGR": ann_ret,
                    "Vol": ann_vol,
                    "Sharpe": sharpe,
                    "Max DD": max_dd,
                })

        if stat_rows:
            stats_table = pd.DataFrame(stat_rows)
            st.dataframe(
                stats_table.style.format({
                    "CAGR": "{:+.1%}",
                    "Vol": "{:.1%}",
                    "Sharpe": "{:.2f}",
                    "Max DD": "{:.1%}",
                }),
                use_container_width=True,
                hide_index=True,
                height=260,
            )
