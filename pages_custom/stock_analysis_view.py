"""Stock Analysis View for Quant Terminal."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import plot_candlestick_with_indicators
from components.metric_cards import render_kpi_card
from core.feature_engineering import calculate_technical_indicators
from utils.formatting import format_currency, format_number, format_pct, tone_class


def render_stock_analysis_view(
    selected_ticker: str,
    prices_df: pd.DataFrame,
) -> None:
    """Dedicated interactive stock research interface with minimal clutter."""
    st.markdown(f'<div class="view-title">STOCK ANALYSIS — {selected_ticker}</div>', unsafe_allow_html=True)

    if prices_df is None or prices_df.empty:
        st.warning(f"No price data available for {selected_ticker}.")
        return

    tech_df = calculate_technical_indicators(prices_df)

    # Top compact stats row
    last_row = tech_df.iloc[-1]
    last_close = float(last_row["Close"])
    daily_ret = float(tech_df["Close"].pct_change().iloc[-1])
    currency_symbol = "₹" if ".NS" in selected_ticker else "$"
    rsi_val = float(last_row.get("RSI_14", 50.0))
    macd_val = float(last_row.get("MACD", 0.0))
    vol_20d = float(last_row.get("Rolling_Vol_20d", 0.0))
    atr_val = float(last_row.get("ATR_14", 0.0))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card("LATEST CLOSE", format_currency(last_close, symbol=currency_symbol), format_pct(daily_ret), tone_class(daily_ret))
    with c2:
        render_kpi_card("RSI (14)", f"{rsi_val:.1f}", delta_tone="negative" if rsi_val > 70 else ("positive" if rsi_val < 30 else "neutral"))
    with c3:
        render_kpi_card("MACD", format_number(macd_val), delta_tone=tone_class(macd_val))
    with c4:
        render_kpi_card("ATR (14)", format_currency(atr_val, symbol=currency_symbol))
    with c5:
        render_kpi_card("VOLATILITY (20D)", f"{vol_20d*100.0:.1f}%")

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    # Collapsible Indicator Controls
    with st.expander("Indicator Overlays & Subplots", expanded=False):
        c_i1, c_i2, c_i3, c_i4, c_i5, c_i6 = st.columns(6)
        with c_i1:
            show_sma_20 = st.checkbox("SMA 20", value=True)
        with c_i2:
            show_sma_50 = st.checkbox("SMA 50", value=True)
        with c_i3:
            show_sma_200 = st.checkbox("SMA 200", value=False)
        with c_i4:
            show_bb = st.checkbox("Bollinger Bands", value=True)
        with c_i5:
            show_rsi = st.checkbox("RSI Subplot", value=True)
        with c_i6:
            show_macd = st.checkbox("MACD Subplot", value=True)

    # Interactive Chart
    fig = plot_candlestick_with_indicators(
        df=tech_df,
        ticker=selected_ticker,
        show_sma_20=show_sma_20,
        show_sma_50=show_sma_50,
        show_sma_200=show_sma_200,
        show_bb=show_bb,
        show_rsi=show_rsi,
        show_macd=show_macd,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recent OHLCV Session Data Table (Collapsible)
    with st.expander("Recent Historical OHLCV Sessions", expanded=False):
        table_cols = ["Open", "High", "Low", "Close", "Volume", "Daily_Return", "SMA_20", "RSI_14", "ATR_14"]
        existing_cols = [c for c in table_cols if c in tech_df.columns]
        recent_df = tech_df[existing_cols].tail(15).iloc[::-1]

        st.dataframe(
            recent_df.style.format({
                "Open": f"{currency_symbol}{{:,.2f}}",
                "High": f"{currency_symbol}{{:,.2f}}",
                "Low": f"{currency_symbol}{{:,.2f}}",
                "Close": f"{currency_symbol}{{:,.2f}}",
                "Volume": "{:,.0f}",
                "Daily_Return": "{:+.2%}",
                "SMA_20": f"{currency_symbol}{{:,.2f}}",
                "RSI_14": "{:.1f}",
                "ATR_14": f"{currency_symbol}{{:,.2f}}",
            }),
            use_container_width=True,
        )
