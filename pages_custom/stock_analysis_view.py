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
    """Dedicated interactive stock research and technical analysis interface."""
    st.markdown(f'<div class="view-header">STOCK ANALYSIS: {selected_ticker}</div>', unsafe_allow_html=True)
    st.caption("Interactive technical analysis with configurable overlays, indicators, and volume profiling.")

    if prices_df is None or prices_df.empty:
        st.warning(f"No price data available for {selected_ticker}.")
        return

    # Calculate indicators
    tech_df = calculate_technical_indicators(prices_df)

    # Indicator control panel
    with st.expander("TECHNICAL INDICATOR CONTROLS", expanded=True):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            show_sma_20 = st.checkbox("SMA 20 (Cyan)", value=True)
        with c2:
            show_sma_50 = st.checkbox("SMA 50 (Amber)", value=True)
        with c3:
            show_sma_200 = st.checkbox("SMA 200 (Pink)", value=False)
        with c4:
            show_bb = st.checkbox("Bollinger Bands (20,2)", value=True)
        with c5:
            show_rsi = st.checkbox("RSI (14)", value=True)
        with c6:
            show_macd = st.checkbox("MACD (12,26,9)", value=True)

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

    # Technical Metrics Summary Grid
    st.markdown('<div class="section-title">KEY TECHNICAL STATS & INDICATOR VALUES</div>', unsafe_allow_html=True)
    last_row = tech_df.iloc[-1]
    last_close = float(last_row["Close"])
    daily_ret = float(tech_df["Close"].pct_change().iloc[-1])
    currency_symbol = "₹" if ".NS" in selected_ticker else "$"

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        render_kpi_card(
            label="LATEST CLOSE",
            value=format_currency(last_close, symbol=currency_symbol),
            delta=format_pct(daily_ret),
            delta_tone=tone_class(daily_ret),
        )
    with col2:
        rsi_val = float(last_row.get("RSI_14", 50.0))
        rsi_tone = "negative" if rsi_val > 70 else ("positive" if rsi_val < 30 else "neutral")
        render_kpi_card(label="RSI (14)", value=f"{rsi_val:.1f}", delta_tone=rsi_tone, subtext="Momentum")
    with col3:
        macd_val = float(last_row.get("MACD", 0.0))
        render_kpi_card(label="MACD (12,26)", value=format_number(macd_val), delta_tone=tone_class(macd_val))
    with col4:
        atr_val = float(last_row.get("ATR_14", 0.0))
        render_kpi_card(label="ATR (14)", value=format_currency(atr_val, symbol=currency_symbol), subtext="True Range")
    with col5:
        vol_20d = float(last_row.get("Rolling_Vol_20d", 0.0))
        render_kpi_card(label="20D VOLATILITY", value=f"{vol_20d*100.0:.1f}%", subtext="Annualized")
    with col6:
        high_52w = float(tech_df["High"].tail(252).max())
        render_kpi_card(label="52W HIGH", value=format_currency(high_52w, symbol=currency_symbol), subtext="Trailing Year")

    # Recent OHLCV Data Table
    st.markdown('<div class="section-title">RECENT OHLCV SESSIONS (LAST 15 TRADING DAYS)</div>', unsafe_allow_html=True)
    table_cols = ["Open", "High", "Low", "Close", "Volume", "Daily_Return", "SMA_20", "RSI_14", "ATR_14"]
    existing_cols = [c for c in table_cols if c in tech_df.columns]
    recent_df = tech_df[existing_cols].tail(15).iloc[::-1]

    st.dataframe(
        recent_df.style.format(
            {
                "Open": f"{currency_symbol}{{:,.2f}}",
                "High": f"{currency_symbol}{{:,.2f}}",
                "Low": f"{currency_symbol}{{:,.2f}}",
                "Close": f"{currency_symbol}{{:,.2f}}",
                "Volume": "{:,.0f}",
                "Daily_Return": "{:+.2%}",
                "SMA_20": f"{currency_symbol}{{:,.2f}}",
                "RSI_14": "{:.1f}",
                "ATR_14": f"{currency_symbol}{{:,.2f}}",
            }
        ),
        use_container_width=True,
    )

