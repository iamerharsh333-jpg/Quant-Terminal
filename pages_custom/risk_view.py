"""Modern Risk Analytics View for Quant Terminal."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import style_figure
from components.metric_cards import render_kpi_card
from core.data_loader import load_market_data
from core.risk_engine import compute_risk_analytics
from utils.formatting import format_number, format_pct, tone_class


def render_risk_view(
    selected_ticker: str,
    prices_df: pd.DataFrame,
    data_mode: str = "demo",
) -> None:
    """Compact institutional risk dashboard without text bloat."""
    st.markdown(f'<div class="view-title">RISK ANALYTICS — {selected_ticker}</div>', unsafe_allow_html=True)

    if prices_df is None or prices_df.empty:
        st.warning(f"No price data available for {selected_ticker}.")
        return

    default_bench = "^NSEI" if ".NS" in selected_ticker else "^GSPC"
    bench_label = "NIFTY 50" if ".NS" in selected_ticker else "S&P 500"

    bench_df, _ = load_market_data(default_bench, mode=data_mode)
    stock_returns = prices_df["Close"].pct_change().dropna()
    bench_returns = bench_df["Close"].pct_change().dropna() if not bench_df.empty else None

    try:
        risk_res = compute_risk_analytics(
            stock_returns=stock_returns,
            benchmark_returns=bench_returns,
        )
    except Exception as exc:
        st.error(f"Error computing risk analytics: {exc}")
        return

    # Primary Risk KPI Cards
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    with r1:
        render_kpi_card("95% VaR (1D)", f"-{risk_res.historical_var_95 * 100.0:.2f}%", delta_tone="negative")
    with r2:
        render_kpi_card("95% CVaR (1D)", f"-{risk_res.cvar_95 * 100.0:.2f}%", delta_tone="negative")
    with r3:
        b_tone = "positive" if risk_res.beta < 1.0 else "neutral"
        render_kpi_card(f"BETA ({bench_label[:5]})", format_number(risk_res.beta), delta_tone=b_tone)
    with r4:
        render_kpi_card("ALPHA (ANN.)", format_pct(risk_res.alpha), delta_tone=tone_class(risk_res.alpha))
    with r5:
        render_kpi_card("MAX DD", f"-{risk_res.max_drawdown*100.0:.1f}%", delta_tone="negative")
    with r6:
        render_kpi_card("ANNUAL VOL", f"{risk_res.annualized_volatility * 100.0:.1f}%")

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    # 2 Charts: 20D Rolling Vol & 60D Rolling Sharpe
    col_v, col_s = st.columns(2)

    with col_v:
        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=risk_res.rolling_volatility_20d.index,
                y=risk_res.rolling_volatility_20d * 100.0,
                mode="lines",
                name="20D Vol",
                line=dict(color="#F59E0B", width=1.6),
            )
        )
        fig_vol = style_figure(fig_vol, title="20-Day Rolling Annualized Volatility (%)", height=320)
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_s:
        fig_sh = go.Figure()
        fig_sh.add_trace(
            go.Scatter(
                x=risk_res.rolling_sharpe_60d.index,
                y=risk_res.rolling_sharpe_60d,
                mode="lines",
                name="60D Sharpe",
                line=dict(color="#22D3EE", width=1.6),
            )
        )
        fig_sh.add_hline(y=0.0, line_dash="dash", line_color="#8792A5", opacity=0.4)
        fig_sh = style_figure(fig_sh, title="60-Day Rolling Sharpe Ratio", height=320)
        st.plotly_chart(fig_sh, use_container_width=True)

    # Value at Risk Matrix (Compact)
    st.markdown('<div class="card-title">VALUE AT RISK (VaR) & CVaR COMPARISON MATRIX</div>', unsafe_allow_html=True)
    var_matrix = pd.DataFrame([
        {
            "Confidence": "95.0%",
            "Historical VaR": f"-{risk_res.historical_var_95 * 100.0:.2f}%",
            "Parametric VaR": f"-{risk_res.parametric_var_95 * 100.0:.2f}%",
            "Expected Shortfall (CVaR)": f"-{risk_res.cvar_95 * 100.0:.2f}%",
        },
        {
            "Confidence": "99.0%",
            "Historical VaR": f"-{risk_res.historical_var_99 * 100.0:.2f}%",
            "Parametric VaR": f"-{risk_res.parametric_var_99 * 100.0:.2f}%",
            "Expected Shortfall (CVaR)": f"-{risk_res.cvar_99 * 100.0:.2f}%",
        },
    ])
    st.dataframe(var_matrix, use_container_width=True, hide_index=True)
