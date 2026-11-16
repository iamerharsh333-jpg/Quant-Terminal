"""Risk Analytics View for Quant Terminal."""

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
    """Render institutional risk diagnostics: VaR, CVaR, Beta, Drawdown, and Rolling Volatility."""
    st.markdown(f'<div class="view-header">QUANTITATIVE RISK ANALYTICS: {selected_ticker}</div>', unsafe_allow_html=True)
    st.caption("Tail-risk modeling, Value at Risk (VaR), Conditional VaR, and benchmark sensitivity (CAPM Beta).")

    if prices_df is None or prices_df.empty:
        st.warning(f"No price data available for {selected_ticker}.")
        return

    # Select benchmark
    default_bench = "^NSEI" if ".NS" in selected_ticker else "^GSPC"
    bench_label = "NIFTY 50" if ".NS" in selected_ticker else "S&P 500"

    # Fetch benchmark
    bench_df, _ = load_market_data(default_bench, mode=data_mode)
    stock_returns = prices_df["Close"].pct_change().dropna()
    bench_returns = bench_df["Close"].pct_change().dropna() if not bench_df.empty else None

    # Compute risk analytics
    try:
        risk_res = compute_risk_analytics(
            stock_returns=stock_returns,
            benchmark_returns=bench_returns,
        )
    except Exception as exc:
        st.error(f"Error computing risk analytics: {exc}")
        return

    # 1. Primary Risk KPI Metrics
    st.markdown('<div class="section-title">TAIL RISK & SENSITIVITY ATTRIBUTION</div>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        render_kpi_card(
            label="95% 1-DAY HISTORICAL VAR",
            value=f"-{risk_res.historical_var_95 * 100.0:.2f}%",
            delta_tone="negative",
            subtext="5% Worst Case (1-Day)",
        )
    with r2:
        render_kpi_card(
            label="95% CONDITIONAL VAR (CVAR)",
            value=f"-{risk_res.cvar_95 * 100.0:.2f}%",
            delta_tone="negative",
            subtext="Expected Tail Loss",
        )
    with r3:
        b_tone = "positive" if risk_res.beta < 1.0 else "negative"
        render_kpi_card(
            label=f"BETA VS {bench_label}",
            value=format_number(risk_res.beta),
            delta_tone=b_tone,
            subtext="Market Systematic Risk",
        )
    with r4:
        render_kpi_card(
            label="JENSEN'S ALPHA",
            value=format_pct(risk_res.alpha),
            delta_tone=tone_class(risk_res.alpha),
            subtext="Annualized Excess Alpha",
        )

    r5, r6, r7, r8 = st.columns(4)
    with r5:
        render_kpi_card(
            label="99% 1-DAY HISTORICAL VAR",
            value=f"-{risk_res.historical_var_99 * 100.0:.2f}%",
            delta_tone="negative",
            subtext="1% Severe Tail Risk",
        )
    with r6:
        render_kpi_card(
            label="99% CONDITIONAL VAR (CVAR)",
            value=f"-{risk_res.cvar_99 * 100.0:.2f}%",
            delta_tone="negative",
            subtext="Extreme Expected Shortfall",
        )
    with r7:
        render_kpi_card(
            label="MAX DRAWDOWN DURATION",
            value=f"{risk_res.max_drawdown_duration_days} Days",
            subtext="Peak Recovery Period",
        )
    with r8:
        render_kpi_card(
            label="ANNUALIZED VOLATILITY",
            value=f"{risk_res.annualized_volatility * 100.0:.2f}%",
            subtext="Full Period Standard Dev",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Charts: 20-Day Rolling Volatility & 60-Day Rolling Sharpe
    col_v, col_s = st.columns(2)

    with col_v:
        st.markdown('<div class="section-title">20-DAY ROLLING ANNUALIZED VOLATILITY (%)</div>', unsafe_allow_html=True)
        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=risk_res.rolling_volatility_20d.index,
                y=risk_res.rolling_volatility_20d * 100.0,
                mode="lines",
                name="20D Volatility",
                line=dict(color="#f59e0b", width=1.8),
            )
        )
        fig_vol = style_figure(fig_vol, title="Rolling Historical Volatility Regime", height=380)
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_s:
        st.markdown('<div class="section-title">60-DAY ROLLING SHARPE RATIO</div>', unsafe_allow_html=True)
        fig_sh = go.Figure()
        fig_sh.add_trace(
            go.Scatter(
                x=risk_res.rolling_sharpe_60d.index,
                y=risk_res.rolling_sharpe_60d,
                mode="lines",
                name="60D Rolling Sharpe",
                line=dict(color="#22d3ee", width=1.8),
            )
        )
        fig_sh.add_hline(y=0.0, line_dash="dash", line_color="rgba(148, 163, 184, 0.4)")
        fig_sh = style_figure(fig_sh, title="Rolling Risk-Adjusted Efficiency", height=380)
        st.plotly_chart(fig_sh, use_container_width=True)

    # 3. Parametric vs Historical VaR Comparison Table
    st.markdown('<div class="section-title">VALUE AT RISK (VAR) COMPARISON MATRIX</div>', unsafe_allow_html=True)
    var_matrix = pd.DataFrame(
        [
            {
                "Confidence Level": "95.0%",
                "Historical VaR (1-Day)": f"-{risk_res.historical_var_95 * 100.0:.2f}%",
                "Parametric Gaussian VaR": f"-{risk_res.parametric_var_95 * 100.0:.2f}%",
                "Expected Shortfall (CVaR)": f"-{risk_res.cvar_95 * 100.0:.2f}%",
            },
            {
                "Confidence Level": "99.0%",
                "Historical VaR (1-Day)": f"-{risk_res.historical_var_99 * 100.0:.2f}%",
                "Parametric Gaussian VaR": f"-{risk_res.parametric_var_99 * 100.0:.2f}%",
                "Expected Shortfall (CVaR)": f"-{risk_res.cvar_99 * 100.0:.2f}%",
            },
        ]
    )
    st.dataframe(var_matrix, use_container_width=True)

    # 4. Underlying Risk Assumptions Expander
    with st.expander("MATHEMATICAL RISK ASSUMPTIONS", expanded=False):
        for assump in risk_res.assumptions:
            st.markdown(f"- {assump}")

