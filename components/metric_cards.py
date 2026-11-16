"""Institutional terminal KPI and metric cards."""

from __future__ import annotations

import streamlit as st
from utils.formatting import format_currency, format_number, format_pct, tone_class


def render_kpi_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_tone: str = "neutral",
    subtext: str | None = None,
) -> None:
    """Render a single styled institutional KPI card."""
    delta_html = ""
    if delta is not None:
        delta_html = f'<div class="kpi-delta {delta_tone}">{delta}</div>'

    subtext_html = ""
    if subtext:
        subtext_html = f'<div class="kpi-subtext">{subtext}</div>'

    st.markdown(
        f"""
        <div class="terminal-kpi-card">
            <div class="kpi-header">{label}</div>
            <div class="kpi-main">
                <div class="kpi-value">{value}</div>
                {delta_html}
            </div>
            {subtext_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_kpis(
    latest_price: float,
    daily_return: float,
    predicted_signal: str,
    confidence: float,
    strategy_return: float,
    sharpe_ratio: float,
    max_drawdown: float,
    currency_symbol: str = "₹",
) -> None:
    """Render the standard top 7 terminal KPI metrics across columns."""
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    with c1:
        render_kpi_card(
            label="PRICE",
            value=format_currency(latest_price, symbol=currency_symbol),
            subtext="Last Available",
        )
    with c2:
        d_tone = tone_class(daily_return)
        render_kpi_card(
            label="24H CHANGE",
            value=format_pct(daily_return),
            delta_tone=d_tone,
            subtext="Daily Return",
        )
    with c3:
        sig_class = "positive" if predicted_signal == "BUY" else ("negative" if predicted_signal == "SELL" else "neutral")
        render_kpi_card(
            label="ML SIGNAL",
            value=predicted_signal,
            delta=f"{confidence * 100.0:.0f}% Conf",
            delta_tone=sig_class,
            subtext="Directional Bias",
        )
    with c4:
        s_tone = tone_class(strategy_return)
        render_kpi_card(
            label="STRATEGY RET",
            value=format_pct(strategy_return),
            delta_tone=s_tone,
            subtext="Test Period",
        )
    with c5:
        sh_tone = tone_class(sharpe_ratio)
        render_kpi_card(
            label="SHARPE RATIO",
            value=format_number(sharpe_ratio),
            delta_tone=sh_tone,
            subtext="Annualized Risk-Adj",
        )
    with c6:
        render_kpi_card(
            label="MAX DRAWDOWN",
            value=f"-{max_drawdown * 100.0:.1f}%",
            delta_tone="negative",
            subtext="Peak to Trough",
        )
    with c7:
        render_kpi_card(
            label="STATUS",
            value="ACTIVE",
            delta="Online",
            delta_tone="positive",
            subtext="Engine Synced",
        )

