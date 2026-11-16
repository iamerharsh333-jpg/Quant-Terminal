"""Compact institutional metric cards for Quant Terminal."""

from __future__ import annotations

import streamlit as st
from utils.formatting import format_currency, format_number, format_pct, tone_class


def render_kpi_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_tone: str = "neutral",
    tooltip: str | None = None,
) -> None:
    """Render a single compact fintech KPI card."""
    delta_html = f'<span class="kpi-delta {delta_tone}">{delta}</span>' if delta else ""
    help_attr = f'title="{tooltip}"' if tooltip else ""

    st.markdown(
        f"""
        <div class="terminal-card kpi-card" {help_attr}>
            <div class="kpi-header">{label}</div>
            <div class="kpi-row">
                <span class="kpi-value">{value}</span>
                {delta_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_top_kpis(
    latest_price: float,
    daily_return: float,
    predicted_signal: str,
    confidence: float,
    currency_symbol: str = "₹",
) -> None:
    """Render the exact four compact KPI cards required for the dashboard top section."""
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_kpi_card(
            label="MARKET PRICE",
            value=format_currency(latest_price, symbol=currency_symbol),
            tooltip="Latest available closing price",
        )
    with c2:
        d_tone = tone_class(daily_return)
        render_kpi_card(
            label="DAILY CHANGE",
            value=format_pct(daily_return),
            delta_tone=d_tone,
            tooltip="24-hour return",
        )
    with c3:
        sig_class = "positive" if predicted_signal == "BUY" else ("negative" if predicted_signal == "SELL" else "neutral")
        render_kpi_card(
            label="MODEL SIGNAL",
            value=predicted_signal,
            delta="Conviction" if predicted_signal != "HOLD" else "Neutral",
            delta_tone=sig_class,
            tooltip="Directional classification signal",
        )
    with c4:
        render_kpi_card(
            label="PREDICTION CONFIDENCE",
            value=f"{confidence * 100.0:.0f}%",
            delta="High" if confidence > 0.6 else "Moderate",
            delta_tone="positive" if confidence > 0.6 else "neutral",
            tooltip="Probability distance from decision threshold",
        )
