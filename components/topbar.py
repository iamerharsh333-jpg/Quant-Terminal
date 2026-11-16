"""Institutional top header component."""

from __future__ import annotations

from datetime import datetime
import streamlit as st


def render_topbar(
    selected_ticker: str,
    market_name: str,
    data_provider_name: str,
    is_synthetic: bool = False,
) -> None:
    """Render Bloomberg/TradingView style institutional topbar."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    status_badge = (
        '<span class="terminal-badge badge-demo">● DEMO MODE (SYNTHETIC)</span>'
        if is_synthetic
        else '<span class="terminal-badge badge-live">● HISTORICAL DATA</span>'
    )

    st.markdown(
        f"""
        <div class="terminal-topbar">
            <div class="topbar-left">
                <span class="terminal-logo">QUANT TERMINAL</span>
                <span class="topbar-divider">|</span>
                <span class="ticker-chip">{selected_ticker}</span>
                <span class="market-chip">{market_name}</span>
                {status_badge}
            </div>
            <div class="topbar-right">
                <span class="provider-info">SOURCE: {data_provider_name}</span>
                <span class="timestamp-info">{now_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

