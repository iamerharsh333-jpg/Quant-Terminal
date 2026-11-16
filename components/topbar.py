"""Compact institutional top header component."""

from __future__ import annotations

from datetime import datetime
import streamlit as st


def render_topbar(
    selected_ticker: str,
    market_name: str,
    data_provider_name: str,
    is_synthetic: bool = False,
) -> None:
    """Render compact Bloomberg/TradingView style top navigation bar."""
    now_str = datetime.now().strftime("%H:%M:%S UTC")

    status_badge = (
        '<span class="terminal-badge badge-demo">DEMO (SYNTHETIC)</span>'
        if is_synthetic
        else '<span class="terminal-badge badge-live">HISTORICAL</span>'
    )

    st.markdown(
        f"""
        <div class="terminal-topbar">
            <div class="topbar-left">
                <span class="terminal-logo">QUANT TERMINAL</span>
                <span class="terminal-tag">v2.0</span>
            </div>
            <div class="topbar-right">
                <span class="ticker-pill">{selected_ticker}</span>
                <span class="market-pill">{market_name}</span>
                {status_badge}
                <span class="time-pill">{now_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
