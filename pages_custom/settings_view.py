"""Modern Settings View for Quant Terminal."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from config.settings import API_CONFIG, APP_CONFIG, CACHE_DIR, TRADING_CONFIG


def render_settings_view() -> None:
    """Streamlined settings and configuration view."""
    st.markdown('<div class="view-title">TERMINAL SETTINGS & DATA CONFIGURATION</div>', unsafe_allow_html=True)

    # 1. Data Provider Mode Switcher
    st.markdown('<div class="card-title">DATA ENGINE MODE</div>', unsafe_allow_html=True)
    current_mode = st.session_state.get("data_provider_mode", "demo")

    mode_options = {
        "demo": "Demo Mode (Offline Synthetic Engine — Zero API Keys Required) [Default]",
        "historical": "Historical Data Mode (Yahoo Finance with Local Disk Cache)",
        "api": "External API Mode (Commercial Vendors via Environment Variables)",
    }

    selected_mode = st.radio(
        "Mode Selector",
        options=list(mode_options.keys()),
        format_func=lambda k: mode_options[k],
        index=0 if current_mode == "demo" else (1 if current_mode == "historical" else 2),
        label_visibility="collapsed",
    )

    if selected_mode != current_mode:
        st.session_state["data_provider_mode"] = selected_mode
        st.rerun()

    if selected_mode == "api" and not API_CONFIG.has_any_credentials:
        st.caption("External API keys not detected. Configure keys in `.env` to activate live commercial streaming.")

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    # 2. Local Disk Cache Maintenance
    st.markdown('<div class="card-title">DISK CACHE STATUS</div>', unsafe_allow_html=True)
    cache_path = Path(CACHE_DIR)
    cached_files = list(cache_path.glob("*.csv")) if cache_path.exists() else []
    cache_size_kb = sum(f.stat().st_size for f in cached_files) / 1024.0 if cached_files else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Cached Files", len(cached_files))
    with c2:
        st.metric("Cache Size", f"{cache_size_kb:.1f} KB")
    with c3:
        if st.button("Purge Cache"):
            for f in cached_files:
                f.unlink()
            st.rerun()

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    # 3. Environment Variables & API Key Integration Points
    st.markdown('<div class="card-title">API ADAPTER CREDENTIALS STATUS</div>', unsafe_allow_html=True)
    cred_status = [
        {"Provider": "Alpha Vantage", "Variable": "ALPHA_VANTAGE_API_KEY", "Status": "Ready" if API_CONFIG.alpha_vantage_api_key else "Not Set"},
        {"Provider": "Polygon.io", "Variable": "POLYGON_API_KEY", "Status": "Ready" if API_CONFIG.polygon_api_key else "Not Set"},
        {"Provider": "Tiingo", "Variable": "TIINGO_API_KEY", "Status": "Ready" if API_CONFIG.tiingo_api_key else "Not Set"},
        {"Provider": "Zerodha Kite", "Variable": "ZERODHA_API_KEY", "Status": "Ready" if API_CONFIG.zerodha_api_key else "Not Set"},
    ]
    st.dataframe(pd.DataFrame(cred_status), use_container_width=True, hide_index=True)
