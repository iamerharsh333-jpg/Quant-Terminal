"""Settings & Configuration View for Quant Terminal."""

from __future__ import annotations

import shutil
from pathlib import Path
import streamlit as st

from config.settings import API_CONFIG, APP_CONFIG, CACHE_DIR, OUTPUT_DIR, TRADING_CONFIG


def render_settings_view() -> None:
    """Render system settings, data architecture mode switcher, and API credentials status."""
    st.markdown('<div class="view-header">SETTINGS & DATA ARCHITECTURE CONFIGURATION</div>', unsafe_allow_html=True)
    st.caption("Central settings, environment credentials status, caching maintenance, and data mode controls.")

    # 1. Data Provider Mode Switcher
    st.markdown('<div class="section-title">ACTIVE DATA ARCHITECTURE MODE</div>', unsafe_allow_html=True)

    current_mode = st.session_state.get("data_provider_mode", "demo")

    mode_options = {
        "demo": "1. Demo Mode (Offline Synthetic Engine - Zero API Keys, Zero Network Required) [DEFAULT]",
        "historical": "2. Historical Data Mode (Yahoo Finance with Automatic Disk Caching)",
        "api": "3. External Commercial API Mode (Requires API Credentials)",
    }

    selected_mode_key = st.radio(
        "Select Active Market Data Engine",
        options=list(mode_options.keys()),
        format_func=lambda k: mode_options[k],
        index=0 if current_mode == "demo" else (1 if current_mode == "historical" else 2),
    )

    if selected_mode_key != current_mode:
        st.session_state["data_provider_mode"] = selected_mode_key
        st.success(f"Switched data engine to: {mode_options[selected_mode_key]}")
        st.rerun()

    if selected_mode_key == "api" and not API_CONFIG.has_any_credentials:
        st.warning(
            "External API Mode is currently INACTIVE because no credentials are configured in environment variables. "
            "To connect external data vendors, set ALPHA_VANTAGE_API_KEY, POLYGON_API_KEY, or TIINGO_API_KEY in your .env file."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Local Caching Maintenance
    st.markdown('<div class="section-title">LOCAL DISK CACHE STATUS</div>', unsafe_allow_html=True)
    cache_path = Path(CACHE_DIR)
    cached_files = list(cache_path.glob("*.csv")) if cache_path.exists() else []
    cache_size_kb = sum(f.stat().st_size for f in cached_files) / 1024.0 if cached_files else 0.0

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.metric("Cached Datasets", len(cached_files))
    with col_c2:
        st.metric("Total Cache Size", f"{cache_size_kb:,.1f} KB")
    with col_c3:
        st.metric("Cache Directory", str(cache_path.name))

    if st.button("Purge Local Cache Directory", type="secondary"):
        if cache_path.exists():
            for f in cache_path.glob("*.csv"):
                f.unlink()
            st.success("Local cache successfully cleared.")
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Environment Variables & API Key Integration Points
    st.markdown('<div class="section-title">FUTURE API CREDENTIALS & SECURITY STATUS</div>', unsafe_allow_html=True)
    st.info(
        "Quant Terminal is architected with a modular provider abstraction. "
        "Credentials are read strictly from secure environment variables or a local `.env` file and are never committed to git."
    )

    cred_status = [
        {
            "Provider Adapter": "Alpha Vantage API",
            "Env Variable": "ALPHA_VANTAGE_API_KEY",
            "Status": "CONFIGURED" if API_CONFIG.alpha_vantage_api_key else "OPTIONAL (Not Set)",
        },
        {
            "Provider Adapter": "Polygon.io API",
            "Env Variable": "POLYGON_API_KEY",
            "Status": "CONFIGURED" if API_CONFIG.polygon_api_key else "OPTIONAL (Not Set)",
        },
        {
            "Provider Adapter": "Tiingo Financial API",
            "Env Variable": "TIINGO_API_KEY",
            "Status": "CONFIGURED" if API_CONFIG.tiingo_api_key else "OPTIONAL (Not Set)",
        },
        {
            "Provider Adapter": "Zerodha Kite Connect",
            "Env Variable": "ZERODHA_API_KEY",
            "Status": "CONFIGURED" if API_CONFIG.zerodha_api_key else "OPTIONAL (Not Set)",
        },
    ]
    st.dataframe(pd.DataFrame(cred_status), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. System Specifications
    st.markdown('<div class="section-title">SYSTEM SPECIFICATIONS & ARCHITECTURE</div>', unsafe_allow_html=True)
    spec_cols = st.columns(4)
    with spec_cols[0]:
        st.caption("PLATFORM")
        st.text(APP_CONFIG.title)
    with spec_cols[1]:
        st.caption("VERSION")
        st.text(APP_CONFIG.version)
    with spec_cols[2]:
        st.caption("DEFAULT CAPITAL")
        st.text(f"${TRADING_CONFIG.initial_capital:,.0f}")
    with spec_cols[3]:
        st.caption("ANNUAL FACTOR")
        st.text(f"{TRADING_CONFIG.annualization_factor} Days")

