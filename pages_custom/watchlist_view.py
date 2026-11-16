"""Watchlist View for Quant Terminal."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import INDIAN_EQUITIES, US_EQUITIES
from core.data_loader import load_market_data
from core.feature_engineering import make_features
from core.model_engine import predict_signals, train_model
from utils.formatting import format_currency, format_pct, tone_class


def render_watchlist_view(data_mode: str = "demo") -> None:
    """Render dynamic market watchlist with technical indicators and ML signal ratings."""
    st.markdown('<div class="view-header">QUANT WATCHLIST & OPPORTUNITY SCANNER</div>', unsafe_allow_html=True)
    st.caption("Cross-asset universe monitor with real-time volatility and directional conviction signals.")

    # Session State Watchlist initialization
    if "watchlist_symbols" not in st.session_state:
        st.session_state["watchlist_symbols"] = [
            "RELIANCE.NS",
            "TCS.NS",
            "INFY.NS",
            "HDFCBANK.NS",
            "AAPL",
            "MSFT",
            "NVDA",
        ]

    # Add custom ticker bar
    c_add1, c_add2 = st.columns([3, 1])
    with c_add1:
        new_ticker = st.text_input("Add Ticker to Watchlist", placeholder="e.g. TATAMOTORS.NS or TSLA").strip().upper()
    with c_add2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Add Ticker", use_container_width=True) and new_ticker:
            if new_ticker not in st.session_state["watchlist_symbols"]:
                st.session_state["watchlist_symbols"].append(new_ticker)
                st.success(f"Added {new_ticker} to watchlist.")

    symbols = st.session_state["watchlist_symbols"]

    # Compute Watchlist Metrics
    watchlist_rows = []
    for sym in symbols:
        try:
            df, _ = load_market_data(sym, mode=data_mode)
            if not df.empty and len(df) > 30:
                last_p = float(df["Close"].iloc[-1])
                daily_chg = float(df["Close"].pct_change().iloc[-1])
                ret_5d = float((df["Close"].iloc[-1] / df["Close"].iloc[-5]) - 1.0) if len(df) >= 5 else 0.0
                vol_20d = float(df["Close"].pct_change().rolling(20).std().iloc[-1] * (252**0.5))
                high_52w = float(df["High"].tail(252).max())
                low_52w = float(df["Low"].tail(252).min())
                cur_sym = "₹" if ".NS" in sym else "$"

                # Lightweight quick ML signal
                try:
                    f_set = make_features(df, horizon=3)
                    if len(f_set.X) > 50:
                        rf = train_model(f_set.X.iloc[:-1], f_set.y.iloc[:-1], n_estimators=30, max_depth=4)
                        sig = predict_signals(rf, f_set.X.tail(1)).iloc[0]
                    else:
                        sig = "NEUTRAL"
                except Exception:
                    sig = "NEUTRAL"

                watchlist_rows.append(
                    {
                        "Ticker": sym,
                        "Price": f"{cur_sym}{last_p:,.2f}",
                        "1D Change": daily_chg,
                        "5D Return": ret_5d,
                        "20D Volatility": vol_20d,
                        "52W Range": f"{cur_sym}{low_52w:,.1f} - {cur_sym}{high_52w:,.1f}",
                        "Quant Signal": sig,
                    }
                )
        except Exception:
            pass

    if watchlist_rows:
        wl_df = pd.DataFrame(watchlist_rows)

        def style_signal(val: str) -> str:
            if val == "BUY":
                return "color: #10b981; font-weight: bold;"
            elif val == "SELL":
                return "color: #ef4444; font-weight: bold;"
            return "color: #94a3b8;"

        st.markdown('<div class="section-title">ACTIVE WATCHLIST GRID</div>', unsafe_allow_html=True)
        st.dataframe(
            wl_df.style.format(
                {
                    "1D Change": "{:+.2%}",
                    "5D Return": "{:+.2%}",
                    "20D Volatility": "{:.1%}",
                }
            ).map(style_signal, subset=["Quant Signal"]),
            use_container_width=True,
        )

    # Option to clear or reset watchlist
    if st.button("Reset Watchlist to Defaults"):
        st.session_state["watchlist_symbols"] = [
            "RELIANCE.NS",
            "TCS.NS",
            "INFY.NS",
            "HDFCBANK.NS",
            "AAPL",
            "MSFT",
            "NVDA",
        ]
        st.rerun()

