"""Watchlist View for Quant Terminal."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.data_loader import load_market_data
from core.feature_engineering import make_features
from core.model_engine import predict_signals, train_model


def render_watchlist_view(data_mode: str = "demo") -> None:
    """Clean multi-ticker scanner grid."""
    st.markdown('<div class="view-title">WATCHLIST SCANNER</div>', unsafe_allow_html=True)

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

    c_add1, c_add2 = st.columns([3, 1])
    with c_add1:
        new_ticker = st.text_input("Add Ticker", placeholder="e.g. TATAMOTORS.NS or TSLA", label_visibility="collapsed").strip().upper()
    with c_add2:
        if st.button("Add Ticker", use_container_width=True) and new_ticker:
            if new_ticker not in st.session_state["watchlist_symbols"]:
                st.session_state["watchlist_symbols"].append(new_ticker)
                st.rerun()

    symbols = st.session_state["watchlist_symbols"]

    watchlist_rows = []
    for sym in symbols:
        try:
            df, _ = load_market_data(sym, mode=data_mode)
            if not df.empty and len(df) > 30:
                last_p = float(df["Close"].iloc[-1])
                daily_chg = float(df["Close"].pct_change().iloc[-1])
                ret_5d = float((df["Close"].iloc[-1] / df["Close"].iloc[-5]) - 1.0) if len(df) >= 5 else 0.0
                vol_20d = float(df["Close"].pct_change().rolling(20).std().iloc[-1] * (252**0.5))
                cur_sym = "₹" if ".NS" in sym else "$"

                try:
                    f_set = make_features(df, horizon=3)
                    if len(f_set.X) > 50:
                        rf = train_model(f_set.X.iloc[:-1], f_set.y.iloc[:-1], n_estimators=25, max_depth=3)
                        sig = predict_signals(rf, f_set.X.tail(1)).iloc[0]
                    else:
                        sig = "HOLD"
                except Exception:
                    sig = "HOLD"

                watchlist_rows.append({
                    "Ticker": sym,
                    "Price": f"{cur_sym}{last_p:,.2f}",
                    "1D Change": daily_chg,
                    "5D Return": ret_5d,
                    "20D Vol": vol_20d,
                    "Signal": sig,
                })
        except Exception:
            pass

    if watchlist_rows:
        wl_df = pd.DataFrame(watchlist_rows)

        def style_signal(val: str) -> str:
            if val == "BUY":
                return "color: #16C784; font-weight: 700;"
            elif val == "SELL":
                return "color: #EA3943; font-weight: 700;"
            return "color: #8792A5;"

        def color_num(val):
            return "color: #16C784;" if val > 0 else ("color: #EA3943;" if val < 0 else "color: #8792A5;")

        st.dataframe(
            wl_df.style.format({
                "1D Change": "{:+.2%}",
                "5D Return": "{:+.2%}",
                "20D Vol": "{:.1%}",
            }).map(style_signal, subset=["Signal"]).map(color_num, subset=["1D Change", "5D Return"]),
            use_container_width=True,
            hide_index=True,
        )
