"""Modern High-Density Dashboard View for Quant Terminal."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import plot_candlestick_with_indicators, plot_equity_and_drawdown
from components.metric_cards import render_dashboard_top_kpis, render_kpi_card
from config.settings import BENCHMARK_INDICES
from core.backtesting_engine import BacktestResult
from core.data_loader import load_market_data
from core.feature_engineering import calculate_technical_indicators
from utils.formatting import format_currency, format_number, format_pct, tone_class


def render_dashboard_view(
    selected_ticker: str,
    market_code: str,
    prices_df: pd.DataFrame,
    features_df: pd.DataFrame,
    latest_signal: str,
    latest_prob: float,
    confidence: float,
    recommendation: object,
    backtest_result: BacktestResult | None,
    comparison_frame: pd.DataFrame | None,
    data_mode: str = "demo",
) -> None:
    """Render 4-section institutional dashboard."""
    currency_symbol = "₹" if ".NS" in selected_ticker else "$"
    latest_price = float(prices_df["Close"].iloc[-1]) if not prices_df.empty else 0.0
    daily_ret = float(prices_df["Close"].pct_change().iloc[-1]) if len(prices_df) > 1 else 0.0

    # ==========================================
    # SECTION 1: TOP 4 COMPACT KPI CARDS
    # ==========================================
    render_dashboard_top_kpis(
        latest_price=latest_price,
        daily_return=daily_ret,
        predicted_signal=latest_signal,
        confidence=confidence,
        currency_symbol=currency_symbol,
    )

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    # ==========================================
    # SECTION 2: PRICE CHART & MARKET OVERVIEW
    # ==========================================
    col_chart, col_market = st.columns([2.2, 1.0])

    with col_chart:
        tech_df = calculate_technical_indicators(prices_df)
        fig_candle = plot_candlestick_with_indicators(
            df=tech_df,
            ticker=selected_ticker,
            show_sma_20=True,
            show_sma_50=True,
            show_sma_200=False,
            show_bb=True,
            show_rsi=False,
            show_macd=False,
        )
        st.plotly_chart(fig_candle, use_container_width=True)

    with col_market:
        st.markdown('<div class="card-title">MARKET OVERVIEW</div>', unsafe_allow_html=True)
        bench_rows = []
        for name, sym in BENCHMARK_INDICES.items():
            try:
                b_df, _ = load_market_data(sym, mode=data_mode)
                if not b_df.empty:
                    p = b_df["Close"].iloc[-1]
                    chg = b_df["Close"].pct_change().iloc[-1]
                    sym_c = "₹" if "NSE" in sym or "BSE" in sym else "$"
                    bench_rows.append({
                        "Index": name,
                        "Price": f"{sym_c}{p:,.0f}",
                        "Change": chg,
                    })
            except Exception:
                pass

        if bench_rows:
            b_df = pd.DataFrame(bench_rows)
            def color_chg(val):
                return "color: #16C784;" if val > 0 else ("color: #EA3943;" if val < 0 else "color: #8792A5;")

            st.dataframe(
                b_df.style.format({"Change": "{:+.2%}"}).map(color_chg, subset=["Change"]),
                use_container_width=True,
                hide_index=True,
                height=195,
            )

        # Recommendation badge
        sig_class = "positive" if latest_signal == "BUY" else ("negative" if latest_signal == "SELL" else "neutral")
        st.markdown(
            f"""
            <div class="terminal-card" style="margin-top: 10px; padding: 12px;">
                <div class="kpi-header">RECOMMENDATION SYNTHESIS</div>
                <div class="kpi-row" style="margin: 4px 0 8px 0;">
                    <span class="kpi-value {sig_class}">{latest_signal}</span>
                    <span class="kpi-delta neutral">{latest_prob*100.0:.1f}% Up</span>
                </div>
                <div style="font-size: 0.78rem; color: #8792A5; line-height: 1.4;">
                    {recommendation.summary if hasattr(recommendation, 'summary') else ''}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    # ==========================================
    # SECTION 3: STRATEGY PERFORMANCE, MODEL COMPARISON, PORTFOLIO SUMMARY
    # ==========================================
    col_strat, col_comp, col_port = st.columns([1.1, 1.3, 0.9])

    with col_strat:
        st.markdown('<div class="card-title">STRATEGY PERFORMANCE</div>', unsafe_allow_html=True)
        if backtest_result and not backtest_result.frame.empty:
            m = backtest_result.metrics
            s_ret = m.get("strategy_total_return", 0.0)
            sh = m.get("strategy_sharpe_ratio", 0.0)
            dd = m.get("strategy_max_drawdown", 0.0)
            wr = m.get("win_rate", 0.0)

            c_sub1, c_sub2 = st.columns(2)
            with c_sub1:
                render_kpi_card("STRATEGY RET", format_pct(s_ret), delta_tone=tone_class(s_ret))
                render_kpi_card("MAX DRAWDOWN", f"-{dd*100.0:.1f}%", delta_tone="negative")
            with c_sub2:
                render_kpi_card("SHARPE", format_number(sh), delta_tone=tone_class(sh))
                render_kpi_card("WIN RATE", f"{wr*100.0:.0f}%", delta_tone="positive" if wr > 0.5 else "neutral")
        else:
            st.info("Run strategy to evaluate backtest.")

    with col_comp:
        st.markdown('<div class="card-title">MODEL BENCHMARK</div>', unsafe_allow_html=True)
        if comparison_frame is not None and not comparison_frame.empty:
            st.dataframe(
                comparison_frame[["model", "accuracy", "strategy_total_return", "strategy_sharpe_ratio"]].style.format({
                    "accuracy": "{:.1%}",
                    "strategy_total_return": "{:+.1%}",
                    "strategy_sharpe_ratio": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
                height=165,
            )

    with col_port:
        st.markdown('<div class="card-title">PORTFOLIO QUICK VIEW</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="terminal-card" style="padding: 12px;">
                <div class="kpi-header">ACTIVE ASSET WEIGHT</div>
                <div class="kpi-value" style="font-size: 1.25rem;">100.0%</div>
                <div style="font-size: 0.75rem; color: #8792A5; margin-top: 6px;">
                    Ticker: <b style="color: #E6EAF0;">{selected_ticker}</b><br>
                    Vol (20D): <b style="color: #E6EAF0;">{features_df.get('annualized_volatility_20d', pd.Series([0.2])).iloc[-1]*100.0:.1f}%</b><br>
                    RSI (14): <b style="color: #E6EAF0;">{features_df.get('rsi_14', pd.Series([50.0])).iloc[-1]:.1f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    # ==========================================
    # SECTION 4: RECENT PREDICTIONS & WATCHLIST
    # ==========================================
    col_preds, col_watch = st.columns([1.3, 1.0])

    with col_preds:
        st.markdown('<div class="card-title">RECENT PREDICTIONS</div>', unsafe_allow_html=True)
        if backtest_result and not backtest_result.frame.empty and "signal" in backtest_result.frame.columns:
            recent_p = backtest_result.frame[["signal", "strategy_return", "market_return"]].tail(5).iloc[::-1]
            st.dataframe(
                recent_p.style.format({
                    "strategy_return": "{:+.2%}",
                    "market_return": "{:+.2%}",
                }),
                use_container_width=True,
                height=165,
            )

    with col_watch:
        st.markdown('<div class="card-title">WATCHLIST SNIPPET</div>', unsafe_allow_html=True)
        quick_wl = [
            {"Ticker": "RELIANCE.NS", "Price": "₹2,420.50", "Signal": "BUY"},
            {"Ticker": "TCS.NS", "Price": "₹3,910.10", "Signal": "HOLD"},
            {"Ticker": "AAPL", "Price": "$224.30", "Signal": "BUY"},
            {"Ticker": "NVDA", "Price": "$142.50", "Signal": "BUY"},
        ]
        st.dataframe(pd.DataFrame(quick_wl), use_container_width=True, hide_index=True, height=165)
