"""Dashboard View for Quant Terminal."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import plot_equity_and_drawdown
from components.metric_cards import render_dashboard_kpis, render_kpi_card
from config.settings import BENCHMARK_INDICES
from core.backtesting_engine import BacktestResult
from core.data_loader import load_market_data
from utils.formatting import format_number, format_pct, tone_class


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
    """Render comprehensive institutional terminal dashboard."""
    currency_symbol = "₹" if ".NS" in selected_ticker else "$"

    # 1. Market Indices Ticker Bar
    st.markdown('<div class="section-title">GLOBAL BENCHMARK INDICATORS</div>', unsafe_allow_html=True)
    idx_cols = st.columns(len(BENCHMARK_INDICES))
    for col, (name, sym) in zip(idx_cols, BENCHMARK_INDICES.items()):
        with col:
            try:
                idx_df, _ = load_market_data(sym, mode=data_mode)
                if not idx_df.empty:
                    last_p = idx_df["Close"].iloc[-1]
                    d_ret = idx_df["Close"].pct_change().iloc[-1]
                    sym_c = "₹" if "NSE" in sym or "BSE" in sym else "$"
                    render_kpi_card(
                        label=name,
                        value=f"{sym_c}{last_p:,.0f}",
                        delta=format_pct(d_ret),
                        delta_tone=tone_class(d_ret),
                        subtext="Benchmark Index",
                    )
                else:
                    render_kpi_card(label=name, value="N/A", subtext="Unavailable")
            except Exception:
                render_kpi_card(label=name, value="N/A", subtext="Offline")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Main KPI Cards for Selected Asset
    st.markdown(f'<div class="section-title">ACTIVE ASSET METRICS: {selected_ticker}</div>', unsafe_allow_html=True)
    latest_price = float(prices_df["Close"].iloc[-1]) if not prices_df.empty else 0.0
    daily_ret = float(prices_df["Close"].pct_change().iloc[-1]) if len(prices_df) > 1 else 0.0

    strat_ret = backtest_result.metrics.get("strategy_total_return", 0.0) if backtest_result else 0.0
    sharpe = backtest_result.metrics.get("strategy_sharpe_ratio", 0.0) if backtest_result else 0.0
    max_dd = backtest_result.metrics.get("strategy_max_drawdown", 0.0) if backtest_result else 0.0

    render_dashboard_kpis(
        latest_price=latest_price,
        daily_return=daily_ret,
        predicted_signal=latest_signal,
        confidence=confidence,
        strategy_return=strat_ret,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        currency_symbol=currency_symbol,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Two-Column Layout: Chart Left, Recommendation/Signals Right
    col_chart, col_side = st.columns([1.6, 1.0])

    with col_chart:
        st.markdown('<div class="section-title">STRATEGY VS BENCHMARK EQUITY CURVE</div>', unsafe_allow_html=True)
        if backtest_result and not backtest_result.frame.empty:
            fig = plot_equity_and_drawdown(
                backtest_result.frame,
                title=f"{selected_ticker} Strategy Performance vs Buy & Hold",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run strategy backtest to generate performance equity curves.")

    with col_side:
        st.markdown('<div class="section-title">QUANTITATIVE RECOMMENDATION ENGINE</div>', unsafe_allow_html=True)
        sig_class = "positive" if latest_signal == "BUY" else ("negative" if latest_signal == "SELL" else "neutral")

        st.markdown(
            f"""
            <div class="terminal-card">
                <div class="kpi-header">DIRECTIONAL SYNTHESIS</div>
                <div class="signal-banner {sig_class}">
                    ACTION: {latest_signal} ({latest_prob * 100.0:.1f}% Upside Prob)
                </div>
                <p class="card-copy">{recommendation.summary if hasattr(recommendation, 'summary') else ''}</p>
                <div class="kpi-header" style="margin-top: 10px;">KEY RATIONALE</div>
                <ul class="terminal-list">
                    {''.join(f'<li>{r}</li>' for r in getattr(recommendation, 'reasons', []))}
                </ul>
                <div class="kpi-header" style="margin-top: 10px;">RISK FACTORS</div>
                <ul class="terminal-list">
                    {''.join(f'<li>{n}</li>' for n in getattr(recommendation, 'risk_notes', []))}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Model Comparison Quick Summary
    if comparison_frame is not None and not comparison_frame.empty:
        st.markdown('<div class="section-title">MODEL PERFORMANCE COMPARISON (TEST DATA)</div>', unsafe_allow_html=True)
        st.dataframe(
            comparison_frame.style.format(
                {
                    "accuracy": "{:.2%}",
                    "precision": "{:.2%}",
                    "recall": "{:.2%}",
                    "f1": "{:.3f}",
                    "strategy_total_return": "{:+.2%}",
                    "strategy_sharpe_ratio": "{:.2f}",
                    "strategy_max_drawdown": "{:.2%}",
                    "buy_hold_total_return": "{:+.2%}",
                    "win_rate": "{:.1%}",
                }
            ),
            use_container_width=True,
        )

