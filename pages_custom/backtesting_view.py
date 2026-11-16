"""Strategy Backtesting View for Quant Terminal."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import plot_equity_and_drawdown
from components.metric_cards import render_kpi_card
from core.backtesting_engine import BacktestResult
from utils.formatting import format_currency, format_number, format_pct, tone_class


def render_backtesting_view(
    selected_ticker: str,
    selected_model: str,
    backtest_result: BacktestResult | None,
    initial_capital: float = 100_000.0,
) -> None:
    """Render scientific backtesting performance diagnostics and trade ledger."""
    st.markdown(f'<div class="view-header">STRATEGY BACKTESTING: {selected_ticker}</div>', unsafe_allow_html=True)
    st.caption(f"Strategy: {selected_model} Signals | Realistic 1-Bar Execution Lag | Friction Accounted")

    if backtest_result is None or backtest_result.frame.empty:
        st.info("Execute 'RUN QUANT STRATEGY' in sidebar to run the backtest engine.")
        return

    metrics = backtest_result.metrics
    currency_symbol = "₹" if ".NS" in selected_ticker else "$"

    # 1. Performance Overview KPI Rows
    st.markdown('<div class="section-title">RISK & RETURN ATTRIBUTION</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        s_ret = metrics.get("strategy_total_return", 0.0)
        render_kpi_card("STRATEGY RETURN", format_pct(s_ret), delta_tone=tone_class(s_ret), subtext="Net of Friction")
    with c2:
        b_ret = metrics.get("buy_and_hold_total_return", metrics.get("buy_hold_total_return", 0.0))
        render_kpi_card("BUY & HOLD RETURN", format_pct(b_ret), delta_tone=tone_class(b_ret), subtext="Benchmark")
    with c3:
        alpha_val = metrics.get("strategy_vs_buy_and_hold", s_ret - b_ret)
        render_kpi_card("EXCESS RETURN (ALPHA)", format_pct(alpha_val), delta_tone=tone_class(alpha_val), subtext="Strategy - Buy&Hold")
    with c4:
        cagr = metrics.get("strategy_annualized_return", 0.0)
        render_kpi_card("ANNUALIZED RETURN (CAGR)", format_pct(cagr), delta_tone=tone_class(cagr), subtext="Compounded Annual")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        sh = metrics.get("strategy_sharpe_ratio", 0.0)
        render_kpi_card("SHARPE RATIO", format_number(sh), delta_tone=tone_class(sh), subtext="Annualized Risk-Adj")
    with c6:
        so = metrics.get("strategy_sortino_ratio", 0.0)
        render_kpi_card("SORTINO RATIO", format_number(so), delta_tone=tone_class(so), subtext="Downside Risk-Adj")
    with c7:
        mdd = metrics.get("strategy_max_drawdown", 0.0)
        render_kpi_card("MAX DRAWDOWN", f"-{mdd*100.0:.2f}%", delta_tone="negative", subtext="Peak to Trough")
    with c8:
        calmar = metrics.get("strategy_calmar_ratio", 0.0)
        render_kpi_card("CALMAR RATIO", format_number(calmar), delta_tone=tone_class(calmar), subtext="CAGR / Max DD")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Equity & Drawdown Chart
    st.markdown('<div class="section-title">EQUITY CURVE & UNDERWATER DRAWDOWN TRAJECTORY</div>', unsafe_allow_html=True)
    fig = plot_equity_and_drawdown(
        backtest_result.frame,
        title=f"{selected_ticker} ({selected_model}) - Net Portfolio Value",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Trade Execution & Friction Accounting Table
    st.markdown('<div class="section-title">TRADE EXECUTION & FRICTION METRICS</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        wr = metrics.get("win_rate", 0.0)
        render_kpi_card("WIN RATE", f"{wr*100.0:.1f}%", delta_tone="positive" if wr > 0.5 else "neutral", subtext="% Profitable Active Days")
    with t2:
        pf = metrics.get("profit_factor", 1.0)
        render_kpi_card("PROFIT FACTOR", format_number(pf), delta_tone="positive" if pf > 1.2 else "neutral", subtext="Gross Gain / Gross Loss")
    with t3:
        nt = metrics.get("total_trades", 0.0)
        render_kpi_card("TOTAL TRADES", f"{nt:.0f}", subtext="Position Transitions")
    with t4:
        costs = metrics.get("total_transaction_cost_paid", 0.0)
        render_kpi_card("TOTAL FRICTION PAID", format_currency(costs, symbol=currency_symbol), delta_tone="negative", subtext="Cost + Slippage Paid")

    # 4. Detailed Backtest Ledger
    st.markdown('<div class="section-title">DETAILED AUDIT LEDGER (TEST PERIOD)</div>', unsafe_allow_html=True)
    with st.expander("View Full Ledger Data", expanded=False):
        st.dataframe(backtest_result.frame, use_container_width=True)
        csv_data = backtest_result.frame.to_csv(index_label="Date").encode("utf-8")
        st.download_button(
            label="Download Ledger CSV",
            data=csv_data,
            file_name=f"backtest_ledger_{selected_ticker}.csv",
            mime="text/csv",
        )

