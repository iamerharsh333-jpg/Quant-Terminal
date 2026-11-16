"""Modern Strategy Backtesting View for Quant Terminal."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import plot_equity_and_drawdown, plot_monthly_returns_heatmap
from components.metric_cards import render_kpi_card
from core.backtesting_engine import BacktestResult
from utils.formatting import format_currency, format_number, format_pct, tone_class


def render_backtesting_view(
    selected_ticker: str,
    selected_model: str,
    backtest_result: BacktestResult | None,
    initial_capital: float = 100_000.0,
) -> None:
    """Clean performance overview with monthly return heatmap and trade metrics."""
    st.markdown(f'<div class="view-title">STRATEGY BACKTESTING — {selected_ticker} ({selected_model})</div>', unsafe_allow_html=True)

    if backtest_result is None or backtest_result.frame.empty:
        st.info("Execute 'RUN STRATEGY' in the sidebar to generate backtest performance.")
        return

    m = backtest_result.metrics
    s_ret = m.get("strategy_total_return", 0.0)
    cagr = m.get("strategy_annualized_return", 0.0)
    sh = m.get("strategy_sharpe_ratio", 0.0)
    so = m.get("strategy_sortino_ratio", 0.0)
    mdd = m.get("strategy_max_drawdown", 0.0)
    wr = m.get("win_rate", 0.0)

    # 6 Standard KPI Cards
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        render_kpi_card("TOTAL RETURN", format_pct(s_ret), delta_tone=tone_class(s_ret))
    with k2:
        render_kpi_card("CAGR", format_pct(cagr), delta_tone=tone_class(cagr))
    with k3:
        render_kpi_card("SHARPE", format_number(sh), delta_tone=tone_class(sh))
    with k4:
        render_kpi_card("SORTINO", format_number(so), delta_tone=tone_class(so))
    with k5:
        render_kpi_card("MAX DRAWDOWN", f"-{mdd*100.0:.1f}%", delta_tone="negative")
    with k6:
        render_kpi_card("WIN RATE", f"{wr*100.0:.0f}%", delta_tone="positive" if wr > 0.5 else "neutral")

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    # Strategy vs Buy & Hold + Drawdown Chart
    fig_eq = plot_equity_and_drawdown(
        backtest_result.frame,
        title=f"{selected_ticker} Equity Curve vs Buy & Hold",
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    # Monthly Returns Heatmap & Execution Summary
    col_heat, col_exec = st.columns([1.4, 1.0])

    with col_heat:
        if "strategy_return" in backtest_result.frame.columns:
            fig_hm = plot_monthly_returns_heatmap(backtest_result.frame["strategy_return"])
            st.plotly_chart(fig_hm, use_container_width=True)

    with col_exec:
        st.markdown('<div class="card-title">EXECUTION & FRICTION LEDGER</div>', unsafe_allow_html=True)
        nt = m.get("total_trades", 0.0)
        pf = m.get("profit_factor", 1.0)
        costs = m.get("total_transaction_cost_paid", 0.0)
        currency_sym = "₹" if ".NS" in selected_ticker else "$"

        exec_df = pd.DataFrame([
            {"Parameter": "Total Trades", "Value": f"{nt:.0f}"},
            {"Parameter": "Profit Factor", "Value": f"{pf:.2f}"},
            {"Parameter": "Time in Market", "Value": f"{m.get('percent_time_in_market', 0.0):.1%}"},
            {"Parameter": "Friction Paid", "Value": format_currency(costs, symbol=currency_sym)},
            {"Parameter": "Benchmark Return", "Value": format_pct(m.get('buy_and_hold_total_return', 0.0))},
            {"Parameter": "Net Alpha", "Value": format_pct(m.get('strategy_vs_buy_and_hold', 0.0))},
        ])
        st.dataframe(exec_df, use_container_width=True, hide_index=True, height=220)

    # Trade History Table (Collapsible)
    with st.expander("Detailed Trade Audit Ledger", expanded=False):
        st.dataframe(backtest_result.frame, use_container_width=True)
        csv_data = backtest_result.frame.to_csv(index_label="Date").encode("utf-8")
        st.download_button(
            label="Download Ledger CSV",
            data=csv_data,
            file_name=f"backtest_ledger_{selected_ticker}.csv",
            mime="text/csv",
        )
