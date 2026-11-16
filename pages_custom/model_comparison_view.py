"""Model Comparison View for Quant Terminal."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import style_figure


def render_model_comparison_view(
    selected_ticker: str,
    comparison_frame: pd.DataFrame | None,
    all_backtests: dict[str, object] | None,
) -> None:
    """Side-by-side multi-classifier comparison on holdout test set."""
    st.markdown(f'<div class="view-header">QUANT MODEL COMPARISON: {selected_ticker}</div>', unsafe_allow_html=True)
    st.caption("Chronological benchmark evaluation comparing Random Forest, XGBoost, Logistic Regression, and Gradient Boosting.")

    if comparison_frame is None or comparison_frame.empty:
        st.info("Run strategy pipeline to compute multi-model comparison table.")
        return

    # Comparison metrics table
    st.markdown('<div class="section-title">HOLDOUT TEST PERFORMANCE MATRIX</div>', unsafe_allow_html=True)
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
                "total_trades": "{:.0f}",
                "win_rate": "{:.1%}",
                "percent_time_in_market": "{:.1%}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Multi-model equity curves side-by-side
    st.markdown('<div class="section-title">MULTI-MODEL EQUITY CURVES (TEST PERIOD)</div>', unsafe_allow_html=True)
    if all_backtests:
        fig = go.Figure()
        palette = {
            "Random Forest": "#22d3ee",
            "XGBoost": "#10b981",
            "Logistic Regression": "#f59e0b",
            "Gradient Boosting": "#a855f7",
        }

        # Add Buy & Hold once
        first_bk = next(iter(all_backtests.values()), None)
        if first_bk and hasattr(first_bk, "frame") and "equity_market" in first_bk.frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=first_bk.frame.index,
                    y=first_bk.frame["equity_market"],
                    mode="lines",
                    name="Buy & Hold (Benchmark)",
                    line=dict(color="#64748b", width=1.5, dash="dot"),
                )
            )

        for model_name, bk_res in all_backtests.items():
            if bk_res and hasattr(bk_res, "frame") and "equity_strategy" in bk_res.frame.columns:
                c = palette.get(model_name, "#ec4899")
                fig.add_trace(
                    go.Scatter(
                        x=bk_res.frame.index,
                        y=bk_res.frame["equity_strategy"],
                        mode="lines",
                        name=f"{model_name}",
                        line=dict(color=c, width=2.0),
                    )
                )

        fig = style_figure(fig, title="Comparative Cumulative Strategy Trajectories", height=480)
        st.plotly_chart(fig, use_container_width=True)

    # Model Ranking & Best Model Callout
    if "strategy_sharpe_ratio" in comparison_frame.columns:
        best_by_sharpe = comparison_frame.sort_values(by="strategy_sharpe_ratio", ascending=False).iloc[0]
        best_by_return = comparison_frame.sort_values(by="strategy_total_return", ascending=False).iloc[0]

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(
                f"""
                <div class="terminal-card">
                    <div class="kpi-header">HIGHEST RISK-ADJUSTED RETURN (SHARPE)</div>
                    <div class="kpi-value positive">{best_by_sharpe['model']}</div>
                    <p class="card-copy">Sharpe Ratio: <b>{best_by_sharpe['strategy_sharpe_ratio']:.2f}</b> | Return: <b>{best_by_sharpe['strategy_total_return']*100.0:+.2f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""
                <div class="terminal-card">
                    <div class="kpi-header">HIGHEST TOTAL RETURN</div>
                    <div class="kpi-value positive">{best_by_return['model']}</div>
                    <p class="card-copy">Total Return: <b>{best_by_return['strategy_total_return']*100.0:+.2f}%</b> | Max DD: <b>-{best_by_return['strategy_max_drawdown']*100.0:.2f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

