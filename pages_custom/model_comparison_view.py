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
    """Side-by-side comparative benchmarking."""
    st.markdown(f'<div class="view-title">MODEL COMPARISON — {selected_ticker}</div>', unsafe_allow_html=True)

    if comparison_frame is None or comparison_frame.empty:
        st.info("Execute strategy pipeline to compute multi-model comparison table.")
        return

    st.markdown('<div class="card-title">CLASSIFICATION & STRATEGY BENCHMARK MATRIX</div>', unsafe_allow_html=True)
    st.dataframe(
        comparison_frame.style.format({
            "accuracy": "{:.1%}",
            "precision": "{:.1%}",
            "recall": "{:.1%}",
            "f1": "{:.3f}",
            "strategy_total_return": "{:+.1%}",
            "strategy_sharpe_ratio": "{:.2f}",
            "strategy_max_drawdown": "{:.1%}",
            "buy_hold_total_return": "{:+.1%}",
            "total_trades": "{:.0f}",
            "win_rate": "{:.0%}",
            "percent_time_in_market": "{:.0%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    if all_backtests:
        st.markdown('<div class="card-title">COMPARATIVE EQUITY TRAJECTORIES</div>', unsafe_allow_html=True)
        fig = go.Figure()
        palette = {
            "Random Forest": "#22D3EE",
            "XGBoost": "#16C784",
            "Logistic Regression": "#F59E0B",
            "Gradient Boosting": "#8B5CF6",
        }

        first_bk = next(iter(all_backtests.values()), None)
        if first_bk and hasattr(first_bk, "frame") and "equity_market" in first_bk.frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=first_bk.frame.index,
                    y=first_bk.frame["equity_market"],
                    mode="lines",
                    name="Buy & Hold",
                    line=dict(color="#8792A5", width=1.3, dash="dot"),
                )
            )

        for model_name, bk_res in all_backtests.items():
            if bk_res and hasattr(bk_res, "frame") and "equity_strategy" in bk_res.frame.columns:
                c = palette.get(model_name, "#EC4899")
                fig.add_trace(
                    go.Scatter(
                        x=bk_res.frame.index,
                        y=bk_res.frame["equity_strategy"],
                        mode="lines",
                        name=model_name,
                        line=dict(color=c, width=1.8),
                    )
                )

        fig = style_figure(fig, title="", height=400)
        st.plotly_chart(fig, use_container_width=True)
