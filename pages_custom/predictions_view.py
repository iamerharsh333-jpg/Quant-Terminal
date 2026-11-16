"""ML Predictions View for Quant Terminal."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import plot_feature_importance_bar, style_figure
from components.metric_cards import render_kpi_card
from utils.formatting import format_pct


def render_predictions_view(
    selected_ticker: str,
    selected_model: str,
    horizon: int,
    buy_threshold: float,
    sell_threshold: float,
    latest_signal: str,
    latest_prob: float,
    confidence: float,
    classification_metrics: dict[str, float] | None,
    feature_importance_df: pd.DataFrame | None,
    predictions_table: pd.DataFrame | None,
) -> None:
    """Render quantitative machine learning prediction diagnostics and interpretations."""
    st.markdown(f'<div class="view-header">MACHINE LEARNING PREDICTIONS: {selected_ticker}</div>', unsafe_allow_html=True)
    st.caption(f"Model: {selected_model} | Forecast Horizon: {horizon} Trading Day(s) | Directional Up/Down Classification")

    # Disclaimer Callout
    st.markdown(
        """
        <div class="terminal-disclaimer">
            <b>STATISTICAL RESEARCH NOTICE:</b> Model probabilities and signals are mathematical estimates
            derived from historical patterns. They do not represent guaranteed trading outcomes, investment advice,
            or automated trade execution. Strictly manage risk.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Top Signal Panel
    col_sig, col_prob, col_conf, col_thresh = st.columns(4)
    sig_tone = "positive" if latest_signal == "BUY" else ("negative" if latest_signal == "SELL" else "neutral")

    with col_sig:
        render_kpi_card(
            label="LATEST SIGNAL",
            value=latest_signal,
            delta="Active Position",
            delta_tone=sig_tone,
            subtext=f"Next {horizon}-day Outlook",
        )
    with col_prob:
        render_kpi_card(
            label="UPSIDE PROBABILITY",
            value=f"{latest_prob * 100.0:.1f}%",
            delta=f"{(1.0 - latest_prob) * 100.0:.1f}% Down",
            delta_tone="neutral",
            subtext="Estimated Directional Mass",
        )
    with col_conf:
        render_kpi_card(
            label="SIGNAL CONVICTION",
            value=f"{confidence * 100.0:.1f}%",
            delta="Reliability",
            delta_tone="positive" if confidence > 0.5 else "neutral",
            subtext="Distance from Threshold",
        )
    with col_thresh:
        render_kpi_card(
            label="HYSTERESIS BAND",
            value=f"{sell_threshold:.2f} - {buy_threshold:.2f}",
            subtext="Neutral No-Trade Range",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability Distribution Meter
    col_meter, col_metrics = st.columns([1.2, 1.0])

    with col_meter:
        st.markdown('<div class="section-title">PROBABILITY MASS ALLOCATION</div>', unsafe_allow_html=True)
        prob_fig = go.Figure()
        prob_fig.add_trace(
            go.Bar(
                x=[latest_prob * 100.0],
                y=["Direction"],
                orientation="h",
                name="Upside Prob (%)",
                marker_color="#10b981",
                text=f"{latest_prob * 100.0:.1f}% Upside",
                textposition="inside",
            )
        )
        prob_fig.add_trace(
            go.Bar(
                x=[(1.0 - latest_prob) * 100.0],
                y=["Direction"],
                orientation="h",
                name="Downside Prob (%)",
                marker_color="#ef4444",
                text=f"{(1.0 - latest_prob) * 100.0:.1f}% Downside",
                textposition="inside",
            )
        )
        prob_fig.update_layout(
            barmode="stack",
            xaxis=dict(range=[0, 100], title="Probability (%)"),
            yaxis=dict(showticklabels=False),
            height=180,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        prob_fig = style_figure(prob_fig, title="Probability Meter", height=180)
        st.plotly_chart(prob_fig, use_container_width=True)

        # Feature Importance Chart
        if feature_importance_df is not None and not feature_importance_df.empty:
            st.markdown('<div class="section-title">TOP PREDICTIVE FEATURE DRIVERS</div>', unsafe_allow_html=True)
            imp_fig = plot_feature_importance_bar(
                feature_importance_df,
                title=f"{selected_model} - Key Model Drivers",
            )
            st.plotly_chart(imp_fig, use_container_width=True)

    with col_metrics:
        st.markdown('<div class="section-title">CHRONOLOGICAL TEST-SET EVALUATION</div>', unsafe_allow_html=True)
        if classification_metrics:
            m1, m2 = st.columns(2)
            with m1:
                render_kpi_card(
                    label="ACCURACY",
                    value=format_pct(classification_metrics.get("accuracy", 0.0), show_sign=False),
                    subtext="Test Set (Holdout)",
                )
                render_kpi_card(
                    label="PRECISION",
                    value=format_pct(classification_metrics.get("precision", 0.0), show_sign=False),
                    subtext="Positive Predictive Value",
                )
                render_kpi_card(
                    label="BALANCED ACCURACY",
                    value=format_pct(classification_metrics.get("balanced_accuracy", 0.0), show_sign=False),
                    subtext="Class Balanced",
                )
            with m2:
                render_kpi_card(
                    label="F1 SCORE",
                    value=f"{classification_metrics.get('f1', 0.0):.3f}",
                    subtext="Harmonic Mean",
                )
                render_kpi_card(
                    label="RECALL",
                    value=format_pct(classification_metrics.get("recall", 0.0), show_sign=False),
                    subtext="Sensitivity",
                )
                render_kpi_card(
                    label="ROC-AUC",
                    value=f"{classification_metrics.get('roc_auc', 0.50):.3f}",
                    subtext="Discrimination Power",
                )

    # Detailed Recent Predictions Log
    if predictions_table is not None and not predictions_table.empty:
        st.markdown('<div class="section-title">RECENT MODEL SIGNALS & PREDICTION LEDGER</div>', unsafe_allow_html=True)
        display_preds = predictions_table.tail(20).iloc[::-1]
        st.dataframe(display_preds, use_container_width=True)

