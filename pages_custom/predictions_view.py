"""Modern Predictions View for Quant Terminal."""

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
    """Compact visual cards for ML predictions."""
    st.markdown(f'<div class="view-title">PREDICTIONS & SIGNAL CONVICTION — {selected_ticker}</div>', unsafe_allow_html=True)

    # Compact unobtrusive research disclaimer
    st.markdown(
        """
        <div class="terminal-disclaimer">
            <b>NOTICE:</b> Statistical model probabilities are mathematical estimates for research purposes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    # Top Signal Cards
    c1, c2, c3, c4 = st.columns(4)
    sig_tone = "positive" if latest_signal == "BUY" else ("negative" if latest_signal == "SELL" else "neutral")

    with c1:
        render_kpi_card("ACTIVE SIGNAL", latest_signal, f"{horizon}d horizon", sig_tone)
    with c2:
        render_kpi_card("UPSIDE PROBABILITY", f"{latest_prob*100.0:.1f}%", f"{(1.0-latest_prob)*100.0:.1f}% Down", "neutral")
    with c3:
        render_kpi_card("CONVICTION", f"{confidence*100.0:.0f}%", "High" if confidence > 0.6 else "Moderate", "positive" if confidence > 0.6 else "neutral")
    with c4:
        render_kpi_card("THRESHOLD BAND", f"{sell_threshold:.2f} — {buy_threshold:.2f}", "Neutral Range", "neutral")

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    col_chart, col_eval = st.columns([1.3, 1.0])

    with col_chart:
        if feature_importance_df is not None and not feature_importance_df.empty:
            st.markdown('<div class="card-title">KEY PREDICTIVE DRIVERS</div>', unsafe_allow_html=True)
            imp_fig = plot_feature_importance_bar(feature_importance_df, title="")
            st.plotly_chart(imp_fig, use_container_width=True)

    with col_eval:
        st.markdown('<div class="card-title">TEST SET EVALUATION METRICS</div>', unsafe_allow_html=True)
        if classification_metrics:
            m_df = pd.DataFrame([
                {"Metric": "Accuracy", "Value": f"{classification_metrics.get('accuracy', 0.0):.1%}"},
                {"Metric": "Precision", "Value": f"{classification_metrics.get('precision', 0.0):.1%}"},
                {"Metric": "Recall", "Value": f"{classification_metrics.get('recall', 0.0):.1%}"},
                {"Metric": "F1 Score", "Value": f"{classification_metrics.get('f1', 0.0):.3f}"},
                {"Metric": "ROC-AUC", "Value": f"{classification_metrics.get('roc_auc', 0.50):.3f}"},
                {"Metric": "Balanced Acc", "Value": f"{classification_metrics.get('balanced_accuracy', 0.0):.1%}"},
            ])
            st.dataframe(m_df, use_container_width=True, hide_index=True, height=250)

    # Detailed Predictions Table (Collapsible)
    with st.expander("Recent Prediction Logs", expanded=False):
        if predictions_table is not None and not predictions_table.empty:
            st.dataframe(predictions_table.tail(15).iloc[::-1], use_container_width=True)
