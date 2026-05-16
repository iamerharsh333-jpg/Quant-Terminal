"""Premium Streamlit dashboard for the stock prediction project."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import run_backtest
from data import RELIANCE_END, RELIANCE_START, RELIANCE_SYMBOL, load_price_data
from features import DEFAULT_HORIZON, build_feature_frame, make_features
from model import (
    evaluate_classifier,
    get_logistic_regression_coefficients,
    get_random_forest_feature_importance,
    is_xgboost_available,
    predict,
    predict_proba,
    predict_signals,
    train_logistic_regression_model,
    train_xgboost_model,
    time_series_train_test_split,
    train_model,
)
from utils.recommendations import build_recommendation

DEFAULT_TRANSACTION_COST = 0.001
DEFAULT_SLIPPAGE = 0.0005
DEFAULT_POSITION_SIZE = 1.0
DEFAULT_EXECUTION_LAG = 1
BUY_THRESHOLD = 0.6
SELL_THRESHOLD = 0.4
STOCK_TICKER_OPTIONS = {
    "Reliance Industries (RELIANCE.NS)": RELIANCE_SYMBOL,
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Custom ticker": None,
}
MODEL_OPTIONS = ["Random Forest", "XGBoost", "Logistic Regression"]
FORECAST_HORIZON_OPTIONS = [1, 3, 5, 10]

st.set_page_config(
    page_title="Quant Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_dashboard_styles() -> None:
    """Apply a premium dark fintech theme without changing Streamlit behavior."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-primary: #07111f;
            --bg-secondary: #0b1628;
            --card-bg: rgba(15, 23, 42, 0.74);
            --card-border: rgba(148, 163, 184, 0.18);
            --text-primary: #f8fafc;
            --text-secondary: #a7b3c7;
            --cyan: #22d3ee;
            --teal: #2dd4bf;
            --green: #22c55e;
            --amber: #f59e0b;
            --red: #ef4444;
            --blue: #60a5fa;
        }

        html, body, [class*="css"] {
            font-family: "Manrope", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(34, 211, 238, 0.16), transparent 30rem),
                radial-gradient(circle at 88% 16%, rgba(45, 212, 191, 0.12), transparent 28rem),
                linear-gradient(135deg, #050914 0%, #07111f 45%, #0c1729 100%);
            color: var(--text-primary);
        }

        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2.75rem;
            max-width: 1440px;
        }

        h1, h2, h3, h4, h5, h6, p, label, span {
            color: inherit;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(7, 17, 31, 0.98) 0%, rgba(9, 19, 35, 0.96) 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.16);
            box-shadow: 16px 0 48px rgba(2, 6, 23, 0.25);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] p {
            color: #dbeafe;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(148, 163, 184, 0.16);
        }

        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.66));
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            box-shadow: 0 18px 44px rgba(2, 6, 23, 0.24);
        }

        [data-testid="stMetricLabel"] p {
            color: var(--text-secondary);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        [data-testid="stMetricValue"] {
            color: var(--text-primary);
            font-size: 1.45rem;
            font-weight: 800;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 18px 44px rgba(2, 6, 23, 0.22);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.80), rgba(15, 23, 42, 0.52));
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            box-shadow: 0 24px 70px rgba(2, 6, 23, 0.26);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
        }

        .stButton > button,
        [data-testid="stDownloadButton"] button {
            border: 1px solid rgba(34, 211, 238, 0.38);
            border-radius: 14px;
            background: linear-gradient(135deg, #06b6d4 0%, #14b8a6 100%);
            color: #03111d;
            font-weight: 800;
            letter-spacing: 0.01em;
            box-shadow: 0 14px 32px rgba(20, 184, 166, 0.22);
            transition: all 180ms ease;
        }

        .stButton > button:hover,
        [data-testid="stDownloadButton"] button:hover {
            border-color: rgba(125, 211, 252, 0.8);
            transform: translateY(-1px);
            box-shadow: 0 20px 42px rgba(34, 211, 238, 0.28);
        }

        button[data-baseweb="tab"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 14px 14px 0 0;
            color: #a7b3c7;
            font-weight: 800;
            padding: 0.72rem 1.05rem;
            margin-right: 0.35rem;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #ecfeff;
            background: linear-gradient(135deg, rgba(8, 145, 178, 0.54), rgba(20, 184, 166, 0.32));
            border-color: rgba(34, 211, 238, 0.42);
            box-shadow: 0 10px 28px rgba(8, 145, 178, 0.20);
        }

        div[data-testid="stAlert"] {
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 18px 44px rgba(2, 6, 23, 0.18);
        }

        .dashboard-kicker {
            color: #67e8f9;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .dashboard-title {
            color: var(--text-primary);
            font-size: clamp(2.15rem, 4vw, 4.1rem);
            font-weight: 850;
            letter-spacing: -0.06em;
            line-height: 1.1;
            margin-bottom: 0.25rem;
        }

        .dashboard-subtitle {
            color: var(--text-secondary);
            font-size: 1.02rem;
            margin-bottom: 0.3rem;
            max-width: 760px;
        }

        .glass-card {
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.52));
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            box-shadow: 0 24px 70px rgba(2, 6, 23, 0.34);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
        }

        .hero-header {
            position: relative;
            overflow: hidden;
            padding: 1.35rem 1.45rem;
            margin-bottom: 1.05rem;
        }

        .hero-header::after {
            content: "";
            position: absolute;
            width: 26rem;
            height: 26rem;
            right: -10rem;
            top: -12rem;
            background: radial-gradient(circle, rgba(34, 211, 238, 0.28), transparent 60%);
            pointer-events: none;
        }

        .header-meta {
            display: flex;
            gap: 0.7rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .meta-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.46rem 0.72rem;
            border-radius: 999px;
            color: #dbeafe;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            font-size: 0.82rem;
            font-weight: 700;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.78rem;
            padding: 0.9rem 0.75rem 1rem;
            margin-bottom: 0.8rem;
            border-radius: 20px;
            background: linear-gradient(145deg, rgba(34, 211, 238, 0.14), rgba(45, 212, 191, 0.08));
            border: 1px solid rgba(148, 163, 184, 0.16);
        }

        .sidebar-logo {
            display: grid;
            place-items: center;
            width: 2.75rem;
            height: 2.75rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #22d3ee, #2dd4bf);
            color: #03111d;
            font-weight: 900;
            box-shadow: 0 12px 26px rgba(34, 211, 238, 0.22);
        }

        .sidebar-title {
            font-size: 1.02rem;
            line-height: 1.1;
            color: #f8fafc;
            font-weight: 850;
        }

        .sidebar-caption {
            font-size: 0.74rem;
            color: #93a4ba;
            margin-top: 0.18rem;
        }

        .section-title {
            color: #ecfeff;
            font-size: 1.08rem;
            font-weight: 850;
            letter-spacing: -0.02em;
            margin: 0 0 0.55rem;
        }

        .section-copy {
            color: var(--text-secondary);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .premium-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 0.2rem 0 1.05rem;
        }

        .premium-kpi-card {
            padding: 1.05rem;
            min-height: 142px;
            transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        }

        .premium-kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(34, 211, 238, 0.34);
            box-shadow: 0 30px 80px rgba(2, 6, 23, 0.44);
        }

        .kpi-label {
            color: #9fb0c5;
            font-size: 0.74rem;
            font-weight: 850;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .kpi-value {
            margin-top: 0.42rem;
            color: var(--text-primary);
            font-size: clamp(1.55rem, 2.3vw, 2.25rem);
            font-weight: 900;
            letter-spacing: -0.045em;
        }

        .kpi-delta {
            margin-top: 0.42rem;
            color: #9fb0c5;
            font-size: 0.86rem;
            font-weight: 700;
        }

        .tone-positive { color: var(--green); }
        .tone-negative { color: var(--red); }
        .tone-neutral { color: var(--amber); }
        .tone-info { color: var(--cyan); }

        .signal-panel {
            padding: 1.1rem;
            min-height: 100%;
        }

        .signal-topline {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 0.9rem;
        }

        .signal-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 9rem;
            padding: 0.7rem 1rem;
            border-radius: 999px;
            font-size: 1.55rem;
            font-weight: 950;
            letter-spacing: 0.02em;
        }

        .signal-buy {
            color: #04210f;
            background: linear-gradient(135deg, #86efac, #22c55e);
            box-shadow: 0 18px 42px rgba(34, 197, 94, 0.26);
        }

        .signal-sell {
            color: #2b0505;
            background: linear-gradient(135deg, #fca5a5, #ef4444);
            box-shadow: 0 18px 42px rgba(239, 68, 68, 0.25);
        }

        .signal-hold {
            color: #2b1a03;
            background: linear-gradient(135deg, #fde68a, #f59e0b);
            box-shadow: 0 18px 42px rgba(245, 158, 11, 0.24);
        }

        .signal-stat {
            color: #dbeafe;
            font-size: 0.88rem;
            font-weight: 800;
            text-align: right;
        }

        .signal-stat span {
            display: block;
            color: #94a3b8;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.22rem;
        }

        .signal-explanation {
            color: #b7c4d6;
            font-size: 0.9rem;
            line-height: 1.55;
            margin-top: 0.75rem;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.72rem;
            margin-top: 0.9rem;
        }

        .mini-card {
            padding: 0.82rem;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.64);
            border: 1px solid rgba(148, 163, 184, 0.14);
        }

        .mini-label {
            color: #94a3b8;
            font-size: 0.7rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .mini-value {
            color: #f8fafc;
            font-size: 1.08rem;
            font-weight: 850;
            margin-top: 0.25rem;
        }

        .about-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.9rem;
        }

        .about-card {
            padding: 1rem;
        }

        .about-card h4 {
            color: #ecfeff;
            margin: 0 0 0.45rem;
            font-size: 1rem;
        }

        .about-card p {
            color: #a7b3c7;
            margin: 0;
            line-height: 1.55;
            font-size: 0.9rem;
        }

        @media (max-width: 980px) {
            .premium-kpi-grid,
            .about-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 680px) {
            .premium-kpi-grid,
            .mini-grid,
            .about-grid {
                grid-template-columns: 1fr;
            }

            .signal-topline {
                flex-direction: column;
            }

            .signal-stat {
                text-align: left;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _format_number(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def _tone_class(value: float, positive_good: bool = True) -> str:
    """Return CSS tone class for financial values."""
    if pd.isna(value) or float(value) == 0.0:
        return "tone-neutral"
    is_positive = float(value) > 0.0
    if positive_good:
        return "tone-positive" if is_positive else "tone-negative"
    return "tone-negative" if is_positive else "tone-positive"


def _trend_arrow(value: float, positive_good: bool = True) -> str:
    if pd.isna(value) or float(value) == 0.0:
        return "→"
    is_positive = float(value) > 0.0
    if positive_good:
        return "▲" if is_positive else "▼"
    return "▼" if is_positive else "▲"


def _signal_tone_class(signal: str) -> str:
    normalized = signal.lower()
    if normalized == "buy":
        return "tone-positive"
    if normalized == "sell":
        return "tone-negative"
    return "tone-neutral"


def _signal_badge_class(signal: str) -> str:
    normalized = signal.lower()
    if normalized == "buy":
        return "signal-buy"
    if normalized == "sell":
        return "signal-sell"
    return "signal-hold"


def _signal_confidence(
    probability_up: float,
    signal: str,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
) -> float:
    """Convert the thresholded signal rule into a user-friendly confidence score."""
    if signal == "BUY":
        return probability_up
    if signal == "SELL":
        return 1.0 - probability_up

    midpoint = (buy_threshold + sell_threshold) / 2.0
    half_band = (buy_threshold - sell_threshold) / 2.0
    if half_band <= 0:
        return 0.0
    confidence = 1.0 - (abs(probability_up - midpoint) / half_band)
    return float(max(0.0, min(1.0, confidence)))


def _risk_level(confidence: float) -> str:
    """Classify signal risk from confidence, where lower confidence means higher risk."""
    if confidence >= 0.75:
        return "Low"
    if confidence >= 0.60:
        return "Medium"
    return "High"


def _render_trading_signal_alert(
    signal: str,
    confidence: float,
    risk_level: str,
) -> None:
    """Render the signal summary with Streamlit alert color by signal direction."""
    message = (
        f"Current Signal: **{signal}** | "
        f"Confidence: **{_format_pct(confidence)}** | "
        f"Risk Level: **{risk_level}**"
    )
    if signal == "BUY":
        st.success(message)
    elif signal == "SELL":
        st.error(message)
    else:
        st.warning(message)


def _render_dashboard_header(
    stock_ticker: str,
    selected_model: str,
    forecast_horizon: int,
    buy_threshold: float,
    sell_threshold: float,
) -> None:
    """Render the premium dashboard masthead."""
    timestamp = datetime.now().astimezone().strftime("%d %b %Y, %H:%M %Z")
    st.markdown(
        f"""
        <div class="glass-card hero-header">
            <div class="dashboard-kicker">Quant Research Terminal</div>
            <div class="dashboard-title">AI-Powered Quant Trading Dashboard</div>
            <div class="dashboard-subtitle">
                Machine Learning Based Stock Prediction &amp; Backtesting Platform
            </div>
            <div class="header-meta">
                <span class="meta-pill">Selected Stock: {escape(stock_ticker or "None")}</span>
                <span class="meta-pill">Model: {escape(selected_model)}</span>
                <span class="meta-pill">Horizon: {forecast_horizon} trading day(s)</span>
                <span class="meta-pill">Signal Rule: {buy_threshold:.2f} / {sell_threshold:.2f}</span>
                <span class="meta-pill">Timestamp: {escape(timestamp)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section_intro(title: str, copy: str | None = None) -> None:
    """Render compact section heading text."""
    copy_html = f'<div class="section-copy">{escape(copy)}</div>' if copy else ""
    st.markdown(
        f"""
        <div style="margin: 0.6rem 0 0.75rem;">
            <div class="section-title">{escape(title)}</div>
            {copy_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_hero_kpis(model_metrics: pd.Series, latest_signal: str) -> None:
    """Render the four primary KPIs as premium glass cards."""
    strategy_return = float(model_metrics["strategy_return"])
    market_return = float(model_metrics["market_return"])
    sharpe_ratio = float(model_metrics["sharpe_ratio"])
    accuracy = float(model_metrics["accuracy"])
    excess_return = strategy_return - market_return
    accuracy_edge = accuracy - 0.5
    signal_tone = _signal_tone_class(latest_signal)

    cards = [
        {
            "label": "Strategy Return",
            "value": _format_pct(strategy_return),
            "tone": _tone_class(strategy_return),
            "delta": f"{_trend_arrow(excess_return)} {_format_pct(excess_return)} vs market",
        },
        {
            "label": "Sharpe Ratio",
            "value": _format_number(sharpe_ratio, 3),
            "tone": _tone_class(sharpe_ratio),
            "delta": f"{_trend_arrow(sharpe_ratio)} Risk-adjusted performance",
        },
        {
            "label": "Accuracy",
            "value": _format_pct(accuracy),
            "tone": _tone_class(accuracy_edge),
            "delta": f"{_trend_arrow(accuracy_edge)} {_format_pct(accuracy_edge)} vs 50% baseline",
        },
        {
            "label": "Current Signal",
            "value": latest_signal,
            "tone": signal_tone,
            "delta": "Probability-threshold trade state",
        },
    ]

    cards_html = "".join(
        f"""
        <div class="glass-card premium-kpi-card">
            <div class="kpi-label">{escape(card["label"])}</div>
            <div class="kpi-value {card["tone"]}">{escape(card["value"])}</div>
            <div class="kpi-delta">{escape(card["delta"])}</div>
        </div>
        """
        for card in cards
    )
    st.markdown(
        f'<div class="premium-kpi-grid">{cards_html}</div>',
        unsafe_allow_html=True,
    )


def _signal_explanation(
    signal: str,
    probability_up: float,
    buy_threshold: float,
    sell_threshold: float,
    forecast_horizon: int,
) -> str:
    if signal == "BUY":
        return (
            f"The model estimates upside probability at {_format_pct(probability_up)}, "
            f"above the buy threshold of {buy_threshold:.2f}. The strategy is willing "
            f"to hold risk for the next {forecast_horizon} trading day(s)."
        )
    if signal == "SELL":
        return (
            f"The model estimates upside probability at {_format_pct(probability_up)}, "
            f"below the sell threshold of {sell_threshold:.2f}. The strategy avoids "
            "long exposure and moves to cash under the current rules."
        )
    return (
        f"The model estimates upside probability at {_format_pct(probability_up)}, "
        f"inside the no-trade band between {sell_threshold:.2f} and {buy_threshold:.2f}. "
        "The dashboard marks this as HOLD to avoid low-confidence trades."
    )


def _render_trading_signal_panel(
    signal: str,
    confidence: float,
    risk_level: str,
    probability_up: float,
    selected_model: str,
    buy_threshold: float,
    sell_threshold: float,
    latest_date: pd.Timestamp,
    forecast_horizon: int,
) -> None:
    """Render a high-emphasis trading signal card."""
    probability_pct = max(0.0, min(100.0, probability_up * 100.0))
    explanation = _signal_explanation(
        signal,
        probability_up,
        buy_threshold,
        sell_threshold,
        forecast_horizon,
    )
    st.markdown(
        f"""
        <div class="glass-card signal-panel">
            <div class="signal-topline">
                <div>
                    <div class="kpi-label">Current Trading Signal</div>
                    <div class="signal-badge {_signal_badge_class(signal)}">{escape(signal)}</div>
                </div>
                <div class="signal-stat">
                    <span>Confidence</span>{_format_pct(confidence)}
                    <br><br>
                    <span>Risk Level</span>{escape(risk_level)}
                </div>
            </div>
            <div class="mini-grid">
                <div class="mini-card">
                    <div class="mini-label">Probability Up</div>
                    <div class="mini-value">{_format_pct(probability_up)}</div>
                </div>
                <div class="mini-card">
                    <div class="mini-label">Latest Feature Date</div>
                    <div class="mini-value">{latest_date.strftime("%Y-%m-%d")}</div>
                </div>
                <div class="mini-card">
                    <div class="mini-label">Primary Model</div>
                    <div class="mini-value">{escape(selected_model)}</div>
                </div>
                <div class="mini-card">
                    <div class="mini-label">Threshold Band</div>
                    <div class="mini-value">{buy_threshold:.2f} / {sell_threshold:.2f}</div>
                </div>
            </div>
            <div style="margin-top: 1rem;">
                <div class="mini-label">Probability Progress</div>
                <div style="height: 12px; border-radius: 999px; overflow: hidden; background: rgba(148, 163, 184, 0.18); margin-top: 0.42rem;">
                    <div style="height: 100%; width: {probability_pct:.2f}%; border-radius: 999px; background: linear-gradient(90deg, #22d3ee, #2dd4bf);"></div>
                </div>
            </div>
            <div class="signal-explanation">{escape(explanation)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_about_project_content() -> None:
    """Render the project explanation cards."""
    st.markdown(
        """
        <div class="about-grid">
            <div class="glass-card about-card">
                <h4>Project Objective</h4>
                <p>Build a machine learning dashboard that predicts short-horizon stock direction, converts probabilities into BUY, SELL, or HOLD signals, and evaluates strategy behavior against buy-and-hold.</p>
            </div>
            <div class="glass-card about-card">
                <h4>Models Used</h4>
                <p>RandomForest with GridSearchCV, XGBoost, and Logistic Regression are trained on the same time-based split so model performance can be compared consistently.</p>
            </div>
            <div class="glass-card about-card">
                <h4>Methodology</h4>
                <p>The system downloads OHLCV data, engineers momentum, trend, volatility, ATR, Bollinger Band, MACD, RSI, and volume features, then forecasts forward price direction over the selected horizon.</p>
            </div>
            <div class="glass-card about-card">
                <h4>Backtesting Logic</h4>
                <p>Predicted probabilities are converted into threshold-based signals. The backtest applies transaction costs, position sizing, HOLD support, and compares equity curves against market exposure.</p>
            </div>
            <div class="glass-card about-card">
                <h4>Limitations</h4>
                <p>This is a research tool, not financial advice. Results can be sensitive to data quality, market regime shifts, transaction assumptions, overfitting, and survivorship or selection bias.</p>
            </div>
            <div class="glass-card about-card">
                <h4>Future Scope</h4>
                <p>Possible upgrades include walk-forward validation, portfolio optimization, live data feeds, risk budgeting, trade logs, explainability reports, and broker-paper-trading integration.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_recommendation_card(recommendation: object) -> None:
    """Render the recommendation engine output as a professional explanation card."""
    reasons_html = "".join(
        f"<li>{escape(reason)}</li>" for reason in recommendation.reasons
    )
    risks_html = "".join(
        f"<li>{escape(note)}</li>" for note in recommendation.risk_notes
    )
    st.markdown(
        f"""
        <div class="glass-card about-card">
            <div class="kpi-label">Recommendation Engine</div>
            <div class="kpi-value {_signal_tone_class(recommendation.action)}">
                Recommended Action: {escape(recommendation.action)}
            </div>
            <p class="section-copy">{escape(recommendation.summary)}</p>
            <div class="mini-grid">
                <div class="mini-card">
                    <div class="mini-label">Reasons</div>
                    <ul class="section-copy" style="margin: 0.45rem 0 0; padding-left: 1.1rem;">{reasons_html}</ul>
                </div>
                <div class="mini-card">
                    <div class="mini-label">Risk Notes</div>
                    <ul class="section-copy" style="margin: 0.45rem 0 0; padding-left: 1.1rem;">{risks_html}</ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _is_invalid_ticker_error(exc: Exception) -> bool:
    """Identify errors that usually mean the ticker symbol is invalid."""
    error_text = str(exc)
    return (
        isinstance(exc, ValueError)
        and "No price data returned for symbol=" in error_text
    )


def _style_plotly_figure(fig: go.Figure, title: str, height: int = 520) -> go.Figure:
    """Apply a dark trading-dashboard style to Plotly figures."""
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=height,
        margin=dict(l=28, r=28, t=72, b=38),
        paper_bgcolor="rgba(15, 23, 42, 0)",
        plot_bgcolor="#08111f",
        font=dict(color="#e5e7eb", family="Manrope, sans-serif"),
        hovermode="x unified",
        dragmode="zoom",
        title_font=dict(size=20, color="#ecfeff"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.45)",
            bordercolor="rgba(148, 163, 184, 0.18)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.13)",
        zerolinecolor="rgba(148, 163, 184, 0.15)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.24)",
        tickfont=dict(color="#a7b3c7"),
    )
    fig.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.13)",
        zerolinecolor="rgba(148, 163, 184, 0.15)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.24)",
        tickfont=dict(color="#a7b3c7"),
    )
    return fig


def _plot_equity_curve(
    random_forest_frame: pd.DataFrame,
    logistic_regression_frame: pd.DataFrame,
    ticker: str,
    xgboost_frame: pd.DataFrame | None = None,
) -> go.Figure:
    """Plot all strategy equity curves and the market baseline with Plotly."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=random_forest_frame.index,
            y=random_forest_frame["equity_curve"],
            mode="lines",
            name="Random Forest",
            line=dict(color="#2dd4bf", width=2.6),
        )
    )
    if xgboost_frame is not None:
        fig.add_trace(
            go.Scatter(
                x=xgboost_frame.index,
                y=xgboost_frame["equity_curve"],
                mode="lines",
                name="XGBoost",
                line=dict(color="#fb7185", width=2.4),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=logistic_regression_frame.index,
            y=logistic_regression_frame["equity_curve"],
            mode="lines",
            name="Logistic Regression",
            line=dict(color="#f59e0b", width=2.3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=random_forest_frame.index,
            y=random_forest_frame["buy_hold_equity_curve"],
            mode="lines",
            name="Market",
            line=dict(color="#60a5fa", width=2.4, dash="dot"),
        )
    )
    return _style_plotly_figure(fig, f"Strategy vs Market Equity Curve: {ticker}")


def _plot_candlestick(prices: pd.DataFrame, ticker: str) -> go.Figure:
    """Render an interactive OHLC candlestick chart."""
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=prices.index,
                open=prices["Open"],
                high=prices["High"],
                low=prices["Low"],
                close=prices["Close"],
                name=ticker,
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
                increasing_fillcolor="#22c55e",
                decreasing_fillcolor="#ef4444",
            )
        ]
    )
    fig.update_layout(
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_bgcolor="rgba(15, 23, 42, 0.68)",
        yaxis_title="Price",
    )
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="rgba(15, 23, 42, 0.72)",
            activecolor="#14b8a6",
            font=dict(color="#dbeafe"),
        )
    )
    return _style_plotly_figure(fig, f"OHLC Price Action: {ticker}", height=560)


def _plot_feature_importance(
    frame: pd.DataFrame,
    title: str,
    value_col: str,
) -> go.Figure:
    """Render an interactive horizontal feature-importance bar chart."""
    plot_frame = frame.sort_values(value_col, ascending=False).head(20)
    plot_frame = plot_frame.sort_values(value_col, ascending=True)
    fig = go.Figure(
        go.Bar(
            x=plot_frame[value_col],
            y=plot_frame["feature"],
            orientation="h",
            marker=dict(color="#2dd4bf"),
            hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title=value_col.replace("_", " ").title(), yaxis_title="")
    return _style_plotly_figure(fig, title, height=620)


def _maybe_get_best_params(model: object) -> dict[str, object] | None:
    best_params = getattr(model, "best_params_", None)
    if isinstance(best_params, dict):
        return best_params
    return None


def _build_comparison_frame(model_results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for model_result in model_results:
        classification_metrics = model_result["classification_metrics"]
        backtest_metrics = model_result["backtest_result"].metrics
        rows.append(
            {
                "model": model_result["model_name"],
                "accuracy": classification_metrics["accuracy"],
                "precision": classification_metrics["precision"],
                "recall": classification_metrics["recall"],
                "f1": classification_metrics["f1"],
                "strategy_return": backtest_metrics["strategy_total_return"],
                "market_return": backtest_metrics["buy_hold_total_return"],
                "sharpe_ratio": backtest_metrics["strategy_sharpe_ratio"],
                "max_drawdown": backtest_metrics["strategy_max_drawdown"],
                "annualized_return": backtest_metrics["strategy_annualized_return"],
            }
        )
    return pd.DataFrame(rows)


def _get_selected_model_metrics(
    comparison_frame: pd.DataFrame,
    selected_model: str,
) -> pd.Series:
    """Return the comparison row for the model selected in the sidebar."""
    matches = comparison_frame.loc[comparison_frame["model"] == selected_model]
    if matches.empty:
        return comparison_frame.iloc[0]
    return matches.iloc[0]


def _render_metric_cards(model_metrics: pd.Series) -> None:
    """Render the primary selected-model KPIs as dashboard cards."""
    metric_cols = st.columns(5)
    strategy_return = float(model_metrics["strategy_return"])
    market_return = float(model_metrics["market_return"])
    metric_cols[0].metric(
        "Strategy Return",
        _format_pct(strategy_return),
        delta=_format_pct(strategy_return - market_return),
    )
    metric_cols[1].metric("Market Return", _format_pct(market_return))
    metric_cols[2].metric("Sharpe Ratio", f"{model_metrics['sharpe_ratio']:.3f}")
    metric_cols[3].metric("Max Drawdown", _format_pct(model_metrics["max_drawdown"]))
    metric_cols[4].metric("Accuracy", _format_pct(model_metrics["accuracy"]))


def _prediction_label(value: int) -> str:
    return "UP" if int(value) == 1 else "DOWN"


def _prepare_predictions_table(predictions: pd.DataFrame, rows: int = 20) -> pd.DataFrame:
    """Format recent predictions for dashboard display."""
    frame = predictions.tail(rows).reset_index()
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    predicted_probability = frame["predicted_probability"].astype(float)
    confidence = predicted_probability.where(
        frame["predicted_class"].astype(int) == 1,
        1.0 - predicted_probability,
    )
    display = pd.DataFrame(
        {
            "Date": pd.to_datetime(frame[date_col]).dt.strftime("%Y-%m-%d"),
            "Prediction": frame["predicted_class"].map(_prediction_label),
            "Confidence": confidence.map(_format_pct),
            "Actual Movement": frame["actual_target"].map(_prediction_label),
            "Trade Signal": frame["predicted_signal"],
        }
    )
    return display


def _portfolio_analytics(backtest_frame: pd.DataFrame, metrics: dict[str, float]) -> dict[str, float]:
    """Compute simple portfolio analytics from the selected backtest."""
    active_returns = backtest_frame.loc[
        backtest_frame["position"].abs() > 0,
        "strategy_return",
    ].dropna()
    if active_returns.empty:
        win_rate = 0.0
        average_trade_return = 0.0
    else:
        win_rate = float((active_returns > 0).mean())
        average_trade_return = float(active_returns.mean())

    return {
        "win_rate": win_rate,
        "total_trades": float(metrics.get("total_trades", 0.0)),
        "average_trade_return": average_trade_return,
        "percent_time_in_market": float(metrics.get("percent_time_in_market", 0.0)),
        "annualized_return": float(metrics.get("strategy_annualized_return", 0.0)),
        "max_drawdown": float(metrics.get("strategy_max_drawdown", 0.0)),
        "risk_adjusted_return": float(metrics.get("risk_adjusted_return", metrics.get("strategy_sharpe_ratio", 0.0))),
    }


def _load_multi_stock_comparison(selected_model: str) -> pd.DataFrame:
    """Load the cached multi-stock comparison table for the selected model."""
    comparison_path = Path("outputs/multi_stock_model_comparison.csv")
    if not comparison_path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(comparison_path)
    frame = frame.loc[frame["model"] == selected_model].copy()
    if frame.empty:
        return frame

    columns = [
        "stock",
        "symbol",
        "strategy_total_return",
        "strategy_sharpe_ratio",
        "strategy_max_drawdown",
        "accuracy",
    ]
    if "strategy_annualized_return" in frame.columns:
        columns.insert(5, "strategy_annualized_return")

    display = frame[columns].rename(
        columns={
            "stock": "Stock",
            "symbol": "Symbol",
            "strategy_total_return": "Strategy Return",
            "strategy_sharpe_ratio": "Sharpe Ratio",
            "strategy_max_drawdown": "Max Drawdown",
            "strategy_annualized_return": "Annualized Return",
            "accuracy": "Accuracy",
        }
    )
    return display


_inject_dashboard_styles()
available_model_options = (
    MODEL_OPTIONS
    if is_xgboost_available()
    else [model for model in MODEL_OPTIONS if model != "XGBoost"]
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">QT</div>
            <div>
                <div class="sidebar-title">Quant Terminal</div>
                <div class="sidebar-caption">ML signals and backtesting</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">Strategy Controls</div>', unsafe_allow_html=True)
    ticker_option = st.selectbox(
        "Stock Selector",
        options=list(STOCK_TICKER_OPTIONS.keys()),
    )
    selected_ticker = STOCK_TICKER_OPTIONS[ticker_option]

    if selected_ticker is None:
        stock_ticker = st.text_input("Enter Custom Stock Ticker", value=RELIANCE_SYMBOL)
    else:
        stock_ticker = selected_ticker

    if stock_ticker:
        st.caption(f"Selected ticker: `{stock_ticker}`")

    selected_model = st.selectbox("Model Selector", options=available_model_options)
    if "XGBoost" not in available_model_options:
        st.caption("XGBoost is hidden because the package is not installed.")
    default_horizon_index = (
        FORECAST_HORIZON_OPTIONS.index(DEFAULT_HORIZON)
        if DEFAULT_HORIZON in FORECAST_HORIZON_OPTIONS
        else 0
    )
    forecast_horizon = st.selectbox(
        "Forecast Horizon",
        options=FORECAST_HORIZON_OPTIONS,
        index=default_horizon_index,
        format_func=lambda days: f"{days} trading day{'s' if days != 1 else ''}",
    )
    signal_threshold = st.slider(
        "Probability Threshold",
        min_value=0.50,
        max_value=0.90,
        value=BUY_THRESHOLD,
        step=0.01,
        help="BUY above this probability. SELL below the mirrored downside threshold.",
    )
    buy_threshold = float(signal_threshold)
    sell_threshold = float(1.0 - signal_threshold)

    st.divider()
    st.markdown('<div class="section-title">Run Configuration</div>', unsafe_allow_html=True)
    st.caption(f"Forecast horizon: `{forecast_horizon}` trading day(s)")
    st.caption(f"Selected model: `{selected_model}`")
    st.caption(f"Buy threshold: `{buy_threshold:.2f}`")
    st.caption(f"Sell threshold: `{sell_threshold:.2f}`")
    st.caption(f"Transaction cost: `{DEFAULT_TRANSACTION_COST:.3f}`")
    st.caption(f"Slippage: `{DEFAULT_SLIPPAGE:.4f}`")
    st.caption(f"Execution lag: `{DEFAULT_EXECUTION_LAG}` bar")
    st.caption(f"Position size: `{DEFAULT_POSITION_SIZE:.1f}`")
    run_model = st.button("Run Strategy", type="primary", width="stretch")

_render_dashboard_header(
    stock_ticker=stock_ticker or "None",
    selected_model=selected_model,
    forecast_horizon=int(forecast_horizon),
    buy_threshold=buy_threshold,
    sell_threshold=sell_threshold,
)

if run_model:
    ticker = stock_ticker.strip().upper()
    st.session_state.pop("dashboard_results", None)
    if not ticker:
        st.warning("Please enter a stock ticker.")
    else:
        status = st.status(f"Fetching Data for {ticker}", expanded=True)
        try:
            status.write("Fetching Data")
            prices = load_price_data(
                symbol=ticker,
                start=RELIANCE_START,
                end=RELIANCE_END,
            )

            status.update(label="Processing Features", state="running", expanded=True)
            status.write("Processing Features")
            feature_set = make_features(prices, horizon=int(forecast_horizon))
            latest_feature_frame = build_feature_frame(prices)
            if latest_feature_frame.empty:
                raise ValueError("Not enough history to compute the latest feature values.")
            latest_features = latest_feature_frame.tail(1)
            split = time_series_train_test_split(
                feature_set.X,
                feature_set.y,
                test_size=0.2,
            )

            status.update(label="Training Model", state="running", expanded=True)
            status.write("Training Model")
            random_forest_model = train_model(split.X_train, split.y_train)
            random_forest_best_params = _maybe_get_best_params(random_forest_model)
            rf_predicted_classes = predict(random_forest_model, split.X_test)
            rf_predicted_probabilities = predict_proba(random_forest_model, split.X_test)
            rf_predicted_signals = predict_signals(
                random_forest_model,
                split.X_test,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
            )
            rf_classification_metrics = evaluate_classifier(
                split.y_test,
                rf_predicted_classes,
            )

            logistic_regression_model = train_logistic_regression_model(
                split.X_train,
                split.y_train,
            )
            logistic_regression_best_params = _maybe_get_best_params(
                logistic_regression_model
            )
            xgboost_model = (
                train_xgboost_model(
                    split.X_train,
                    split.y_train,
                )
                if is_xgboost_available()
                else None
            )
            lr_predicted_classes = predict(logistic_regression_model, split.X_test)
            lr_predicted_probabilities = predict_proba(
                logistic_regression_model,
                split.X_test,
            )
            lr_predicted_signals = predict_signals(
                logistic_regression_model,
                split.X_test,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
            )
            lr_classification_metrics = evaluate_classifier(
                split.y_test,
                lr_predicted_classes,
            )
            trained_models = {
                "Random Forest": random_forest_model,
                "Logistic Regression": logistic_regression_model,
            }
            if xgboost_model is not None:
                xgb_predicted_classes = predict(xgboost_model, split.X_test)
                xgb_predicted_probabilities = predict_proba(xgboost_model, split.X_test)
                xgb_predicted_signals = predict_signals(
                    xgboost_model,
                    split.X_test,
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                )
                xgb_classification_metrics = evaluate_classifier(
                    split.y_test,
                    xgb_predicted_classes,
                )
                trained_models["XGBoost"] = xgboost_model
            else:
                xgb_predicted_classes = None
                xgb_predicted_probabilities = None
                xgb_predicted_signals = None
                xgb_classification_metrics = None
            selected_trained_model = trained_models[selected_model]
            latest_probability = float(
                predict_proba(selected_trained_model, latest_features).iloc[0]
            )
            latest_signal = str(
                predict_signals(
                    selected_trained_model,
                    latest_features,
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                ).iloc[0]
            )
            latest_confidence = _signal_confidence(
                latest_probability,
                latest_signal,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
            )

            status.update(label="Running Backtest", state="running", expanded=True)
            status.write("Running Backtest")
            market_returns = (
                prices["Close"]
                .pct_change(int(forecast_horizon))
                .shift(-int(forecast_horizon))
                .reindex(split.X_test.index)
            )
            random_forest_backtest = run_backtest(
                market_returns=market_returns,
                predictions=rf_predicted_signals,
                transaction_cost=DEFAULT_TRANSACTION_COST,
                slippage=DEFAULT_SLIPPAGE,
                position_size=DEFAULT_POSITION_SIZE,
                execution_lag=DEFAULT_EXECUTION_LAG,
            )
            logistic_regression_backtest = run_backtest(
                market_returns=market_returns,
                predictions=lr_predicted_signals,
                transaction_cost=DEFAULT_TRANSACTION_COST,
                slippage=DEFAULT_SLIPPAGE,
                position_size=DEFAULT_POSITION_SIZE,
                execution_lag=DEFAULT_EXECUTION_LAG,
            )
            xgboost_backtest = (
                run_backtest(
                    market_returns=market_returns,
                    predictions=xgb_predicted_signals,
                    transaction_cost=DEFAULT_TRANSACTION_COST,
                    slippage=DEFAULT_SLIPPAGE,
                    position_size=DEFAULT_POSITION_SIZE,
                    execution_lag=DEFAULT_EXECUTION_LAG,
                )
                if xgb_predicted_signals is not None
                else None
            )

            random_forest_predictions = pd.concat(
                [
                    split.y_test.rename("actual_target"),
                    rf_predicted_classes,
                    rf_predicted_probabilities,
                    rf_predicted_signals.rename("predicted_signal"),
                    market_returns.rename("realized_return"),
                ],
                axis=1,
            )
            logistic_regression_predictions = pd.concat(
                [
                    split.y_test.rename("actual_target"),
                    lr_predicted_classes,
                    lr_predicted_probabilities,
                    lr_predicted_signals.rename("predicted_signal"),
                    market_returns.rename("realized_return"),
                ],
                axis=1,
            )
            xgboost_predictions = (
                pd.concat(
                    [
                        split.y_test.rename("actual_target"),
                        xgb_predicted_classes,
                        xgb_predicted_probabilities,
                        xgb_predicted_signals.rename("predicted_signal"),
                        market_returns.rename("realized_return"),
                    ],
                    axis=1,
                )
                if xgb_predicted_signals is not None
                else None
            )
            random_forest_feature_importance = get_random_forest_feature_importance(
                random_forest_model,
                split.X_train.columns,
            )
            logistic_regression_coefficients = get_logistic_regression_coefficients(
                logistic_regression_model,
                split.X_train.columns,
            )

            model_results = [
                {
                    "model_name": "Random Forest",
                    "classification_metrics": rf_classification_metrics,
                    "backtest_result": random_forest_backtest,
                    "predictions": random_forest_predictions,
                    "best_params": random_forest_best_params,
                },
                {
                    "model_name": "Logistic Regression",
                    "classification_metrics": lr_classification_metrics,
                    "backtest_result": logistic_regression_backtest,
                    "predictions": logistic_regression_predictions,
                    "best_params": None,
                },
            ]
            if xgboost_backtest is not None and xgboost_predictions is not None:
                model_results.insert(
                    1,
                    {
                        "model_name": "XGBoost",
                        "classification_metrics": xgb_classification_metrics,
                        "backtest_result": xgboost_backtest,
                        "predictions": xgboost_predictions,
                        "best_params": None,
                    },
                )
            comparison_frame = _build_comparison_frame(model_results)
        except Exception as exc:
            if _is_invalid_ticker_error(exc):
                status.update(label="Invalid stock ticker", state="error", expanded=True)
                st.error(
                    "Invalid stock ticker or no data found. "
                    "Please enter a valid Yahoo Finance symbol such as "
                    "`RELIANCE.NS`, `AAPL`, or `MSFT`."
                )
            else:
                status.update(label="Pipeline failed", state="error", expanded=True)
                st.error(f"Pipeline failed: {exc}")
        else:
            status.update(label="Pipeline completed", state="complete", expanded=False)
            st.session_state["dashboard_results"] = {
                "ticker": ticker,
                "prices": prices,
                "comparison_frame": comparison_frame,
                "random_forest_best_params": random_forest_best_params,
                "logistic_regression_best_params": logistic_regression_best_params,
                "rf_classification_metrics": rf_classification_metrics,
                "xgb_classification_metrics": xgb_classification_metrics,
                "lr_classification_metrics": lr_classification_metrics,
                "selected_model": selected_model,
                "forecast_horizon": int(forecast_horizon),
                "buy_threshold": buy_threshold,
                "sell_threshold": sell_threshold,
                "latest_signal": latest_signal,
                "latest_confidence": latest_confidence,
                "latest_probability": latest_probability,
                "latest_features": latest_features,
                "random_forest_predictions": random_forest_predictions,
                "xgboost_predictions": xgboost_predictions,
                "logistic_regression_predictions": logistic_regression_predictions,
                "random_forest_backtest": random_forest_backtest,
                "xgboost_backtest": xgboost_backtest,
                "logistic_regression_backtest": logistic_regression_backtest,
                "random_forest_feature_importance": random_forest_feature_importance,
                "logistic_regression_coefficients": logistic_regression_coefficients,
            }

results = st.session_state.get("dashboard_results")
dashboard_tab, insights_tab, backtesting_tab, about_tab = st.tabs(
    ["Dashboard", "Model Insights", "Backtesting", "About Project"]
)

if results is None:
    with dashboard_tab:
        _render_section_intro(
            "Dashboard Ready",
            "Select a stock, choose a model, tune thresholds, and run the strategy from the sidebar.",
        )
        with st.container(border=True):
            st.write(
                "The quant research workspace is ready. Results will populate the "
                "hero KPIs, trading signal panel, Plotly charts, prediction table, "
                "and backtesting analytics after a run."
            )
    with insights_tab:
        st.info("Model insights will appear after a strategy run.")
    with backtesting_tab:
        st.info("Backtesting analytics will appear after a strategy run.")
    with about_tab:
        _render_section_intro(
            "About Project",
            "A compact research platform for experimenting with ML-driven trading signals.",
        )
        _render_about_project_content()
else:
    ticker = results["ticker"]
    prices = results["prices"]
    comparison_frame = results["comparison_frame"]
    random_forest_best_params = results["random_forest_best_params"]
    logistic_regression_best_params = results.get("logistic_regression_best_params")
    rf_classification_metrics = results["rf_classification_metrics"]
    xgb_classification_metrics = results["xgb_classification_metrics"]
    lr_classification_metrics = results["lr_classification_metrics"]
    selected_model = results["selected_model"]
    result_forecast_horizon = int(results.get("forecast_horizon", DEFAULT_HORIZON))
    buy_threshold = results["buy_threshold"]
    sell_threshold = results["sell_threshold"]
    latest_signal = results["latest_signal"]
    latest_confidence = results["latest_confidence"]
    latest_probability = results["latest_probability"]
    latest_risk_level = _risk_level(latest_confidence)
    latest_features = results["latest_features"]
    random_forest_predictions = results["random_forest_predictions"]
    xgboost_predictions = results.get("xgboost_predictions")
    logistic_regression_predictions = results["logistic_regression_predictions"]
    random_forest_backtest = results["random_forest_backtest"]
    xgboost_backtest = results.get("xgboost_backtest")
    logistic_regression_backtest = results["logistic_regression_backtest"]
    random_forest_feature_importance = results["random_forest_feature_importance"]
    logistic_regression_coefficients = results["logistic_regression_coefficients"]
    selected_model_metrics = _get_selected_model_metrics(
        comparison_frame,
        selected_model,
    )
    predictions_by_model = {
        "Random Forest": random_forest_predictions,
        "Logistic Regression": logistic_regression_predictions,
    }
    backtests_by_model = {
        "Random Forest": random_forest_backtest,
        "Logistic Regression": logistic_regression_backtest,
    }
    if xgboost_predictions is not None and xgboost_backtest is not None:
        predictions_by_model["XGBoost"] = xgboost_predictions
        backtests_by_model["XGBoost"] = xgboost_backtest
    selected_predictions = predictions_by_model[selected_model]
    selected_backtest = backtests_by_model[selected_model]
    portfolio_metrics = _portfolio_analytics(
        selected_backtest.frame,
        selected_backtest.metrics,
    )
    recommendation = build_recommendation(
        latest_features,
        signal=latest_signal,
        probability_up=latest_probability,
        confidence=latest_confidence,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    multi_stock_comparison = _load_multi_stock_comparison(selected_model)
    plotly_config = {
        "displayModeBar": True,
        "scrollZoom": True,
        "responsive": True,
    }

    with dashboard_tab:
        _render_section_intro(
            "Dashboard",
            "Live model output, selected-strategy performance, and current signal state.",
        )
        _render_hero_kpis(selected_model_metrics, latest_signal)

        overview_left, overview_right = st.columns([1.35, 1.0])
        with overview_left.container(border=True):
            _render_section_intro("Model Results")
            st.success(f"Pipeline completed for {ticker}.")
            st.caption(f"Selected model: {selected_model}")
            _render_metric_cards(selected_model_metrics)
            st.caption(
                "Backtest assumptions: 0.10% transaction cost, 0.05% slippage, "
                "one-bar execution lag, fixed 100% BUY allocation, and HOLD/SELL "
                "stay in cash."
            )
            if random_forest_best_params:
                st.write("Best RandomForest parameters from GridSearchCV:")
                st.write(random_forest_best_params)
            if logistic_regression_best_params:
                st.write("Best Logistic Regression parameters from GridSearchCV:")
                st.write(logistic_regression_best_params)

        with overview_right.container(border=True):
            _render_trading_signal_panel(
                latest_signal,
                latest_confidence,
                latest_risk_level,
                latest_probability,
                selected_model,
                buy_threshold,
                sell_threshold,
                latest_features.index[-1],
                result_forecast_horizon,
            )

        with st.container(border=True):
            _render_section_intro(
                "Recommendation",
                "Transparent rule-based explanation layered on top of model probability.",
            )
            _render_recommendation_card(recommendation)

        with st.container(border=True):
            _render_section_intro(
                "Price Visualization",
                "Interactive OHLC candlesticks with range controls, zoom, and hover inspection.",
            )
            st.plotly_chart(
                _plot_candlestick(prices, ticker),
                width="stretch",
                config=plotly_config,
            )

        with st.container(border=True):
            _render_section_intro(
                "Multi-Stock Comparison",
                "Cached cross-sectional comparison for the selected model.",
            )
            if multi_stock_comparison.empty:
                st.info("Run the multi-stock comparison script to populate this table.")
            else:
                multi_stock_formatters = {
                    "Strategy Return": "{:.2%}",
                    "Sharpe Ratio": "{:.3f}",
                    "Max Drawdown": "{:.2%}",
                    "Accuracy": "{:.3f}",
                }
                if "Annualized Return" in multi_stock_comparison.columns:
                    multi_stock_formatters["Annualized Return"] = "{:.2%}"
                st.dataframe(
                    multi_stock_comparison.style.format(multi_stock_formatters),
                    width="stretch",
                )

    with insights_tab:
        _render_section_intro(
            "Model Insights",
            "Inspect feature drivers, coefficient direction, and the latest engineered inputs.",
        )

        with st.container(border=True):
            _render_section_intro(
                "RandomForest Feature Importance",
                "Top model inputs ranked by importance. Higher values had more influence in the RandomForest model.",
            )
            rf_importance_fig = _plot_feature_importance(
                random_forest_feature_importance,
                title="RandomForest Feature Importance",
                value_col="importance",
            )
            st.plotly_chart(
                rf_importance_fig,
                width="stretch",
                config=plotly_config,
            )
            sorted_importance = random_forest_feature_importance.sort_values(
                "importance",
                ascending=False,
            ).head(20)
            st.dataframe(
                sorted_importance.style.format(
                    {"importance": "{:.4f}"}
                ),
                width="stretch",
            )

        insights_left, insights_right = st.columns(2)
        with insights_left.container(border=True):
            _render_section_intro("Logistic Regression Coefficients")
            lr_coefficient_fig = _plot_feature_importance(
                logistic_regression_coefficients,
                title="Top Logistic Regression Coefficients",
                value_col="abs_coefficient",
            )
            st.plotly_chart(
                lr_coefficient_fig,
                width="stretch",
                config=plotly_config,
            )
        with insights_right.container(border=True):
            _render_section_intro("Latest Feature Values")
            st.dataframe(
                latest_features.T.rename(columns={latest_features.index[-1]: "value"})
                .style.format("{:.4f}"),
                width="stretch",
            )

    with backtesting_tab:
        _render_section_intro(
            "Backtesting",
            "Strategy simulation with transaction costs, benchmark comparison, and portfolio diagnostics.",
        )
        with st.container(border=True):
            st.caption(f"Selected model: {selected_model}")
            _render_metric_cards(selected_model_metrics)

        with st.container(border=True):
            _render_section_intro(
                "Strategy vs Market",
                "Interactive equity curve comparison across models and buy-and-hold.",
            )
            equity_fig = _plot_equity_curve(
                random_forest_backtest.frame,
                logistic_regression_backtest.frame,
                ticker,
                xgboost_frame=(
                    xgboost_backtest.frame if xgboost_backtest is not None else None
                ),
            )
            st.plotly_chart(
                equity_fig,
                width="stretch",
                config=plotly_config,
        )

        _render_section_intro("Performance Analytics")
        analytics_cols = st.columns(7)
        analytics_cols[0].metric("Total Trades", f"{portfolio_metrics['total_trades']:.0f}")
        analytics_cols[1].metric("Win Rate", _format_pct(portfolio_metrics["win_rate"]))
        analytics_cols[2].metric(
            "Percent Time In Market",
            _format_pct(portfolio_metrics["percent_time_in_market"]),
        )
        analytics_cols[3].metric(
            "Annualized Return",
            _format_pct(portfolio_metrics["annualized_return"]),
        )
        analytics_cols[4].metric(
            "Max Drawdown",
            _format_pct(portfolio_metrics["max_drawdown"]),
        )
        analytics_cols[5].metric(
            "Average Trade Return",
            _format_pct(portfolio_metrics["average_trade_return"]),
        )
        analytics_cols[6].metric(
            "Risk-Adjusted Return",
            _format_number(portfolio_metrics["risk_adjusted_return"], 3),
        )

        with st.container(border=True):
            _render_section_intro("Model Comparison")
            st.dataframe(
                comparison_frame.style.format(
                    {
                        "accuracy": "{:.3f}",
                        "precision": "{:.3f}",
                        "recall": "{:.3f}",
                        "f1": "{:.3f}",
                        "strategy_return": "{:.2%}",
                        "market_return": "{:.2%}",
                        "sharpe_ratio": "{:.3f}",
                        "max_drawdown": "{:.2%}",
                        "annualized_return": "{:.2%}",
                    }
                ),
                width="stretch",
            )

        with st.container(border=True):
            _render_section_intro(
                "Recent Predictions",
                "Latest test-period predictions with model confidence, realized movement, and trade signal.",
            )
            formatted_predictions = _prepare_predictions_table(selected_predictions)
            st.dataframe(formatted_predictions, width="stretch", hide_index=True)

            download_col_1, download_col_2 = st.columns(2)
            download_col_1.download_button(
                "Download Predictions CSV",
                data=selected_predictions.to_csv(index_label="Date"),
                file_name=f"predictions_{selected_model.lower().replace(' ', '_')}_{ticker}.csv",
                mime="text/csv",
                width="stretch",
            )
            download_col_2.download_button(
                "Download Backtest CSV",
                data=selected_backtest.frame.to_csv(index_label="Date"),
                file_name=f"backtest_{selected_model.lower().replace(' ', '_')}_{ticker}.csv",
                mime="text/csv",
                width="stretch",
            )

    with about_tab:
        _render_section_intro(
            "About Project",
            "A compact research platform for experimenting with ML-driven trading signals.",
        )
        _render_about_project_content()
