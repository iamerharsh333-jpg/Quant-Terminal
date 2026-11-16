"""Institutional Dark-Themed Plotly Charts for Quant Terminal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CHART_THEME = dict(
    paper_bgcolor="rgba(11, 17, 32, 0.95)",
    plot_bgcolor="rgba(15, 23, 42, 0.65)",
    font=dict(color="#94a3b8", family="Arial, sans-serif"),
    gridcolor="rgba(148, 163, 184, 0.10)",
)


def style_figure(fig: go.Figure, title: str = "", height: int = 500) -> go.Figure:
    """Apply consistent Bloomberg/TradingView dark styling to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color="#f8fafc", size=14, family="Arial, sans-serif"),
            x=0.01,
            y=0.98,
        ),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        xaxis=dict(
            gridcolor=CHART_THEME["gridcolor"],
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=CHART_THEME["gridcolor"],
            showgrid=True,
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#e2e8f0", size=11),
        ),
        margin=dict(l=45, r=45, t=55, b=40),
        height=height,
    )
    return fig


def plot_candlestick_with_indicators(
    df: pd.DataFrame,
    ticker: str,
    show_sma_20: bool = True,
    show_sma_50: bool = True,
    show_sma_200: bool = False,
    show_bb: bool = True,
    show_rsi: bool = False,
    show_macd: bool = False,
) -> go.Figure:
    """
    Render professional TradingView-style Candlestick chart with volume,
    moving averages, Bollinger Bands, and optional RSI / MACD subplots.
    """
    # Determine rows based on subplots
    extra_rows = (1 if show_rsi else 0) + (1 if show_macd else 0)
    total_rows = 2 + extra_rows
    row_heights = [0.65, 0.15] + ([0.10] * extra_rows)
    # Normalize row heights to sum to 1.0
    row_heights = [h / sum(row_heights) for h in row_heights]

    subplot_titles = [f"{ticker} OHLCV", "Volume"]
    if show_rsi:
        subplot_titles.append("RSI (14)")
    if show_macd:
        subplot_titles.append("MACD")

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444",
            increasing_fillcolor="#10b981",
            decreasing_fillcolor="#ef4444",
        ),
        row=1,
        col=1,
    )

    # Overlays: SMA 20, 50, 200
    if show_sma_20 and "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_20"], line=dict(color="#38bdf8", width=1.4), name="SMA 20"),
            row=1,
            col=1,
        )
    if show_sma_50 and "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_50"], line=dict(color="#f59e0b", width=1.4), name="SMA 50"),
            row=1,
            col=1,
        )
    if show_sma_200 and "SMA_200" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_200"], line=dict(color="#ec4899", width=1.4), name="SMA 200"),
            row=1,
            col=1,
        )

    # Bollinger Bands
    if show_bb and "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                line=dict(color="rgba(148, 163, 184, 0.4)", width=1, dash="dot"),
                name="BB Upper",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                line=dict(color="rgba(148, 163, 184, 0.4)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(148, 163, 184, 0.05)",
                name="BB Lower",
            ),
            row=1,
            col=1,
        )

    # 2. Volume bar chart
    vol_colors = [
        "#10b981" if c >= o else "#ef4444"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color=vol_colors,
            name="Volume",
            opacity=0.75,
        ),
        row=2,
        col=1,
    )

    curr_row = 3
    # 3. Optional RSI
    if show_rsi and "RSI_14" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI_14"], line=dict(color="#a855f7", width=1.5), name="RSI 14"),
            row=curr_row,
            col=1,
        )
        # 70 and 30 reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239, 68, 68, 0.5)", row=curr_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(16, 185, 129, 0.5)", row=curr_row, col=1)
        curr_row += 1

    # 4. Optional MACD
    if show_macd and "MACD" in df.columns and "MACD_Signal" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#38bdf8", width=1.4), name="MACD"),
            row=curr_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD_Signal"], line=dict(color="#fb923c", width=1.4), name="Signal"),
            row=curr_row,
            col=1,
        )
        hist_colors = [
            "#10b981" if h >= 0 else "#ef4444"
            for h in df.get("MACD_Hist", [0] * len(df))
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=hist_colors, name="Hist"),
            row=curr_row,
            col=1,
        )

    fig.update_xaxes(rangeslider_visible=False)
    total_height = 560 + (extra_rows * 120)
    return style_figure(fig, title=f"{ticker} Institutional Chart", height=total_height)


def plot_equity_and_drawdown(
    frame: pd.DataFrame,
    title: str = "Strategy Performance & Drawdown Analysis",
) -> go.Figure:
    """Render combined 2-panel chart: Equity Curve above, Underwater Drawdown below."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.70, 0.30],
        subplot_titles=["Portfolio Value ($)", "Drawdown (%)"],
    )

    if not frame.empty and "equity_strategy" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["equity_strategy"],
                line=dict(color="#22d3ee", width=2.2),
                name="Quant Strategy",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["equity_market"],
                line=dict(color="#94a3b8", width=1.5, dash="dot"),
                name="Buy & Hold",
            ),
            row=1,
            col=1,
        )

        # Drawdown panel
        strat_dd_pct = frame["drawdown_strategy"] * 100.0
        mkt_dd_pct = frame["drawdown_market"] * 100.0

        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=strat_dd_pct,
                line=dict(color="#f43f5e", width=1.2),
                fill="tozeroy",
                fillcolor="rgba(244, 63, 94, 0.20)",
                name="Strategy Drawdown",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=mkt_dd_pct,
                line=dict(color="#64748b", width=1.0, dash="dot"),
                name="Market Drawdown",
            ),
            row=2,
            col=1,
        )

    return style_figure(fig, title=title, height=520)


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Render correlation matrix heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale=[[0, "#3b82f6"], [0.5, "#0f172a"], [1, "#10b981"]],
            zmin=-1.0,
            zmax=1.0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont=dict(color="#f8fafc", size=12),
        )
    )
    return style_figure(fig, title="Asset Return Correlation Matrix", height=420)


def plot_feature_importance_bar(df: pd.DataFrame, title: str = "Feature Importance") -> go.Figure:
    """Horizontal bar chart for top predictive features."""
    top = df.head(15).iloc[::-1]  # Reverse for top-down descending
    metric_col = "importance" if "importance" in top.columns else ("abs_coefficient" if "abs_coefficient" in top.columns else top.columns[1])

    fig = go.Figure(
        go.Bar(
            x=top[metric_col],
            y=top["feature"],
            orientation="h",
            marker=dict(
                color=top[metric_col],
                colorscale="Viridis",
            ),
        )
    )
    return style_figure(fig, title=title, height=450)

