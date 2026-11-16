"""Modern Institutional Plotly Charts for Quant Terminal."""

from __future__ import annotations

import calendar
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Exact Design Direction Palette
THEME = dict(
    bg="#0B0E14",
    card_bg="#151B26",
    border="#252D3A",
    text_main="#E6EAF0",
    text_muted="#8792A5",
    green="#16C784",
    red="#EA3943",
    accent="#3861FB",
    cyan="#22D3EE",
    amber="#F59E0B",
    purple="#8B5CF6",
)


def style_figure(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Apply modern institutional TradingView/Bloomberg dark styling."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color=THEME["text_main"], size=13, family="'JetBrains Mono', 'Inter', monospace"),
            x=0.01,
            y=0.98,
        ),
        paper_bgcolor=THEME["card_bg"],
        plot_bgcolor=THEME["card_bg"],
        font=dict(color=THEME["text_muted"], family="'Inter', -apple-system, sans-serif", size=11),
        xaxis=dict(
            gridcolor=THEME["border"],
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            color=THEME["text_muted"],
        ),
        yaxis=dict(
            gridcolor=THEME["border"],
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            color=THEME["text_muted"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=THEME["text_main"], size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=40, r=40, t=45, b=35),
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
    """Interactive Candlestick chart with volume, moving averages, and indicators."""
    extra_rows = (1 if show_rsi else 0) + (1 if show_macd else 0)
    total_rows = 2 + extra_rows
    row_heights = [0.68, 0.16] + ([0.14] * extra_rows)
    row_heights = [h / sum(row_heights) for h in row_heights]

    subplot_titles = [f"{ticker} Price", "Volume"]
    if show_rsi:
        subplot_titles.append("RSI (14)")
    if show_macd:
        subplot_titles.append("MACD (12, 26, 9)")

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
            increasing_line_color=THEME["green"],
            decreasing_line_color=THEME["red"],
            increasing_fillcolor=THEME["green"],
            decreasing_fillcolor=THEME["red"],
        ),
        row=1,
        col=1,
    )

    if show_sma_20 and "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_20"], line=dict(color=THEME["cyan"], width=1.3), name="SMA 20"),
            row=1,
            col=1,
        )
    if show_sma_50 and "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_50"], line=dict(color=THEME["amber"], width=1.3), name="SMA 50"),
            row=1,
            col=1,
        )
    if show_sma_200 and "SMA_200" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_200"], line=dict(color=THEME["purple"], width=1.3), name="SMA 200"),
            row=1,
            col=1,
        )

    if show_bb and "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                line=dict(color="rgba(135, 146, 165, 0.4)", width=1, dash="dot"),
                name="BB Upper",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                line=dict(color="rgba(135, 146, 165, 0.4)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(135, 146, 165, 0.04)",
                name="BB Lower",
            ),
            row=1,
            col=1,
        )

    # 2. Volume bar chart
    vol_colors = [
        THEME["green"] if c >= o else THEME["red"]
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
    if show_rsi and "RSI_14" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI_14"], line=dict(color=THEME["purple"], width=1.4), name="RSI"),
            row=curr_row,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color=THEME["red"], opacity=0.4, row=curr_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=THEME["green"], opacity=0.4, row=curr_row, col=1)
        curr_row += 1

    if show_macd and "MACD" in df.columns and "MACD_Signal" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD"], line=dict(color=THEME["cyan"], width=1.3), name="MACD"),
            row=curr_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD_Signal"], line=dict(color=THEME["amber"], width=1.3), name="Signal"),
            row=curr_row,
            col=1,
        )
        hist_colors = [
            THEME["green"] if h >= 0 else THEME["red"]
            for h in df.get("MACD_Hist", [0] * len(df))
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=hist_colors, name="Hist"),
            row=curr_row,
            col=1,
        )

    fig.update_xaxes(rangeslider_visible=False)
    total_height = 460 + (extra_rows * 110)
    return style_figure(fig, title=f"{ticker} Technical Chart", height=total_height)


def plot_equity_and_drawdown(
    frame: pd.DataFrame,
    title: str = "Strategy vs Buy & Hold",
) -> go.Figure:
    """Strategy Equity Curve with Drawdown subpanel."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=["Portfolio Value ($)", "Drawdown (%)"],
    )

    if not frame.empty and "equity_strategy" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["equity_strategy"],
                line=dict(color=THEME["cyan"], width=2.0),
                name="Quant Strategy",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["equity_market"],
                line=dict(color=THEME["text_muted"], width=1.3, dash="dot"),
                name="Buy & Hold",
            ),
            row=1,
            col=1,
        )

        strat_dd_pct = frame["drawdown_strategy"] * 100.0
        mkt_dd_pct = frame["drawdown_market"] * 100.0

        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=strat_dd_pct,
                line=dict(color=THEME["red"], width=1.2),
                fill="tozeroy",
                fillcolor="rgba(234, 57, 67, 0.15)",
                name="Strategy DD",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=mkt_dd_pct,
                line=dict(color="rgba(135, 146, 165, 0.4)", width=1.0, dash="dot"),
                name="Market DD",
            ),
            row=2,
            col=1,
        )

    return style_figure(fig, title=title, height=440)


def plot_monthly_returns_heatmap(returns_series: pd.Series) -> go.Figure:
    """Monthly Return Heatmap matrix for performance attribution."""
    if returns_series.empty or len(returns_series) < 10:
        return go.Figure()

    # Resample to monthly returns
    monthly = (1.0 + returns_series).resample("ME").prod() - 1.0
    monthly_df = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values * 100.0,
    })

    pivot = monthly_df.pivot(index="Year", columns="Month", values="Return")
    month_names = [calendar.month_abbr[m] for m in range(1, 13)]
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[range(1, 13)]
    pivot.columns = month_names

    z_vals = pivot.values
    text_vals = np.where(np.isnan(z_vals), "", np.char.add(np.round(z_vals, 1).astype(str), "%"))

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=month_names,
            y=pivot.index.astype(str),
            colorscale=[
                [0.0, THEME["red"]],
                [0.5, THEME["card_bg"]],
                [1.0, THEME["green"]],
            ],
            zmid=0.0,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(color=THEME["text_main"], size=10, family="'JetBrains Mono', monospace"),
            showscale=False,
        )
    )
    return style_figure(fig, title="Monthly Returns (%)", height=280)


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Asset correlation matrix heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale=[[0, THEME["accent"]], [0.5, THEME["card_bg"]], [1, THEME["green"]]],
            zmin=-1.0,
            zmax=1.0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont=dict(color=THEME["text_main"], size=11, family="'JetBrains Mono', monospace"),
            showscale=False,
        )
    )
    return style_figure(fig, title="Asset Return Correlation", height=350)


def plot_feature_importance_bar(df: pd.DataFrame, title: str = "Key Drivers") -> go.Figure:
    """Horizontal bar chart for model drivers."""
    top = df.head(10).iloc[::-1]
    metric_col = "importance" if "importance" in top.columns else ("abs_coefficient" if "abs_coefficient" in top.columns else top.columns[1])

    fig = go.Figure(
        go.Bar(
            x=top[metric_col],
            y=top["feature"],
            orientation="h",
            marker=dict(color=THEME["accent"]),
        )
    )
    return style_figure(fig, title=title, height=360)
