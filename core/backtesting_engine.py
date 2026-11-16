"""Institutional Backtesting Engine for Quantitative Strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go


@dataclass
class BacktestResult:
    """Quantitative backtest execution container."""

    frame: pd.DataFrame
    metrics: dict[str, float]
    trades: list[dict[str, object]] = field(default_factory=list)


def calculate_performance_metrics(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    positions: pd.Series,
    turnover: pd.Series,
    costs_paid: float,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.065,
    annualization_factor: int = 252,
) -> dict[str, float]:
    """
    Calculate comprehensive, institutional-grade risk and performance statistics.
    """
    strat_clean = strategy_returns.fillna(0.0)
    mkt_clean = market_returns.fillna(0.0)
    n_days = len(strat_clean)

    if n_days < 2:
        return {
            "strategy_total_return": 0.0,
            "buy_and_hold_total_return": 0.0,
            "buy_hold_total_return": 0.0,
            "strategy_annualized_return": 0.0,
            "buy_and_hold_annualized_return": 0.0,
            "strategy_sharpe_ratio": 0.0,
            "buy_and_hold_sharpe_ratio": 0.0,
            "strategy_sortino_ratio": 0.0,
            "strategy_max_drawdown": 0.0,
            "buy_and_hold_max_drawdown": 0.0,
            "strategy_calmar_ratio": 0.0,
            "strategy_volatility": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "percent_time_in_market": 0.0,
            "total_trades": 0.0,
            "total_transaction_cost_paid": 0.0,
        }

    years = max(n_days / annualization_factor, 0.05)

    # Cumulative wealth vectors
    strat_cum = (1.0 + strat_clean).cumprod()
    mkt_cum = (1.0 + mkt_clean).cumprod()

    strat_total_ret = float(strat_cum.iloc[-1] - 1.0)
    mkt_total_ret = float(mkt_cum.iloc[-1] - 1.0)

    # Annualized return (CAGR)
    strat_cagr = float((1.0 + strat_total_ret) ** (1.0 / years) - 1.0) if strat_cum.iloc[-1] > 0 else -1.0
    mkt_cagr = float((1.0 + mkt_total_ret) ** (1.0 / years) - 1.0) if mkt_cum.iloc[-1] > 0 else -1.0

    # Volatility
    strat_vol = float(strat_clean.std() * np.sqrt(annualization_factor))
    mkt_vol = float(mkt_clean.std() * np.sqrt(annualization_factor))

    # Sharpe ratio
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0
    excess_strat = strat_clean - daily_rf
    strat_sharpe = float((excess_strat.mean() / (strat_clean.std() + 1e-9)) * np.sqrt(annualization_factor))

    excess_mkt = mkt_clean - daily_rf
    mkt_sharpe = float((excess_mkt.mean() / (mkt_clean.std() + 1e-9)) * np.sqrt(annualization_factor))

    # Downside deviation & Sortino ratio
    downside_returns = strat_clean[strat_clean < 0.0]
    downside_std = float(downside_returns.std() * np.sqrt(annualization_factor)) if len(downside_returns) > 1 else strat_vol
    strat_sortino = float((strat_cagr - risk_free_rate) / (downside_std + 1e-9))

    # Drawdowns
    strat_peaks = np.maximum.accumulate(strat_cum)
    strat_dd = (strat_cum - strat_peaks) / strat_peaks
    strat_max_dd = float(abs(strat_dd.min()))

    mkt_peaks = np.maximum.accumulate(mkt_cum)
    mkt_dd = (mkt_cum - mkt_peaks) / mkt_peaks
    mkt_max_dd = float(abs(mkt_dd.min()))

    # Calmar ratio
    calmar = float(strat_cagr / (strat_max_dd + 1e-9))

    # Win rate on active days
    active_days = strat_clean[positions.abs() > 0.001]
    win_rate = float((active_days > 0.0).sum() / len(active_days)) if len(active_days) > 0 else 0.0

    # Profit factor
    gross_gains = float(active_days[active_days > 0.0].sum())
    gross_losses = float(abs(active_days[active_days < 0.0].sum()))
    profit_factor = float(gross_gains / (gross_losses + 1e-9)) if gross_losses > 0 else (99.0 if gross_gains > 0 else 1.0)

    # Activity & Trades
    time_in_market = float((positions.abs() > 0.001).sum() / n_days)
    total_trades = int((turnover.abs() > 0.01).sum())

    return {
        "strategy_total_return": strat_total_ret,
        "buy_and_hold_total_return": mkt_total_ret,
        "buy_hold_total_return": mkt_total_ret,
        "strategy_annualized_return": strat_cagr,
        "buy_and_hold_annualized_return": mkt_cagr,
        "strategy_sharpe_ratio": strat_sharpe,
        "buy_and_hold_sharpe_ratio": mkt_sharpe,
        "strategy_sortino_ratio": strat_sortino,
        "strategy_max_drawdown": strat_max_dd,
        "buy_and_hold_max_drawdown": mkt_max_dd,
        "strategy_calmar_ratio": calmar,
        "strategy_volatility": strat_vol,
        "buy_and_hold_volatility": mkt_vol,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "percent_time_in_market": time_in_market,
        "total_trades": float(total_trades),
        "total_transaction_cost_paid": float(costs_paid),
        "strategy_vs_buy_and_hold": strat_total_ret - mkt_total_ret,
    }


def run_backtest(
    market_returns: pd.Series,
    predictions: pd.Series,
    transaction_cost: float = 0.0010,
    slippage: float = 0.0005,
    position_size: float = 1.0,
    execution_lag: int = 1,
    short_on_sell: bool = False,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """
    Run scientifically rigorous backtest without look-ahead bias.

    Signals generated at time t are executed at t + execution_lag.
    Transaction costs and slippage are charged on position changes (turnover)
    to prevent penalizing static holds.
    """
    # Align signals and market returns
    common_idx = market_returns.dropna().index.intersection(predictions.dropna().index)
    if len(common_idx) < 2:
        empty_frame = pd.DataFrame(index=common_idx)
        return BacktestResult(frame=empty_frame, metrics={})

    mkt_ret = market_returns.reindex(common_idx).fillna(0.0)
    sig = predictions.reindex(common_idx).fillna("HOLD")

    # Map signals to intended target positions
    target_pos = pd.Series(0.0, index=common_idx)
    target_pos.loc[sig == "BUY"] = float(position_size)
    if short_on_sell:
        target_pos.loc[sig == "SELL"] = -float(position_size)
    else:
        target_pos.loc[sig == "SELL"] = 0.0

    # Apply execution lag: trade executes on next bar
    lagged_pos = target_pos.shift(execution_lag).fillna(0.0)

    # Position turnover (trades occur only when position changes)
    turnover = (lagged_pos - lagged_pos.shift(1).fillna(0.0)).abs()

    # Total trading cost per bar: turnover * (cost + slippage)
    total_cost_rate = transaction_cost + slippage
    friction = turnover * total_cost_rate

    # Strategy net returns
    raw_strategy_ret = lagged_pos * mkt_ret
    net_strategy_ret = raw_strategy_ret - friction

    # Cumulative wealth
    strat_cum = (1.0 + net_strategy_ret).cumprod()
    mkt_cum = (1.0 + mkt_ret).cumprod()

    # Drawdowns
    strat_peaks = np.maximum.accumulate(strat_cum)
    strat_dd = (strat_cum - strat_peaks) / strat_peaks

    mkt_peaks = np.maximum.accumulate(mkt_cum)
    mkt_dd = (mkt_cum - mkt_peaks) / mkt_peaks

    # Total friction in cash terms
    total_costs_paid = float((friction * initial_capital * strat_cum.shift(1).fillna(1.0)).sum())

    # Build detailed ledger frame
    frame = pd.DataFrame(
        {
            "signal": sig,
            "target_position": target_pos,
            "position": lagged_pos,
            "turnover": turnover,
            "market_return": mkt_ret,
            "transaction_cost": friction,
            "strategy_return": net_strategy_ret,
            "cumulative_market_return": mkt_cum - 1.0,
            "cumulative_strategy_return": strat_cum - 1.0,
            "drawdown_market": mkt_dd,
            "drawdown_strategy": strat_dd,
            "equity_strategy": initial_capital * strat_cum,
            "equity_market": initial_capital * mkt_cum,
        },
        index=common_idx,
    )

    metrics = calculate_performance_metrics(
        strategy_returns=net_strategy_ret,
        market_returns=mkt_ret,
        positions=lagged_pos,
        turnover=turnover,
        costs_paid=total_costs_paid,
        initial_capital=initial_capital,
    )

    return BacktestResult(frame=frame, metrics=metrics)


def plot_equity_curve(
    backtest_result: BacktestResult,
    title: str = "Strategy vs Buy & Hold Equity Curve",
    output_path: str | None = None,
) -> go.Figure:
    """Create TradingView / institutional styled Plotly equity curve."""
    df = backtest_result.frame

    fig = go.Figure()

    if not df.empty and "equity_strategy" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["equity_strategy"],
                mode="lines",
                name="Quant Strategy",
                line=dict(color="#22d3ee", width=2.2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["equity_market"],
                mode="lines",
                name="Buy & Hold Benchmark",
                line=dict(color="#94a3b8", width=1.5, dash="dot"),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#f8fafc", size=15)),
        paper_bgcolor="rgba(11, 15, 25, 0.95)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#94a3b8", family="Arial, sans-serif"),
        xaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.12)",
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.12)",
            showgrid=True,
            zeroline=False,
            title="Portfolio Value ($)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#f8fafc"),
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        height=480,
    )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(out_p))
        except Exception:
            # Fallback if kaleido is not installed
            html_path = out_p.with_suffix(".html")
            fig.write_html(str(html_path))

    return fig
