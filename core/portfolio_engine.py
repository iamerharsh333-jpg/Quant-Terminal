"""Portfolio Analytics and Multi-Asset Risk Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class PortfolioAnalyticsResult:
    """Portfolio evaluation metrics and risk attribution container."""

    portfolio_equity: pd.Series
    daily_returns: pd.Series
    asset_returns: pd.DataFrame
    weights: dict[str, float]
    metrics: dict[str, float]
    correlation_matrix: pd.DataFrame
    risk_contributions: pd.DataFrame
    assumptions: list[str] = field(default_factory=list)


def build_portfolio_analytics(
    asset_prices: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.065,
    annualization_factor: int = 252,
) -> PortfolioAnalyticsResult:
    """
    Compute multi-asset portfolio returns, volatility, risk contributions,
    and correlation matrix.
    """
    if not asset_prices:
        raise ValueError("asset_prices dictionary cannot be empty.")

    # Combine price series into single dataframe
    price_df = pd.DataFrame(asset_prices).dropna().sort_index()
    if len(price_df) < 10:
        raise ValueError("Insufficient overlapping history across portfolio assets.")

    tickers = list(price_df.columns)
    n_assets = len(tickers)

    # Normalize weights
    if weights is None:
        raw_weights = {t: 1.0 / n_assets for t in tickers}
    else:
        # Keep only assets present in prices
        present_weights = {t: float(weights.get(t, 0.0)) for t in tickers}
        total_w = sum(present_weights.values())
        if total_w <= 0:
            raw_weights = {t: 1.0 / n_assets for t in tickers}
        else:
            raw_weights = {t: present_weights[t] / total_w for t in tickers}

    weight_vector = np.array([raw_weights[t] for t in tickers])

    # Asset daily returns
    returns_df = price_df.pct_change().dropna()
    cov_matrix = returns_df.cov() * annualization_factor
    corr_matrix = returns_df.corr()

    # Portfolio daily returns (rebalanced daily to fixed target weights)
    port_daily_ret = (returns_df * weight_vector).sum(axis=1)

    # Portfolio cumulative equity
    port_cum = (1.0 + port_daily_ret).cumprod()
    port_equity = initial_capital * port_cum

    # Performance metrics
    total_ret = float(port_cum.iloc[-1] - 1.0)
    years = max(len(returns_df) / annualization_factor, 0.05)
    cagr = float((1.0 + total_ret) ** (1.0 / years) - 1.0) if port_cum.iloc[-1] > 0 else -1.0

    # Portfolio variance and volatility
    port_variance = float(weight_vector.T @ cov_matrix.values @ weight_vector)
    port_vol = float(np.sqrt(max(port_variance, 1e-8)))

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0
    excess_ret = port_daily_ret - daily_rf
    sharpe = float((excess_ret.mean() / (port_daily_ret.std() + 1e-9)) * np.sqrt(annualization_factor))

    # Downside & Sortino
    downside = port_daily_ret[port_daily_ret < 0.0]
    downside_vol = float(downside.std() * np.sqrt(annualization_factor)) if len(downside) > 1 else port_vol
    sortino = float((cagr - risk_free_rate) / (downside_vol + 1e-9))

    # Max Drawdown
    peaks = np.maximum.accumulate(port_cum)
    drawdowns = (port_cum - peaks) / peaks
    max_dd = float(abs(drawdowns.min()))

    # Risk Attribution: Marginal & Component Contribution to Risk
    # sigma_p = sqrt(w' * Cov * w)
    # Marginal Risk Contribution MCR = (Cov * w) / sigma_p
    # Component Risk Contribution CCR = w * MCR
    cov_values = cov_matrix.values
    marginal_contrib = (cov_values @ weight_vector) / port_vol
    component_contrib = weight_vector * marginal_contrib
    pct_risk_contrib = component_contrib / port_vol

    risk_contrib_df = pd.DataFrame(
        {
            "Ticker": tickers,
            "Weight": [raw_weights[t] for t in tickers],
            "Annualized_Vol": returns_df.std().values * np.sqrt(annualization_factor),
            "Marginal_Risk_Contrib": marginal_contrib,
            "Component_Risk_Contrib": component_contrib,
            "Pct_Risk_Contribution": pct_risk_contrib,
        }
    ).sort_values(by="Pct_Risk_Contribution", ascending=False)

    metrics = {
        "portfolio_total_return": total_ret,
        "portfolio_cagr": cagr,
        "portfolio_volatility": port_vol,
        "portfolio_sharpe_ratio": sharpe,
        "portfolio_sortino_ratio": sortino,
        "portfolio_max_drawdown": max_dd,
        "calmar_ratio": float(cagr / (max_dd + 1e-9)),
    }

    assumptions = [
        "Daily rebalancing to target weights with zero friction costs.",
        "Dividends and corporate actions are pre-adjusted in price series.",
        "No fractional share execution constraints applied.",
        "Risk metrics assume stationary historical covariance structure.",
    ]

    return PortfolioAnalyticsResult(
        portfolio_equity=port_equity,
        daily_returns=port_daily_ret,
        asset_returns=returns_df,
        weights=raw_weights,
        metrics=metrics,
        correlation_matrix=corr_matrix,
        risk_contributions=risk_contrib_df,
        assumptions=assumptions,
    )

