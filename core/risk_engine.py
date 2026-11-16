"""Institutional Risk Analytics Engine: VaR, CVaR, Beta, and Rolling Dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RiskAnalyticsResult:
    """Comprehensive risk analytics container."""

    historical_var_95: float
    historical_var_99: float
    parametric_var_95: float
    parametric_var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    max_drawdown_duration_days: int
    beta: float
    alpha: float
    correlation_with_benchmark: float
    annualized_volatility: float
    rolling_volatility_20d: pd.Series
    rolling_sharpe_60d: pd.Series
    drawdown_series: pd.Series
    assumptions: list[str] = field(default_factory=list)


def compute_risk_analytics(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.065,
    annualization_factor: int = 252,
) -> RiskAnalyticsResult:
    """
    Compute rigorous risk metrics: Historical VaR, Parametric VaR, CVaR (Expected Shortfall),
    Drawdown analysis, and Benchmark Beta/Alpha.
    """
    clean_stock = stock_returns.dropna()
    if len(clean_stock) < 20:
        raise ValueError("At least 20 historical return periods required for risk modeling.")

    # Daily risk-free rate
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0

    # 1. Historical Value at Risk (VaR)
    # VaR is reported as a positive percentage loss
    hist_var_95 = float(-np.percentile(clean_stock, 5.0))
    hist_var_99 = float(-np.percentile(clean_stock, 1.0))

    # 2. Parametric (Gaussian) VaR
    mean_ret = float(clean_stock.mean())
    std_ret = float(clean_stock.std())
    param_var_95 = float(-(mean_ret - 1.64485 * std_ret))
    param_var_99 = float(-(mean_ret - 2.32635 * std_ret))

    # 3. Conditional Value at Risk (CVaR / Expected Shortfall)
    # Average loss given that loss exceeds the VaR threshold
    tail_95 = clean_stock[clean_stock <= -hist_var_95]
    cvar_95 = float(-tail_95.mean()) if len(tail_95) > 0 else hist_var_95

    tail_99 = clean_stock[clean_stock <= -hist_var_99]
    cvar_99 = float(-tail_99.mean()) if len(tail_99) > 0 else hist_var_99

    # 4. Drawdowns and duration
    cum_ret = (1.0 + clean_stock).cumprod()
    peaks = np.maximum.accumulate(cum_ret)
    dd_series = (cum_ret - peaks) / peaks
    max_dd = float(abs(dd_series.min()))

    # Calculate max drawdown duration (bars between peak and recovery)
    is_at_peak = dd_series == 0.0
    current_duration = 0
    max_duration = 0
    for at_peak in is_at_peak:
        if at_peak:
            current_duration = 0
        else:
            current_duration += 1
            if current_duration > max_duration:
                max_duration = current_duration

    # 5. Annualized Volatility
    ann_vol = float(std_ret * np.sqrt(annualization_factor))

    # 6. Rolling Volatility (20-day)
    roll_vol_20d = clean_stock.rolling(20).std() * np.sqrt(annualization_factor)

    # 7. Rolling Sharpe Ratio (60-day)
    roll_excess = clean_stock - daily_rf
    roll_sharpe_60d = (
        roll_excess.rolling(60).mean() / (clean_stock.rolling(60).std() + 1e-9)
    ) * np.sqrt(annualization_factor)

    # 8. Benchmark Beta and Alpha
    if benchmark_returns is not None and not benchmark_returns.empty:
        common_idx = clean_stock.index.intersection(benchmark_returns.dropna().index)
        if len(common_idx) > 10:
            s_aligned = clean_stock.reindex(common_idx)
            b_aligned = benchmark_returns.reindex(common_idx)

            cov = np.cov(s_aligned, b_aligned)[0, 1]
            var_b = np.var(b_aligned)
            beta = float(cov / (var_b + 1e-9))
            corr = float(stats.pearsonr(s_aligned, b_aligned)[0])

            # Jensen's Alpha annualized
            s_ann = float(s_aligned.mean() * annualization_factor)
            b_ann = float(b_aligned.mean() * annualization_factor)
            alpha = float(s_ann - (risk_free_rate + beta * (b_ann - risk_free_rate)))
        else:
            beta, alpha, corr = 1.0, 0.0, 1.0
    else:
        beta, alpha, corr = 1.0, 0.0, 1.0

    assumptions = [
        "Historical VaR/CVaR assume the empirical return distribution reflects forward tail risk.",
        "Parametric VaR assumes independent, identically distributed normal returns.",
        "Beta is estimated via ordinary least squares against the chosen benchmark.",
        f"Risk-free rate assumed at {risk_free_rate * 100:.2f}% per annum.",
    ]

    return RiskAnalyticsResult(
        historical_var_95=hist_var_95,
        historical_var_99=hist_var_99,
        parametric_var_95=param_var_95,
        parametric_var_99=param_var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_duration,
        beta=beta,
        alpha=alpha,
        correlation_with_benchmark=corr,
        annualized_volatility=ann_vol,
        rolling_volatility_20d=roll_vol_20d,
        rolling_sharpe_60d=roll_sharpe_60d,
        drawdown_series=dd_series,
        assumptions=assumptions,
    )

